/******************************************************************************
 * Copyright (c) 2025, xxx.
 ******************************************************************************/

#include <torch/python.h>
#include <torch/nn/functional.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include "philox_unpack.cuh"

#include <cutlass/numeric_types.h>

#include "hardware_info.h"
#include "flash.h"
#include "static_switch.h"
#include "compress_attention.h"
//#include "bwd_compress_attention.h"
#include <iostream>
#include <iomanip>
#include <string>
#include <numeric>


#define CHECK_DEVICE(x) TORCH_CHECK(x.is_cuda(), #x " must be on CUDA")
#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

void get_3d_from_4d_strides(const at::Tensor& t4d, int64_t& stride_h_ele, int64_t& stride_token_ele) {
    if (t4d.dim() == 4) {
        stride_h_ele = t4d.stride(2);
        stride_token_ele = t4d.stride(1);
    } else if (t4d.dim() == 3) {
        stride_h_ele = t4d.stride(1);
        stride_token_ele = t4d.stride(0);
    } else {
        TORCH_CHECK(false, "Input tensor for strides must be 3D or 4D");
    }
}


void set_params_fprop(Flash_fwd_params &params,
                      // sizes
                      const size_t b,
                      const size_t seqlen_q,
                      const size_t seqlen_k,
                      const size_t seqlen_q_rounded,
                      const size_t seqlen_k_rounded,
                      const size_t h,
                      const size_t h_k,
                      const size_t d,
                      const size_t d_rounded,
                      // device pointers
                      const at::Tensor q,
                      const at::Tensor k,
                      const at::Tensor v,
                      at::Tensor out,
                      void *cu_seqlens_q_d,
                      void *cu_seqlens_k_d,
                      void *seqused_k,
                      void *p_d,
                      void *softmax_lse_d,
                      float p_dropout,
                      float softmax_scale,
                      int window_size_left,
                      int window_size_right,
                      const float softcap,
                      bool seqlenq_ngroups_swapped=false,
                      const bool unpadded_lse=false) {

    // Reset the parameters
    params = {};

    params.is_bf16 = q.dtype() == torch::kBFloat16;

    // Set the pointers and strides.
    params.q_ptr = q.data_ptr();
    params.k_ptr = k.data_ptr();
    params.v_ptr = v.data_ptr();
    // All stride are in elements, not bytes.
    params.q_row_stride = q.stride(-3);
    params.k_row_stride = k.stride(-3);
    params.v_row_stride = v.stride(-3);
    params.q_head_stride = q.stride(-2);
    params.k_head_stride = k.stride(-2);
    params.v_head_stride = v.stride(-2);
    params.o_ptr = out.data_ptr();
    params.o_row_stride = out.stride(-3);
    params.o_head_stride = out.stride(-2);

    if (cu_seqlens_q_d == nullptr) {
        params.q_batch_stride = q.stride(0);
        params.k_batch_stride = k.stride(0);
        params.v_batch_stride = v.stride(0);
        params.o_batch_stride = out.stride(0);
        if (seqlenq_ngroups_swapped) {
             params.q_batch_stride *= seqlen_q;
             params.o_batch_stride *= seqlen_q;
        }
    }

    params.cu_seqlens_q = static_cast<int *>(cu_seqlens_q_d);
    params.cu_seqlens_k = static_cast<int *>(cu_seqlens_k_d);
    params.seqused_k = static_cast<int *>(seqused_k);

    // P = softmax(QK^T)
    params.p_ptr = p_d;

    // Softmax sum
    params.softmax_lse_ptr = softmax_lse_d;

    // Set the dimensions.
    params.b = b;
    params.h = h;
    params.h_k = h_k;
    params.h_h_k_ratio = h / h_k;
    params.seqlen_q = seqlen_q;
    params.seqlen_k = seqlen_k;
    params.seqlen_q_rounded = seqlen_q_rounded;
    params.seqlen_k_rounded = seqlen_k_rounded;
    params.d = d;
    params.d_rounded = d_rounded;

    // Set the different scale values.
    #ifdef FLASHATTENTION_DISABLE_SOFTCAP
        TORCH_CHECK(softcap <= 0.0, "This flash attention build does not support softcap.");
    #endif
    if (softcap > 0.0) {
        params.softcap = softmax_scale / softcap;
        params.scale_softmax = softcap;
        params.scale_softmax_log2 = softcap * M_LOG2E;
    } else{
        // Remove potential NaN
        params.softcap = 0.0;
        params.scale_softmax = softmax_scale;
        params.scale_softmax_log2 = softmax_scale * M_LOG2E;
    }

    // Set this to probability of keeping an element to simplify things.
    params.p_dropout = 1.f - p_dropout;
    params.p_dropout_in_uint8_t = uint8_t(std::floor(params.p_dropout * 255.0));
    params.rp_dropout = 1.f / params.p_dropout;
    params.scale_softmax_rp_dropout = params.rp_dropout * params.scale_softmax;
    TORCH_CHECK(p_dropout < 1.f);
    #ifdef FLASHATTENTION_DISABLE_DROPOUT
        TORCH_CHECK(p_dropout == 0.0f, "This flash attention build does not support dropout.");
    #endif

    // Causal is the special case where window_size_right == 0 and window_size_left < 0.
    // Local is the more general case where window_size_right >= 0 or window_size_left >= 0.
    params.is_causal = window_size_left < 0 && window_size_right == 0;

    if (window_size_left < 0 && window_size_right >= 0) { window_size_left = seqlen_k; }
    if (window_size_left >= 0 && window_size_right < 0) { window_size_right = seqlen_k; }
    params.window_size_left = window_size_left;
    params.window_size_right = window_size_right;

    #ifdef FLASHATTENTION_DISABLE_LOCAL
        TORCH_CHECK(params.is_causal || (window_size_left < 0 && window_size_right < 0),
            "This flash attention build does not support local attention.");
    #endif

    params.is_seqlens_k_cumulative = true;

    #ifdef FLASHATTENTION_DISABLE_UNEVEN_K
        TORCH_CHECK(d == d_rounded, "This flash attention build does not support headdim not being a multiple of 32.");
    #endif

    params.unpadded_lse = unpadded_lse;
    params.seqlenq_ngroups_swapped = seqlenq_ngroups_swapped;
}

/*void set_params_dgrad(Flash_bwd_params &params,
                      // sizes
                      const size_t b,
                      const size_t seqlen_q,
                      const size_t seqlen_k,
                      const size_t seqlen_q_rounded,
                      const size_t seqlen_k_rounded,
                      const size_t h,
                      const size_t h_k,
                      const size_t d,
                      const size_t d_rounded,
                      // device pointers
                      const at::Tensor q,
                      const at::Tensor k,
                      const at::Tensor v,
                      const at::Tensor out,
                      const at::Tensor dout,
                      at::Tensor dq,
                      at::Tensor dk,
                      at::Tensor dv,
                      void *cu_seqlens_q_d,
                      void *cu_seqlens_k_d,
                      void *dq_accum_d,
                      void *dk_accum_d,
                      void *dv_accum_d,
                      void *softmax_lse_d,
                      void *dsoftmax_sum_d,
                      float p_dropout,
                      float softmax_scale,
                      int window_size_left,
                      int window_size_right,
                      const float softcap,
                      bool deterministic,
                      const bool unpadded_lse) {

    set_params_fprop(params,
                     b, seqlen_q, seqlen_k, seqlen_q_rounded, seqlen_k_rounded, h, h_k, d, d_rounded,
                     q, k, v, out,
                     cu_seqlens_q_d,
                     cu_seqlens_k_d,
                     nullptr,
                     nullptr,
                     softmax_lse_d,
                     p_dropout,
                     softmax_scale,
                     window_size_left,
                     window_size_right,
                     softcap,
                     false, // seqlenq_ngroups_swapped
                     unpadded_lse);

    // Set the pointers and strides.
    params.do_ptr = dout.data_ptr();
    params.do_row_stride = dout.stride(-3);
    params.do_head_stride = dout.stride(-2);
    params.dq_ptr = dq.data_ptr();
    params.dk_ptr = dk.data_ptr();
    params.dv_ptr = dv.data_ptr();
    params.dq_row_stride = dq.stride(-3);
    params.dk_row_stride = dk.stride(-3);
    params.dv_row_stride = dv.stride(-3);
    params.dq_head_stride = dq.stride(-2);
    params.dk_head_stride = dk.stride(-2);
    params.dv_head_stride = dv.stride(-2);

    if (cu_seqlens_q_d == nullptr) {
        params.do_batch_stride = dout.stride(0);
        params.dq_batch_stride = dq.stride(0);
        params.dk_batch_stride = dk.stride(0);
        params.dv_batch_stride = dv.stride(0);
    }

    params.dq_accum_ptr = dq_accum_d;
    params.dk_accum_ptr = dk_accum_d;
    params.dv_accum_ptr = dv_accum_d;

    // Softmax sum
    params.dsoftmax_sum = dsoftmax_sum_d;

    params.deterministic = deterministic;
}
*/
// C++ helper function for rectangular FA in compressed attention
void run_mha_fwd(Flash_fwd_params &params, cudaStream_t stream, bool force_split_kernel=false) {
    TORCH_CHECK(params.compress_attention, "run_mha_fwd (C++ helper) is now only for compress_attention rect FA.");
    TORCH_CHECK(params.is_causal, "For compressed attention's rectangular FA, params.is_causal must be true at this point.");
    TORCH_CHECK(params.num_splits <= 1 && !force_split_kernel,
                "Compressed attention path does not support num_splits > 1 or force_split_kernel=true for its rectangular FA step.");
    TORCH_CHECK(params.p_dropout == 0.0f, "Dropout is not supported in the rectangular FA step of compressed attention.");
    TORCH_CHECK(params.alibi_slopes_ptr == nullptr, "ALiBi is not supported in the rectangular FA step of compressed attention.");

    FP16_SWITCH(!params.is_bf16, [&] {
        HEADDIM_SWITCH(params.d, [&] {
            run_mha_fwd_<elem_type, kHeadDim, /*Is_causal_tpl=*/true>(params, stream);
        });
    });
}



// Find the number of splits that maximizes the occupancy
inline int num_splits_heuristic(int batch_nheads_mblocks, int num_SMs, int num_n_blocks, int max_splits) {
    // If we have enough to almost fill the SMs, then just use 1 split
    if (batch_nheads_mblocks >= 0.8f * num_SMs) { return 1; }
    max_splits = std::min({max_splits, num_SMs, num_n_blocks});
    float max_efficiency = 0.f;
    std::vector<float> efficiency;
    efficiency.reserve(max_splits);
    auto ceildiv = [](int a, int b) { return (a + b - 1) / b; };
    auto is_split_eligible = [&ceildiv, &num_n_blocks](int num_splits) {
        return num_splits == 1 || ceildiv(num_n_blocks, num_splits) != ceildiv(num_n_blocks, num_splits - 1);
    };
    for (int num_splits = 1; num_splits <= max_splits; num_splits++) {
        if (!is_split_eligible(num_splits)) {
            efficiency.push_back(0.f);
        } else {
            float n_waves = float(batch_nheads_mblocks * num_splits) / num_SMs;
            float eff = n_waves / ceil(n_waves);
            if (eff > max_efficiency) { max_efficiency = eff; }
            efficiency.push_back(eff);
        }
    }
    for (int num_splits = 1; num_splits <= max_splits; num_splits++) {
        if (!is_split_eligible(num_splits)) { continue; }
        if (efficiency[num_splits - 1] >= 0.85 * max_efficiency) {
            return num_splits;
        }
    }
    return 1;
}

std::tuple<at::Tensor, at::Tensor> set_params_splitkv(Flash_fwd_params &params, const int batch_size,
    const int num_heads, const int head_size, const int max_seqlen_k, const int max_seqlen_q,
    const int head_size_rounded, const float p_dropout,
    const int num_splits, const int num_sm, struct c10::TensorOptions opts) {

    // This needs to match with run_mha_fwd_splitkv_dispatch
    const int block_n = head_size <= 64 ? 256 : (head_size <= 128 ? 128 : 64);
    const int num_n_blocks = (max_seqlen_k + block_n - 1) / block_n;
    // Technically kBlockM = 64 only for the splitKV kernels, not the standard kernel.
    const int num_m_blocks = (max_seqlen_q + 64 - 1) / 64;
    params.num_splits = num_splits;
    at::Tensor softmax_lse_accum;
    at::Tensor out_accum;

    if (p_dropout == 0.0f) {  // SplitKV is not implemented for dropout
        if (num_splits < 1) {
            // We multiply number of SMs by 2 to hard-code the fact that we're using 128 threads per block.
            params.num_splits = num_splits_heuristic(batch_size * num_heads * num_m_blocks, num_sm * 2, num_n_blocks, 128);
        }
        if (params.num_splits > 1) {
            softmax_lse_accum = torch::empty({params.num_splits, batch_size, num_heads, max_seqlen_q}, opts.dtype(at::kFloat));
            out_accum = torch::empty({params.num_splits, batch_size, num_heads, max_seqlen_q, head_size_rounded}, opts.dtype(at::kFloat));
            params.softmax_lseaccum_ptr = softmax_lse_accum.data_ptr();
            params.oaccum_ptr = out_accum.data_ptr();
        }
        TORCH_CHECK(params.num_splits <= 128, "num_splits > 128 not supported");
    }

    return std::make_tuple(softmax_lse_accum, out_accum);
}

void set_params_alibi(Flash_fwd_params &params, c10::optional<at::Tensor> &alibi_slopes_, int batch_size, int num_heads){
#ifdef FLASHATTENTION_DISABLE_ALIBI
    TORCH_CHECK(!alibi_slopes_.has_value(), "This flash attention build does not support alibi.");
    params.alibi_slopes_ptr = nullptr;
#else
    if (alibi_slopes_.has_value()) {
        auto alibi_slopes = alibi_slopes_.value();
        TORCH_CHECK(alibi_slopes.dtype() == torch::kFloat32, "ALiBi slopes must have dtype fp32");
        CHECK_DEVICE(alibi_slopes);
        TORCH_CHECK(alibi_slopes.stride(-1) == 1, "ALiBi slopes tensor must have contiguous last dimension");
        TORCH_CHECK(alibi_slopes.sizes() == torch::IntArrayRef({num_heads}) || alibi_slopes.sizes() == torch::IntArrayRef({batch_size, num_heads}));
        params.alibi_slopes_ptr = alibi_slopes.data_ptr();
        params.alibi_slopes_batch_stride = alibi_slopes.dim() == 2 ? alibi_slopes.stride(0) : 0;
    } else {
        params.alibi_slopes_ptr = nullptr;
    }
#endif
}

/*void run_mha_bwd(Flash_bwd_params &params, cudaStream_t stream) {
    FP16_SWITCH(!params.is_bf16, [&] {
        HEADDIM_SWITCH(params.d, [&] {
            BOOL_SWITCH(params.is_causal, Is_causal, [&] {
                run_mha_bwd_<elem_type, kHeadDim, Is_causal>(params, stream);
            });
        });
    });
}*/


// STAGE 1: METADATA CALCULATION (PRODUCTION VERSION)
std::tuple<at::Tensor, at::Tensor, int, int, int, int>
get_mha_varlen_fwd_compress_metadata(
    const at::Tensor &q_comp,
    const at::Tensor &k_comp,
    const at::Tensor &cu_seqlens_q_orig,
    const at::Tensor &cu_seqlens_k_orig,
    const at::Tensor &merge_indices,
    const at::Tensor &num_first_compress,
    const int max_seqlen_k
) {
    at::cuda::CUDAGuard device_guard{q_comp.device()};
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    CHECK_CONTIGUOUS(cu_seqlens_q_orig);
    CHECK_CONTIGUOUS(cu_seqlens_k_orig);
    CHECK_CONTIGUOUS(merge_indices);
    CHECK_CONTIGUOUS(num_first_compress);

    const int original_batch_size = cu_seqlens_q_orig.numel() - 1;
    const int original_num_heads = merge_indices.size(1);
    const int pseudo_batch_size = original_batch_size * original_num_heads;
    const auto opts_int32 = torch::dtype(torch::kInt32).device(q_comp.device());

    if (pseudo_batch_size == 0) {
        at::Tensor empty_cu_q = torch::zeros({1}, opts_int32);
        at::Tensor empty_cu_kv = torch::zeros({1}, opts_int32);
        return std::make_tuple(empty_cu_q, empty_cu_kv, 0, 0, 0, 0);
    }

    Flash_fwd_params params{};
    params.b = original_batch_size;
    params.h = original_num_heads;
    params.seqlen_k = max_seqlen_k;
    params.cu_seqlens_q = static_cast<int *>(cu_seqlens_q_orig.data_ptr());
    params.cu_seqlens_k = static_cast<int *>(cu_seqlens_k_orig.data_ptr());
    params.merge_indices_ptr = const_cast<void*>(static_cast<const void*>(merge_indices.data_ptr()));
    params.merge_indices_batch_stride = merge_indices.stride(0);
    params.merge_indices_head_stride = merge_indices.stride(1);
    params.num_first_compress_ptr = static_cast<const int*>(num_first_compress.data_ptr());

    at::Tensor d_per_bh_counts_q_prime = torch::empty({pseudo_batch_size}, opts_int32);
    at::Tensor d_per_bh_counts_kv_prime = torch::empty({pseudo_batch_size}, opts_int32);

    run_precompute_prime_lengths(params, stream, d_per_bh_counts_q_prime.data_ptr<int>(), d_per_bh_counts_kv_prime.data_ptr<int>());

    at::Tensor total_q_prime_tensor = torch::empty({1}, opts_int32);
    at::Tensor total_k_prime_tensor = torch::empty({1}, opts_int32);
    at::Tensor d_max_q_prime_tensor = torch::empty({1}, opts_int32);
    at::Tensor d_max_k_prime_tensor = torch::empty({1}, opts_int32);
    at::Tensor cu_seqlens_q_prime_final = torch::empty({pseudo_batch_size + 1}, opts_int32);
    at::Tensor cu_seqlens_kv_prime_final = torch::empty({pseudo_batch_size + 1}, opts_int32);

    size_t temp_storage_bytes_scan = 0;
    run_scan_for_prime_seqlens(
        nullptr, temp_storage_bytes_scan,
        d_per_bh_counts_q_prime.data_ptr<int>(), d_per_bh_counts_kv_prime.data_ptr<int>(),
        cu_seqlens_q_prime_final.data_ptr<int>(), cu_seqlens_kv_prime_final.data_ptr<int>(),
        total_q_prime_tensor.data_ptr<int>(), total_k_prime_tensor.data_ptr<int>(),
        pseudo_batch_size, stream
    );
    size_t temp_storage_bytes_max = 0;
    run_reduce_max_for_prime_seqlens(
        nullptr, temp_storage_bytes_max,
        d_per_bh_counts_q_prime.data_ptr<int>(), d_per_bh_counts_kv_prime.data_ptr<int>(),
        d_max_q_prime_tensor.data_ptr<int>(), d_max_k_prime_tensor.data_ptr<int>(),
        pseudo_batch_size, stream
    );

    size_t temp_storage_bytes = std::max(temp_storage_bytes_scan, temp_storage_bytes_max);
    at::Tensor d_temp_storage = torch::empty({static_cast<int64_t>(temp_storage_bytes)}, q_comp.options().dtype(torch::kUInt8));

    run_scan_for_prime_seqlens(
        d_temp_storage.data_ptr(), temp_storage_bytes,
        d_per_bh_counts_q_prime.data_ptr<int>(), d_per_bh_counts_kv_prime.data_ptr<int>(),
        cu_seqlens_q_prime_final.data_ptr<int>(), cu_seqlens_kv_prime_final.data_ptr<int>(),
        total_q_prime_tensor.data_ptr<int>(), total_k_prime_tensor.data_ptr<int>(),
        pseudo_batch_size, stream
    );
    run_reduce_max_for_prime_seqlens(
        d_temp_storage.data_ptr(), temp_storage_bytes,
        d_per_bh_counts_q_prime.data_ptr<int>(), d_per_bh_counts_kv_prime.data_ptr<int>(),
        d_max_q_prime_tensor.data_ptr<int>(), d_max_k_prime_tensor.data_ptr<int>(),
        pseudo_batch_size, stream
    );

    return std::make_tuple(
        cu_seqlens_q_prime_final,
        cu_seqlens_kv_prime_final,
        total_q_prime_tensor.item<int>(),
        total_k_prime_tensor.item<int>(),
        d_max_q_prime_tensor.item<int>(),
        d_max_k_prime_tensor.item<int>()
    );
}

// STAGE 2: MAIN FORWARD COMPUTATION (PRODUCTION VERSION)
std::vector<at::Tensor>
mha_varlen_fwd_compress(
    at::Tensor &q_comp,
    const at::Tensor &k_comp,
    const at::Tensor &v_comp,
    c10::optional<at::Tensor> &out_,
    const at::Tensor &cu_seqlens_q_orig,
    const at::Tensor &cu_seqlens_k_orig,
    const at::Tensor &merge_indices,
    const at::Tensor &num_first_compress,
    const float softmax_scale,
    const int max_seqlen_q,
    const int max_seqlen_k,
    const int total_q_prime,
    const int total_k_prime,
    const int max_seqlen_q_prime,
    const int max_seqlen_k_prime,
    at::Tensor &q_prime_data,
    at::Tensor &k_prime_data,
    at::Tensor &v_prime_data,
    at::Tensor &o_temp_data,
    at::Tensor &m_prime_data,
    at::Tensor &l_prime_data,
    at::Tensor &s_k_all_data,
    at::Tensor &cu_seqlens_q_prime,
    at::Tensor &cu_seqlens_kv_prime,
    at::Tensor &row_remapping_q_prime,
    at::Tensor &col_remapping_k_prime,
    at::Tensor &v_remapping_prime,
    at::Tensor &inv_row_remapping_q_prime,
    at::Tensor &q_is_participated_mask,
    at::Tensor &lse_prime_data
) {
    at::cuda::CUDAGuard device_guard{q_comp.device()};
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    auto [cc_major, cc_minor] = get_compute_capability(get_current_device());
    TORCH_CHECK(cc_major >= 8, "Compressed FlashAttention only supports Ampere GPUs or newer.");

    auto q_dtype = q_comp.dtype();
    TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16, "Compressed FlashAttention only supports fp16 and bf16 data types.");

    const int original_batch_size = cu_seqlens_q_orig.numel() - 1;
    const int total_q_elements_orig_runtime = cu_seqlens_q_orig.index({-1}).item<int>();
    const int original_num_heads = q_comp.size(1);
    const int head_size = q_comp.size(2);
    const int original_num_heads_k = k_comp.size(1);
    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    const int head_size_rounded_for_otemp = head_size <= 192 ? round_multiple(head_size, 32) : 256;

    at::Tensor out_final;
    if (out_.has_value()) {
        out_final = out_.value();
    } else {
        out_final = torch::empty({original_batch_size, max_seqlen_q, original_num_heads, head_size}, q_comp.options());
    }
    TORCH_CHECK(out_final.dim() == 4, "Output tensor must be 4D");


    auto opts_float = q_comp.options().dtype(at::kFloat);
    at::Tensor softmax_lse_final = torch::empty({std::max(0, total_q_elements_orig_runtime * original_num_heads)}, opts_float);

    if (total_q_elements_orig_runtime == 0) {
        out_final.zero_();
        return {out_final, softmax_lse_final, q_prime_data, k_prime_data, v_prime_data,
                o_temp_data, m_prime_data, l_prime_data, s_k_all_data,
                cu_seqlens_q_prime, cu_seqlens_kv_prime,
                row_remapping_q_prime, col_remapping_k_prime, v_remapping_prime,
                inv_row_remapping_q_prime, q_is_participated_mask,
                lse_prime_data, o_temp_data, l_prime_data
               };
    }

    // ======================= Stage 1: Fused Prime Tensors Generation =======================
    Flash_fwd_params params_gather{};
    params_gather.is_bf16 = (q_dtype == torch::kBFloat16);
    params_gather.b = original_batch_size;
    params_gather.h = original_num_heads;
    params_gather.h_k = original_num_heads_k;
    params_gather.d = head_size;
    params_gather.seqlen_k = max_seqlen_k;
    params_gather.q_ptr = q_comp.data_ptr();
    params_gather.k_ptr = k_comp.data_ptr();
    params_gather.v_ptr = v_comp.data_ptr();
    params_gather.cu_seqlens_q = cu_seqlens_q_orig.data_ptr<int>();
    params_gather.cu_seqlens_k = cu_seqlens_k_orig.data_ptr<int>();
    params_gather.merge_indices_ptr = const_cast<void*>(static_cast<const void*>(merge_indices.data_ptr()));
    params_gather.merge_indices_batch_stride = merge_indices.stride(0);
    params_gather.merge_indices_head_stride = merge_indices.stride(1);
    params_gather.num_first_compress_ptr = num_first_compress.data_ptr<int>();
    params_gather.cu_seqlens_q_prime_ptr = cu_seqlens_q_prime.data_ptr<int>();
    params_gather.cu_seqlens_kv_prime_ptr = cu_seqlens_kv_prime.data_ptr<int>();
    params_gather.q_row_stride = q_comp.stride(0);
    params_gather.q_head_stride = q_comp.stride(1);
    params_gather.k_row_stride = k_comp.stride(0);
    params_gather.k_head_stride = k_comp.stride(1);
    params_gather.v_row_stride = v_comp.stride(0);
    params_gather.v_head_stride = v_comp.stride(1);
    params_gather.q_prime_ptr = q_prime_data.data_ptr();
    params_gather.k_prime_ptr = k_prime_data.data_ptr();
    params_gather.v_prime_ptr = v_prime_data.data_ptr();
    params_gather.row_remapping_q_prime_ptr = row_remapping_q_prime.data_ptr<int>();
    params_gather.col_remapping_k_prime_ptr = col_remapping_k_prime.data_ptr<int>();
    params_gather.v_remapping_prime_ptr = v_remapping_prime.data_ptr<int>();
    params_gather.inv_row_remapping_q_prime_ptr = inv_row_remapping_q_prime.data_ptr<int>();
    params_gather.q_is_participated_mask_ptr = q_is_participated_mask.data_ptr<bool>();

    run_fused_gather_kernel(params_gather, stream);

    // ======================= Stage 2: Rectangular Attention on Prime Tensors =======================
    if (total_q_prime > 0 && total_k_prime > 0) {
        Flash_fwd_params params_rect{};
        params_rect.is_bf16 = (q_dtype == torch::kBFloat16);
        params_rect.b = original_batch_size * original_num_heads; // pseudo-batch
        params_rect.h = 1;
        params_rect.h_k = 1;
        params_rect.h_h_k_ratio = 1;
        params_rect.seqlen_q = max_seqlen_q_prime;
        params_rect.seqlen_k = max_seqlen_k_prime;
        params_rect.d = head_size;
        params_rect.d_rounded = head_size_rounded_for_otemp;
        params_rect.scale_softmax = softmax_scale;
        params_rect.scale_softmax_log2 = softmax_scale * M_LOG2E;
        params_rect.cu_seqlens_q = cu_seqlens_q_prime.data_ptr<int>();
        params_rect.cu_seqlens_k = cu_seqlens_kv_prime.data_ptr<int>();
        params_rect.q_ptr = q_prime_data.data_ptr();
        params_rect.k_ptr = k_prime_data.data_ptr();
        params_rect.v_ptr = v_prime_data.data_ptr();
        params_rect.o_ptr = o_temp_data.data_ptr();
        params_rect.softmax_lse_ptr = m_prime_data.data_ptr();
        params_rect.l_prime_ptr = l_prime_data.data_ptr();
        params_rect.q_row_stride = head_size;
        params_rect.k_row_stride = head_size;
        params_rect.v_row_stride = head_size;
        params_rect.o_row_stride = head_size_rounded_for_otemp;
        params_rect.is_causal = true;
        params_rect.compress_attention = true;

        run_mha_fwd(params_rect, stream);
    }

    run_compute_lse_prime_kernel(
        m_prime_data.data_ptr<float>(),
        l_prime_data.data_ptr<float>(),
        lse_prime_data.data_ptr<float>(),
        total_q_prime, stream, softmax_scale
    );

    // ======================= Stage 3 & 4: Diagonal Score Calculation & Final Combine =======================
    Flash_fwd_params params_combine{};
    params_combine.is_bf16 = (q_dtype == torch::kBFloat16);
    params_combine.b = original_batch_size;
    params_combine.h = original_num_heads;
    params_combine.h_k = original_num_heads_k;
    params_combine.d = head_size;
    params_combine.d_rounded = head_size_rounded_for_otemp;
    params_combine.scale_softmax = softmax_scale;
    params_combine.total_q = total_q_elements_orig_runtime;
    params_combine.q_ptr = q_comp.data_ptr();
    params_combine.k_ptr = k_comp.data_ptr();
    params_combine.v_ptr = v_comp.data_ptr();
    params_combine.o_ptr = out_final.data_ptr();
    params_combine.cu_seqlens_q = cu_seqlens_q_orig.data_ptr<int>();
    params_combine.cu_seqlens_k = cu_seqlens_k_orig.data_ptr<int>();
    params_combine.softmax_lse_ptr = softmax_lse_final.data_ptr();
    params_combine.q_batch_stride = 0; params_combine.q_head_stride = q_comp.stride(1); params_combine.q_row_stride = q_comp.stride(0);
    params_combine.k_batch_stride = 0; params_combine.k_head_stride = k_comp.stride(1); params_combine.k_row_stride = k_comp.stride(0);
    params_combine.v_batch_stride = 0; params_combine.v_head_stride = v_comp.stride(1); params_combine.v_row_stride = v_comp.stride(0);
    params_combine.o_batch_stride = out_final.stride(0);
    params_combine.o_head_stride = out_final.stride(2);
    params_combine.o_row_stride = out_final.stride(1);
    params_combine.compress_attention = true;
    params_combine.inv_row_remapping_q_prime_ptr = inv_row_remapping_q_prime.data_ptr<int>();
    params_combine.o_temp_ptr = o_temp_data.data_ptr();
    params_combine.m_prime_ptr = m_prime_data.data_ptr();
    params_combine.l_prime_ptr = l_prime_data.data_ptr();
    params_combine.s_k_prime_ptr = s_k_all_data.data_ptr();

    auto opts_int = q_comp.options().dtype(torch::kInt32);
    at::Tensor q_orig_to_batch_idx_map = torch::empty({total_q_elements_orig_runtime}, opts_int);
    run_map_q_indices_to_batch(
        cu_seqlens_q_orig.data_ptr<int>(),
        q_orig_to_batch_idx_map.data_ptr<int>(),
        original_batch_size,
        total_q_elements_orig_runtime,
        stream
    );
    params_combine.q_orig_to_batch_idx_ptr_fwd = q_orig_to_batch_idx_map.data_ptr<int>();

    run_calculate_sk_prime_kernel_v2(params_combine, stream);
    run_compress_attn_combine_kernel_v2(params_combine, stream);

    return { out_final, softmax_lse_final,
            q_prime_data, k_prime_data, v_prime_data,
            o_temp_data,  // o_prime_fwd
            m_prime_data,  // m_prime_fwd
            l_prime_data,  // l_prime_fwd
            s_k_all_data,  // s_k_fwd
            lse_prime_data, // lse_prime_fwd
            cu_seqlens_q_prime, cu_seqlens_kv_prime,
            row_remapping_q_prime, col_remapping_k_prime, v_remapping_prime,
            inv_row_remapping_q_prime, q_is_participated_mask
    };
}





PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "SFAttention";
    m.def("get_mha_varlen_fwd_compress_metadata", &get_mha_varlen_fwd_compress_metadata,
        "Phase 1: Get metadata for compressed attention. Returns cu_seqlens_q_prime, cu_seqlens_kv_prime, and scalar metadata.", 
        py::arg("q_comp"),
        py::arg("k_comp"),
        py::arg("cu_seqlens_q_orig"),
        py::arg("cu_seqlens_k_orig"),
        py::arg("merge_indices"),
        py::arg("num_first_compress"),
        py::arg("max_seqlen_k"));
    m.def("varlen_fwd_compress", &mha_varlen_fwd_compress,
        "Phase 2: SFAttention variable-length forward with compression",
        py::arg("q_comp"),
        py::arg("k_comp"),
        py::arg("v_comp"),
        py::arg("out_"),
        py::arg("cu_seqlens_q_orig"),
        py::arg("cu_seqlens_k_orig"),
        py::arg("merge_indices"),
        py::arg("num_first_compress"),
        py::arg("softmax_scale"),
        py::arg("max_seqlen_q"),
        py::arg("max_seqlen_k"),
        py::arg("total_q_prime"),
        py::arg("total_k_prime"),
        py::arg("max_seqlen_q_prime"),
        py::arg("max_seqlen_k_prime"),
        py::arg("q_prime_data"),
        py::arg("k_prime_data"),
        py::arg("v_prime_data"),
        py::arg("o_temp_data"),
        py::arg("m_prime_data"),
        py::arg("l_prime_data"),
        py::arg("s_k_all_data"),
        py::arg("cu_seqlens_q_prime"),
        py::arg("cu_seqlens_kv_prime"),
        py::arg("row_remapping_q_prime"),
        py::arg("col_remapping_k_prime"),
        py::arg("v_remapping_prime"),
        py::arg("inv_row_remapping_q_prime"),
        py::arg("q_is_participated_mask"),
        py::arg("lse_prime_data"));
    //m.def("varlen_bwd_compress", &mha_varlen_bwd_compress,
    //    "FlashAttention-2 variable-length backward with compression",
    //    py::arg("dout"),
    //    py::arg("q_comp"),
    //    py::arg("k_comp"),
    //    py::arg("v_comp"),
    //    py::arg("out"), 
    //    py::arg("o_prime_fwd"),
    //    py::arg("m_prime_fwd"),
    //    py::arg("l_prime_fwd"),
    //    py::arg("s_k_fwd"),
    //    py::arg("lse_prime_fwd"),
    //    py::arg("q_prime_fwd"),
    //    py::arg("k_prime_fwd"),
    //    py::arg("v_prime_fwd"),
    //    py::arg("cu_seqlens_q_orig"),
    //    py::arg("cu_seqlens_k_orig"),
    //    py::arg("cu_seqlens_q_prime"),
    //    py::arg("cu_seqlens_kv_prime"),
    //    py::arg("row_remapping_q_prime"),
    //    py::arg("col_remapping_k_prime"),
    //    py::arg("v_remapping_prime"),
    //    py::arg("inv_row_remapping_q_prime"),
    //    py::arg("q_is_participated_mask"),
    //    py::arg("dq_"),
    //    py::arg("dk_"),
    //    py::arg("dv_"),
    //    py::arg("max_seqlen_q_orig"),
    //    py::arg("max_seqlen_k_orig"),
    //    py::arg("softmax_scale"),
    //    py::arg("deterministic"));
}
