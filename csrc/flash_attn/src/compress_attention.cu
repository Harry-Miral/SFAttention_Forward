/******************************************************************************
 * Copyright (c) 2025.
 * xxx
 * Implements CUDA kernels for compressed attention stages.
 ******************************************************************************/

#include <c10/cuda/CUDAException.h>
#include <ATen/core/TensorBody.h>
#include <ATen/ATen.h>
namespace at {
    using ::at::Layout;
}
#include "compress_attention.h"
#include "flash.h"
#include "kernel_traits.h"
#include "static_switch.h"
#include "utils.h"
#include "block_info.h"

#include <vector>
#include <numeric>
#include <algorithm>
#include <cuda_runtime.h>
#include <limits>
#include <cub/cub.cuh>
#include <cub/block/block_scan.cuh>
#include <cutlass/array.h>

template <typename ElementDummy, int BLOCK_THREADS_X>
__global__ void count_qkv_prime_lengths_kernel(
    const int* __restrict__ cu_seqlens_q_orig_ptr,
    const int* __restrict__ cu_seqlens_k_orig_ptr,
    const bool* __restrict__ merge_indices_ptr,
    const int* __restrict__ num_first_compress_ptr,
    int* __restrict__ d_per_bh_counts_q_prime_ptr,
    int* __restrict__ d_per_bh_counts_kv_prime_ptr,
    const int original_batch_size,
    const int original_num_heads,
    const int64_t merge_indices_stride_b,
    const int64_t merge_indices_stride_h,
    const int max_seq_len_k_orig_for_merge_idx
) {
    const int b_orig = blockIdx.x;
    const int h_orig = blockIdx.y;

    if (b_orig >= original_batch_size || h_orig >= original_num_heads) {
        return;
    }

    if (threadIdx.x != 0) {
        return;
    }

    const int bh_flat_idx = b_orig * original_num_heads + h_orig;

    const int seq_start_q_orig = cu_seqlens_q_orig_ptr[b_orig];
    const int seq_end_q_orig   = cu_seqlens_q_orig_ptr[b_orig + 1];
    const int current_seq_len_q_orig = seq_end_q_orig - seq_start_q_orig;

    const int seq_start_k_orig = cu_seqlens_k_orig_ptr[b_orig];
    const int seq_end_k_orig   = cu_seqlens_k_orig_ptr[b_orig + 1];
    const int current_seq_len_k_orig = seq_end_k_orig - seq_start_k_orig;

    const int nfc_val = num_first_compress_ptr[bh_flat_idx];
    const bool* current_merge_indices_bh_start = merge_indices_ptr
                                         + b_orig * merge_indices_stride_b
                                         + h_orig * merge_indices_stride_h;

    int len_q_prime_bh = 0;
    if (current_seq_len_q_orig > 0) {
        int tokens_to_remove = nfc_val + 1;
        len_q_prime_bh = current_seq_len_q_orig - tokens_to_remove;
        if (len_q_prime_bh < 0) {
            len_q_prime_bh = 0;
        }
    }
    d_per_bh_counts_q_prime_ptr[bh_flat_idx] = len_q_prime_bh;

    volatile int len_kv_prime_bh = 0;

    if (current_seq_len_k_orig > 0) {
        volatile int survivor_k_count = 0;
        const volatile bool* volatile_merge_indices = current_merge_indices_bh_start;

        for (int i = 0; i < current_seq_len_k_orig; ++i) {
            bool is_compressed = false;

            if (i < nfc_val) {
                is_compressed = true;
            }
            else if (i < current_seq_len_k_orig - 1) {
                if (i < max_seq_len_k_orig_for_merge_idx - 1) {
                    bool merge_val = volatile_merge_indices[i];
                    if (merge_val) {
                        is_compressed = true;
                    }
                }
            }

            if (!is_compressed) {
                survivor_k_count = survivor_k_count + 1;
            }
        }

        if (survivor_k_count > 1) {
            len_kv_prime_bh = survivor_k_count - 1;
        }
    }

    d_per_bh_counts_kv_prime_ptr[bh_flat_idx] = max(0, (int)len_kv_prime_bh);

    __threadfence();
}

template <typename Element, int BLOCK_THREADS_X, int HEAD_SIZE_TPL>
__global__ void gather_qkv_prime_kernel(
    const Element* __restrict__ q_comp_ptr, const Element* __restrict__ k_comp_ptr, const Element* __restrict__ v_comp_ptr,
    const int* __restrict__ cu_seqlens_q_orig_ptr, const int* __restrict__ cu_seqlens_k_orig_ptr,
    const bool* __restrict__ merge_indices_ptr, const int* __restrict__ num_first_compress_ptr,
    Element* __restrict__ q_prime_data_ptr, Element* __restrict__ k_prime_data_ptr, Element* __restrict__ v_prime_data_ptr,
    int* __restrict__ row_remapping_q_prime_ptr, int* __restrict__ col_remapping_k_prime_ptr, int* __restrict__ v_remapping_prime_ptr, int* __restrict__ inv_row_remapping_q_prime_ptr,
    const int* __restrict__ d_cu_seqlens_q_prime_ptr, const int* __restrict__ d_cu_seqlens_kv_prime_ptr,
    const int original_batch_size, const int original_num_heads, const int original_num_heads_k,
    const int64_t q_comp_stride_b_ele, const int64_t q_comp_stride_h_ele, const int64_t q_comp_stride_s_ele,
    const int64_t k_comp_stride_b_ele, const int64_t k_comp_stride_h_ele, const int64_t k_comp_stride_s_ele,
    const int64_t v_comp_stride_b_ele, const int64_t v_comp_stride_h_ele, const int64_t v_comp_stride_s_ele,
    const int64_t merge_indices_stride_b, const int64_t merge_indices_stride_h,
    const int total_q_elements_orig_runtime,
    const int max_seq_len_k_orig_for_merge_idx,
    bool* __restrict__ q_is_participated_mask_ptr
    ) {

    typedef cub::BlockScan<int, BLOCK_THREADS_X> BlockScan;
    __shared__ typename BlockScan::TempStorage temp_storage;

    const int b_orig = blockIdx.x;
    const int h_q_orig = blockIdx.y;

    if (b_orig >= original_batch_size || h_q_orig >= original_num_heads) { return; }

    const int h_kv_orig = h_q_orig / (original_num_heads / original_num_heads_k);
    const int bh_flat_idx = b_orig * original_num_heads + h_q_orig;

    const int seq_start_q_orig_batch_flat = cu_seqlens_q_orig_ptr[b_orig];
    const int current_seq_len_q_orig = cu_seqlens_q_orig_ptr[b_orig + 1] - seq_start_q_orig_batch_flat;

    const int seq_start_k_orig_batch_flat = cu_seqlens_k_orig_ptr[b_orig];
    const int current_seq_len_k_orig = cu_seqlens_k_orig_ptr[b_orig + 1] - seq_start_k_orig_batch_flat;

    const int nfc_val = num_first_compress_ptr[bh_flat_idx];
    const bool* current_merge_indices_bh_start = merge_indices_ptr
                                         + b_orig * merge_indices_stride_b
                                         + h_q_orig * merge_indices_stride_h;

    for (int i_chunk_start = 0; i_chunk_start < current_seq_len_q_orig; i_chunk_start += BLOCK_THREADS_X) {
        const int i_local_q = i_chunk_start + threadIdx.x;
        if (i_local_q < current_seq_len_q_orig) {
            const int original_q_global_flat_idx = seq_start_q_orig_batch_flat + i_local_q;
            const int mask_access_idx = original_q_global_flat_idx * original_num_heads + h_q_orig;
            q_is_participated_mask_ptr[mask_access_idx] = false;
        }
    }
    __syncthreads();

    const int q_prime_bh_global_write_offset = d_cu_seqlens_q_prime_ptr[bh_flat_idx];
    const int tokens_to_skip = nfc_val + 1;

    int q_prime_write_idx = 0;
    for (int i_chunk_start = 0; i_chunk_start < current_seq_len_q_orig; i_chunk_start += BLOCK_THREADS_X) {
        const int i_local_q = i_chunk_start + threadIdx.x;

        bool should_include_in_q_prime = false;
        if (i_local_q < current_seq_len_q_orig) {
            if (i_local_q >= tokens_to_skip) {
                should_include_in_q_prime = true;
            }
        }

        int rank_in_chunk;
        int total_in_chunk;
        BlockScan(temp_storage).ExclusiveSum(should_include_in_q_prime ? 1 : 0, rank_in_chunk, total_in_chunk);

        if (should_include_in_q_prime) {
            const int q_prime_global_write_idx = q_prime_bh_global_write_offset + q_prime_write_idx + rank_in_chunk;
            const int original_q_global_flat_idx = seq_start_q_orig_batch_flat + i_local_q;
            const int access_idx = original_q_global_flat_idx * original_num_heads + h_q_orig;

            const Element* src_q_token_head_ptr = q_comp_ptr + (int64_t)original_q_global_flat_idx * q_comp_stride_s_ele + (int64_t)h_q_orig * q_comp_stride_h_ele;
            Element* dst_q_prime_token_ptr = q_prime_data_ptr + (int64_t)q_prime_global_write_idx * HEAD_SIZE_TPL;
            for (int d = 0; d < HEAD_SIZE_TPL; ++d) { dst_q_prime_token_ptr[d] = src_q_token_head_ptr[d]; }

            row_remapping_q_prime_ptr[q_prime_global_write_idx] = original_q_global_flat_idx;
            inv_row_remapping_q_prime_ptr[access_idx] = q_prime_global_write_idx;

            q_is_participated_mask_ptr[access_idx] = true;
        }

        __syncthreads();
        q_prime_write_idx += total_in_chunk;
        __syncthreads();
    }

    if (threadIdx.x == 0 && current_seq_len_k_orig > 0) {
        const int kv_prime_bh_global_write_offset = d_cu_seqlens_kv_prime_ptr[bh_flat_idx];
        const int kv_prime_bh_len_from_count = d_cu_seqlens_kv_prime_ptr[bh_flat_idx + 1] - kv_prime_bh_global_write_offset;

        if (kv_prime_bh_len_from_count > 0) {
            constexpr int MAX_SEQ_LEN_K_GATHER_TEMP_CU_IMPL = 4096;
            __shared__ int survivor_k_local_indices[MAX_SEQ_LEN_K_GATHER_TEMP_CU_IMPL];
            int num_survivors = 0;

            for (int i = 0; i < current_seq_len_k_orig; ++i) {
                bool is_compressed = false;

                if (i < nfc_val) {
                    is_compressed = true;
                }
                else if (i < current_seq_len_k_orig - 1) {
                    if (i < max_seq_len_k_orig_for_merge_idx - 1) {
                        if (current_merge_indices_bh_start[i]) {
                            is_compressed = true;
                        }
                    }
                }

                if (!is_compressed) {
                    if (num_survivors < MAX_SEQ_LEN_K_GATHER_TEMP_CU_IMPL) {
                        survivor_k_local_indices[num_survivors++] = i;
                    } else {
                        break;
                    }
                }
            }

            for (int rank_k_prime = 0; rank_k_prime < kv_prime_bh_len_from_count && rank_k_prime < num_survivors - 1; ++rank_k_prime) {
                const int local_k_idx_orig = survivor_k_local_indices[rank_k_prime];
                const int original_k_global_flat_idx = seq_start_k_orig_batch_flat + local_k_idx_orig;
                const int k_prime_global_write_idx = kv_prime_bh_global_write_offset + rank_k_prime;

                const Element* src_k_token_head_ptr = k_comp_ptr + (int64_t)original_k_global_flat_idx * k_comp_stride_s_ele + (int64_t)h_kv_orig * k_comp_stride_h_ele;
                Element* dst_k_prime_token_ptr = k_prime_data_ptr + (int64_t)k_prime_global_write_idx * HEAD_SIZE_TPL;
                for (int d = 0; d < HEAD_SIZE_TPL; ++d) {
                    dst_k_prime_token_ptr[d] = src_k_token_head_ptr[d];
                }
                col_remapping_k_prime_ptr[k_prime_global_write_idx] = original_k_global_flat_idx;
            }

            for (int rank_v_prime = 0; rank_v_prime < kv_prime_bh_len_from_count && rank_v_prime + 1 < num_survivors; ++rank_v_prime) {
                const int local_v_idx_orig = survivor_k_local_indices[rank_v_prime + 1];
                const int v_prime_global_write_idx = kv_prime_bh_global_write_offset + rank_v_prime;
                const int original_v_global_flat_idx = seq_start_k_orig_batch_flat + local_v_idx_orig;

                const Element* src_v_token_head_ptr = v_comp_ptr + (int64_t)original_v_global_flat_idx * v_comp_stride_s_ele + (int64_t)h_kv_orig * v_comp_stride_h_ele;
                Element* dst_v_prime_token_ptr = v_prime_data_ptr + (int64_t)v_prime_global_write_idx * HEAD_SIZE_TPL;
                for (int d = 0; d < HEAD_SIZE_TPL; ++d) {
                    dst_v_prime_token_ptr[d] = src_v_token_head_ptr[d];
                }
                v_remapping_prime_ptr[v_prime_global_write_idx] = original_v_global_flat_idx;
            }
        }
    }
}

void calculate_qkv_prime_metadata(Flash_fwd_params &params, cudaStream_t stream) {
    const int original_batch_size = params.b;
    const int original_num_heads = params.h;
    const int max_seq_len_k_orig_for_merge_idx = params.seqlen_k;

    const int block_dim_x_counts = 256;

    int* d_per_bh_counts_q_prime_device_ptr = nullptr;
    int* d_per_bh_counts_kv_prime_device_ptr = nullptr;
    const int num_bh_items = original_batch_size * original_num_heads;
    size_t counts_buffer_size = static_cast<size_t>(num_bh_items) * sizeof(int);

    TORCH_CHECK(params.cu_seqlens_q_prime_ptr != nullptr, "params.cu_seqlens_q_prime_ptr must be allocated before calling calculate_qkv_prime_metadata");
    TORCH_CHECK(params.cu_seqlens_kv_prime_ptr != nullptr, "params.cu_seqlens_kv_prime_ptr must be allocated before calling calculate_qkv_prime_metadata");

    if (num_bh_items == 0) {
        compute_cumulative_lengths_from_per_bh_counts(nullptr, nullptr,
            params.cu_seqlens_q_prime_ptr, params.cu_seqlens_kv_prime_ptr,
            params.total_q_prime, params.total_k_prime,
            params.max_seqlen_q_prime, params.max_seqlen_k_prime,
            original_batch_size, original_num_heads, stream);
        return;
    }

    C10_CUDA_CHECK(cudaMallocAsync(&d_per_bh_counts_q_prime_device_ptr, counts_buffer_size, stream));
    C10_CUDA_CHECK(cudaMallocAsync(&d_per_bh_counts_kv_prime_device_ptr, counts_buffer_size, stream));

    dim3 grid_counts(original_batch_size, original_num_heads);
    dim3 block_counts(block_dim_x_counts);

    const int64_t merge_indices_stride_b_ele = params.merge_indices_batch_stride;
    const int64_t merge_indices_stride_h_ele = params.merge_indices_head_stride;

    FP16_SWITCH(!params.is_bf16, [&] {
        count_qkv_prime_lengths_kernel<elem_type, block_dim_x_counts><<<grid_counts, block_counts, 0, stream>>>(
            params.cu_seqlens_q, params.cu_seqlens_k,
            static_cast<const bool*>(params.merge_indices_ptr),
            params.num_first_compress_ptr,
            d_per_bh_counts_q_prime_device_ptr, d_per_bh_counts_kv_prime_device_ptr,
            original_batch_size, original_num_heads,
            merge_indices_stride_b_ele, merge_indices_stride_h_ele,
            max_seq_len_k_orig_for_merge_idx);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    });

    compute_cumulative_lengths_from_per_bh_counts(
        d_per_bh_counts_q_prime_device_ptr, d_per_bh_counts_kv_prime_device_ptr,
        params.cu_seqlens_q_prime_ptr, params.cu_seqlens_kv_prime_ptr,
        params.total_q_prime, params.total_k_prime,
        params.max_seqlen_q_prime, params.max_seqlen_k_prime,
        original_batch_size, original_num_heads, stream);

    C10_CUDA_CHECK(cudaFreeAsync(d_per_bh_counts_q_prime_device_ptr, stream));
    C10_CUDA_CHECK(cudaFreeAsync(d_per_bh_counts_kv_prime_device_ptr, stream));
}

void gather_qkv_prime_data(Flash_fwd_params &params, cudaStream_t stream) {
    const int original_batch_size = params.b;
    const int original_num_heads = params.h;
    const int original_num_heads_k = params.h_k;
    const int head_size = params.d;
    const int max_seq_len_k_orig_for_merge_idx = params.seqlen_k;

    const int block_dim_x_gather = 256;

    if (params.total_q * params.h > 0 && params.inv_row_remapping_q_prime_ptr != nullptr) {
        C10_CUDA_CHECK(cudaMemsetAsync(params.inv_row_remapping_q_prime_ptr, 0xFF, params.total_q * params.h * sizeof(int), stream));
    } else if (params.total_q * params.h > 0 && params.inv_row_remapping_q_prime_ptr == nullptr) {
        TORCH_CHECK(false, "params.inv_row_remapping_q_prime_ptr is null but params.total_q > 0");
    }

    const int64_t q_comp_stride_b_ele_gather = params.q_batch_stride;
    const int64_t q_comp_stride_h_ele_gather = params.q_head_stride;
    const int64_t q_comp_stride_s_ele_gather = params.q_row_stride;

    const int64_t k_comp_stride_b_ele_gather = params.k_batch_stride;
    const int64_t k_comp_stride_h_ele_gather = params.k_head_stride;
    const int64_t k_comp_stride_s_ele_gather = params.k_row_stride;

    const int64_t v_comp_stride_b_ele_gather = params.v_batch_stride;
    const int64_t v_comp_stride_h_ele_gather = params.v_head_stride;
    const int64_t v_comp_stride_s_ele_gather = params.v_row_stride;

    const int64_t merge_indices_stride_b_ele = params.merge_indices_batch_stride;
    const int64_t merge_indices_stride_h_ele = params.merge_indices_head_stride;

    if (params.total_q_prime > 0 || params.total_k_prime > 0) {
        TORCH_CHECK(params.q_prime_ptr != nullptr, "params.q_prime_ptr cannot be null for gather stage if total_q_prime > 0");
        TORCH_CHECK(params.k_prime_ptr != nullptr, "params.k_prime_ptr cannot be null for gather stage if total_k_prime > 0");
        TORCH_CHECK(params.v_prime_ptr != nullptr, "params.v_prime_ptr cannot be null for gather stage if total_k_prime > 0");
        TORCH_CHECK(params.row_remapping_q_prime_ptr != nullptr, "params.row_remapping_q_prime_ptr cannot be null for gather stage");
        TORCH_CHECK(params.col_remapping_k_prime_ptr != nullptr, "params.col_remapping_k_prime_ptr cannot be null for gather stage");
        TORCH_CHECK(params.v_remapping_prime_ptr != nullptr, "params.v_remapping_prime_ptr cannot be null for gather stage");
        TORCH_CHECK(params.q_is_participated_mask_ptr != nullptr, "params.q_is_participated_mask_ptr cannot be null for gather stage");

        dim3 grid_gather(original_batch_size, original_num_heads);
        dim3 block_gather(block_dim_x_gather);

        FP16_SWITCH(!params.is_bf16, [&] {
            HEADDIM_SWITCH(head_size, [&] {
                gather_qkv_prime_kernel<elem_type, block_dim_x_gather, kHeadDim><<<grid_gather, block_gather, 0, stream>>>(
                    static_cast<const elem_type*>(params.q_ptr), static_cast<const elem_type*>(params.k_ptr), static_cast<const elem_type*>(params.v_ptr),
                    params.cu_seqlens_q, params.cu_seqlens_k,
                    static_cast<const bool*>(params.merge_indices_ptr), params.num_first_compress_ptr,
                    static_cast<elem_type*>(params.q_prime_ptr), static_cast<elem_type*>(params.k_prime_ptr), static_cast<elem_type*>(params.v_prime_ptr),
                    const_cast<int*>(params.row_remapping_q_prime_ptr),
                    const_cast<int*>(params.col_remapping_k_prime_ptr),
                    const_cast<int*>(params.v_remapping_prime_ptr),
                    params.inv_row_remapping_q_prime_ptr,
                    params.cu_seqlens_q_prime_ptr, params.cu_seqlens_kv_prime_ptr,
                    original_batch_size, original_num_heads, original_num_heads_k,
                    q_comp_stride_b_ele_gather, q_comp_stride_h_ele_gather, q_comp_stride_s_ele_gather,
                    k_comp_stride_b_ele_gather, k_comp_stride_h_ele_gather, k_comp_stride_s_ele_gather,
                    v_comp_stride_b_ele_gather, v_comp_stride_h_ele_gather, v_comp_stride_s_ele_gather,
                    merge_indices_stride_b_ele, merge_indices_stride_h_ele,
                    params.total_q,
                    max_seq_len_k_orig_for_merge_idx,
                    const_cast<bool*>(params.q_is_participated_mask_ptr)
                );
                C10_CUDA_KERNEL_LAUNCH_CHECK();
            });
        });
    }
}

template <typename Element, typename ElementAccum, int HEAD_SIZE_TPL>
__global__ void calculate_sk_prime_kernel_impl(
    const Element* __restrict__ q_comp_ptr_base,
    const Element* __restrict__ k_comp_ptr_base,
    ElementAccum* __restrict__ s_k_all_data_ptr,
    const float softmax_scale,
    const int total_q,
    const int original_batch_size_for_sk,
    const int num_heads_q_orig,
    const int num_heads_k_orig,
    const int64_t q_comp_stride_b_ele, const int64_t q_comp_stride_h_ele, const int64_t q_comp_stride_s_ele,
    const int64_t k_comp_stride_b_ele, const int64_t k_comp_stride_h_ele, const int64_t k_comp_stride_s_ele,
    const int* __restrict__ cu_seqlens_q_orig_ptr,
    const int* __restrict__ cu_seqlens_k_orig_ptr
) {
    const int idx_flat_q_x_head = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx_flat_q_x_head >= total_q * num_heads_q_orig) {
        return;
    }

    const int original_q_global_flat_idx = idx_flat_q_x_head / num_heads_q_orig;
    const int h_q_idx_orig = idx_flat_q_x_head % num_heads_q_orig;
    const int h_k_idx_orig = h_q_idx_orig / (num_heads_q_orig / num_heads_k_orig);

    const int original_k_global_flat_idx = original_q_global_flat_idx;

    int current_batch_idx_orig = 0;
    int local_row_idx_in_batch_orig = 0;
    int running_sum = 0;
    for (int b_scan = 0; b_scan < original_batch_size_for_sk; ++b_scan) {
        int len_q_b = cu_seqlens_q_orig_ptr[b_scan + 1] - cu_seqlens_q_orig_ptr[b_scan];
        if (original_q_global_flat_idx < running_sum + len_q_b) {
            current_batch_idx_orig = b_scan;
            local_row_idx_in_batch_orig = original_q_global_flat_idx - running_sum;
            break;
        }
        running_sum += len_q_b;
    }

    const int current_k_seq_len = cu_seqlens_k_orig_ptr[current_batch_idx_orig + 1] - cu_seqlens_k_orig_ptr[current_batch_idx_orig];
    if (local_row_idx_in_batch_orig >= current_k_seq_len) {
        s_k_all_data_ptr[idx_flat_q_x_head] = -std::numeric_limits<ElementAccum>::infinity();
        return;
    }

    const Element* q_vec_ptr = q_comp_ptr_base
                             + (int64_t)original_q_global_flat_idx * q_comp_stride_s_ele
                             + (int64_t)h_q_idx_orig * q_comp_stride_h_ele;

    const Element* k_vec_ptr = k_comp_ptr_base
                             + (int64_t)original_k_global_flat_idx * k_comp_stride_s_ele
                             + (int64_t)h_k_idx_orig * k_comp_stride_h_ele;

    ElementAccum score = ElementAccum(0.0f);
    for (int d = 0; d < HEAD_SIZE_TPL; ++d) {
        score += static_cast<ElementAccum>(q_vec_ptr[d]) * static_cast<ElementAccum>(k_vec_ptr[d]);
    }

    score *= softmax_scale;

    s_k_all_data_ptr[idx_flat_q_x_head] = score;
}

void run_calculate_sk_prime_kernel(Flash_fwd_params &params, cudaStream_t stream) {
    const int head_size = params.d;
    const int total_q = params.total_q;
    if (total_q == 0) return;

    const int num_heads_q_orig = params.h;
    const int num_heads_k_orig = params.h_k;
    const int original_batch_size_for_sk_kernel = params.b;

    const int total_s_k_elements = total_q * num_heads_q_orig;
    if (total_s_k_elements == 0) return;

    const int threads_per_block_sk = 256;
    const int num_blocks_sk = (total_s_k_elements + threads_per_block_sk - 1) / threads_per_block_sk;

    const int64_t q_comp_stride_b_ele = params.q_batch_stride;
    const int64_t q_comp_stride_h_ele = params.q_head_stride;
    const int64_t q_comp_stride_s_ele = params.q_row_stride;
    const int64_t k_comp_stride_b_ele = params.k_batch_stride;
    const int64_t k_comp_stride_h_ele = params.k_head_stride;
    const int64_t k_comp_stride_s_ele = params.k_row_stride;

    FP16_SWITCH(!params.is_bf16, [&] {
        HEADDIM_SWITCH(head_size, [&] {
            calculate_sk_prime_kernel_impl<elem_type, float, kHeadDim><<<num_blocks_sk, threads_per_block_sk, 0, stream>>>(
                static_cast<const elem_type*>(params.q_ptr),
                static_cast<const elem_type*>(params.k_ptr),
                static_cast<float*>(params.s_k_prime_ptr),
                params.scale_softmax,
                total_q,
                original_batch_size_for_sk_kernel,
                num_heads_q_orig, num_heads_k_orig,
                q_comp_stride_b_ele, q_comp_stride_h_ele, q_comp_stride_s_ele,
                k_comp_stride_b_ele, k_comp_stride_h_ele, k_comp_stride_s_ele,
                params.cu_seqlens_q,
                params.cu_seqlens_k
            );
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        });
    });
}

template <typename Element, typename ElementAccum, int HEAD_SIZE_TPL, int HEAD_SIZE_ROUNDED_TPL>
__global__ void compress_attn_combine_kernel_impl(
    const Element* __restrict__ o_temp_ptr,
    const ElementAccum* __restrict__ m_prime_ptr,
    const ElementAccum* __restrict__ l_prime_ptr,
    const ElementAccum* __restrict__ s_k_prime_ptr,
    const Element* __restrict__ q_comp_ptr_base,
    const Element* __restrict__ k_comp_ptr_base,
    const Element* __restrict__ v_comp_ptr_base,
    const int* __restrict__ inv_row_remapping_q_prime_ptr,
    const int* __restrict__ num_first_compress_ptr,
    const int* __restrict__ cu_seqlens_q_orig_ptr,
    const int* __restrict__ cu_seqlens_k_orig_ptr,
    Element* __restrict__ o_final_ptr,
    ElementAccum* __restrict__ lse_final_ptr,
    const int original_batch_size,
    const int num_heads_q_orig,
    const int num_heads_k_orig,
    const float softmax_scale,
    const int total_q_elements_orig_runtime,
    const int64_t q_comp_stride_b_ele, const int64_t q_comp_stride_h_ele, const int64_t q_comp_stride_s_ele,
    const int64_t k_comp_stride_b_ele, const int64_t k_comp_stride_h_ele, const int64_t k_comp_stride_s_ele,
    const int64_t v_comp_stride_b_ele, const int64_t v_comp_stride_h_ele, const int64_t v_comp_stride_s_ele,
    const int64_t o_final_stride_b_ele, const int64_t o_final_stride_h_ele, const int64_t o_final_stride_s_ele
) {
    const int idx_flat_orig_q_x_head = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx_flat_orig_q_x_head >= total_q_elements_orig_runtime * num_heads_q_orig) { return; }

    const int original_global_q_flat_idx = idx_flat_orig_q_x_head / num_heads_q_orig;
    const int h_q_idx_orig = idx_flat_orig_q_x_head % num_heads_q_orig;
    const int h_k_idx_orig = h_q_idx_orig / (num_heads_q_orig / num_heads_k_orig);

    int current_batch_idx_orig = 0;
    int local_row_idx_in_batch_orig = original_global_q_flat_idx;
    for (int b_scan = 0; b_scan < original_batch_size; ++b_scan) {
        int len_q_b = cu_seqlens_q_orig_ptr[b_scan+1] - cu_seqlens_q_orig_ptr[b_scan];
        if (local_row_idx_in_batch_orig < len_q_b) {
            current_batch_idx_orig = b_scan;
            break;
        }
        local_row_idx_in_batch_orig -= len_q_b;
    }

    const int k_seq_len = cu_seqlens_k_orig_ptr[current_batch_idx_orig + 1] - cu_seqlens_k_orig_ptr[current_batch_idx_orig];
    const bool has_diagonal_kv = (local_row_idx_in_batch_orig < k_seq_len);

    Element* o_final_vec_ptr = o_final_ptr
                             + (int64_t)current_batch_idx_orig * o_final_stride_b_ele
                             + (int64_t)h_q_idx_orig * o_final_stride_h_ele
                             + (int64_t)local_row_idx_in_batch_orig * o_final_stride_s_ele;

    ElementAccum* lse_final_scalar_ptr = lse_final_ptr + idx_flat_orig_q_x_head;

    const int inv_remap_read_idx = original_global_q_flat_idx * num_heads_q_orig + h_q_idx_orig;
    const int q_prime_global_idx = inv_row_remapping_q_prime_ptr[inv_remap_read_idx];

    if (q_prime_global_idx < 0) {
        if (has_diagonal_kv) {
            const Element* v_comp_vec_ptr = v_comp_ptr_base
                                          + (int64_t)original_global_q_flat_idx * v_comp_stride_s_ele
                                          + (int64_t)h_k_idx_orig * v_comp_stride_h_ele;
            for (int d = 0; d < HEAD_SIZE_TPL; ++d) {
                o_final_vec_ptr[d] = v_comp_vec_ptr[d];
            }
            *lse_final_scalar_ptr = s_k_prime_ptr[idx_flat_orig_q_x_head];
        } else {
            for (int d = 0; d < HEAD_SIZE_TPL; ++d) {
                o_final_vec_ptr[d] = static_cast<Element>(0.0f);
            }
            *lse_final_scalar_ptr = -std::numeric_limits<ElementAccum>::infinity();
        }
    } else {
        const ElementAccum m_p_val = m_prime_ptr[q_prime_global_idx] * softmax_scale;
        const ElementAccum l_p_val = l_prime_ptr[q_prime_global_idx];
        const ElementAccum sk_val = s_k_prime_ptr[idx_flat_orig_q_x_head];
        const bool only_diagonal_is_valid = (l_p_val <= 1e-10f || m_p_val <= -std::numeric_limits<ElementAccum>::infinity())
                                        && (sk_val > -std::numeric_limits<ElementAccum>::infinity())
                                        && has_diagonal_kv;

        if (only_diagonal_is_valid) {
            const Element* v_comp_vec_ptr = v_comp_ptr_base
                                          + (int64_t)original_global_q_flat_idx * v_comp_stride_s_ele
                                          + (int64_t)h_k_idx_orig * v_comp_stride_h_ele;
            for (int d = 0; d < HEAD_SIZE_TPL; ++d) {
                o_final_vec_ptr[d] = v_comp_vec_ptr[d];
            }
            *lse_final_scalar_ptr = sk_val;
        } else if (!has_diagonal_kv && l_p_val > 1e-10f) {
            const Element* o_temp_row_ptr = o_temp_ptr + (int64_t)q_prime_global_idx * HEAD_SIZE_ROUNDED_TPL;
            for (int d = 0; d < HEAD_SIZE_TPL; ++d) {
                ElementAccum normalized_val = static_cast<ElementAccum>(o_temp_row_ptr[d]) / l_p_val;
                o_final_vec_ptr[d] = static_cast<Element>(normalized_val);
            }
            *lse_final_scalar_ptr = m_p_val + logf(l_p_val);
        } else if (has_diagonal_kv) {
            const Element* o_temp_row_ptr = o_temp_ptr + (int64_t)q_prime_global_idx * HEAD_SIZE_ROUNDED_TPL;
            const Element* v_comp_vec_ptr = v_comp_ptr_base
                                          + (int64_t)original_global_q_flat_idx * v_comp_stride_s_ele
                                          + (int64_t)h_k_idx_orig * v_comp_stride_h_ele;

            const ElementAccum lse_rect = (l_p_val > 1e-10f) ? m_p_val + logf(l_p_val) : -std::numeric_limits<ElementAccum>::infinity();
            const ElementAccum lse_diag = sk_val;

            const ElementAccum m_final = fmaxf(lse_rect, lse_diag);
            const ElementAccum exp_rect = __expf(lse_rect - m_final);
            const ElementAccum exp_diag = __expf(lse_diag - m_final);
            const ElementAccum l_final = exp_rect + exp_diag;

            if (l_final <= 1e-10f) {
                for (int d = 0; d < HEAD_SIZE_TPL; ++d) {
                    o_final_vec_ptr[d] = static_cast<Element>(0.0f);
                }
                *lse_final_scalar_ptr = -std::numeric_limits<ElementAccum>::infinity();
            } else {
                const ElementAccum inv_l_final = 1.0f / l_final;
                const ElementAccum alpha = exp_rect * inv_l_final;
                const ElementAccum beta = exp_diag * inv_l_final;

                for (int d = 0; d < HEAD_SIZE_TPL; ++d) {
                    ElementAccum o_rect_normalized = (l_p_val > 1e-10f) ?
                        static_cast<ElementAccum>(o_temp_row_ptr[d]) / l_p_val : 0.0f;

                    ElementAccum o_diag = static_cast<ElementAccum>(v_comp_vec_ptr[d]);

                    ElementAccum combined = alpha * o_rect_normalized + beta * o_diag;

                    o_final_vec_ptr[d] = static_cast<Element>(combined);
                }

                *lse_final_scalar_ptr = m_final + logf(l_final);
            }
        } else {
            for (int d = 0; d < HEAD_SIZE_TPL; ++d) {
                o_final_vec_ptr[d] = static_cast<Element>(0.0f);
            }
            *lse_final_scalar_ptr = -std::numeric_limits<ElementAccum>::infinity();
        }
    }
}

void run_compress_attn_combine_kernel(Flash_fwd_params &params, cudaStream_t stream) {
    const int head_size_tpl = params.d;
    const int head_size_rounded_tpl_runtime = params.d_rounded;

    const int total_q_elements_orig = params.total_q;
    if (total_q_elements_orig == 0) return;

    const int original_batch_size = params.b;
    const int num_heads_q_orig = params.h;
    const int num_heads_k_orig = params.h_k;
    const int total_combine_items = total_q_elements_orig * num_heads_q_orig;
    if (total_combine_items == 0) return;

    const int threads_per_block_combine = 256;
    const int num_blocks_combine = (total_combine_items + threads_per_block_combine - 1) / threads_per_block_combine;

    const int64_t q_comp_stride_b_ele = params.q_batch_stride;
    const int64_t q_comp_stride_h_ele = params.q_head_stride;
    const int64_t q_comp_stride_s_ele = params.q_row_stride;
    const int64_t k_comp_stride_b_ele = params.k_batch_stride;
    const int64_t k_comp_stride_h_ele = params.k_head_stride;
    const int64_t k_comp_stride_s_ele = params.k_row_stride;
    const int64_t v_comp_stride_b_ele = params.v_batch_stride;
    const int64_t v_comp_stride_h_ele = params.v_head_stride;
    const int64_t v_comp_stride_s_ele = params.v_row_stride;
    const int64_t o_final_stride_b_ele = params.o_batch_stride;
    const int64_t o_final_stride_h_ele = params.o_head_stride;
    const int64_t o_final_stride_s_ele = params.o_row_stride;

    FP16_SWITCH(!params.is_bf16, [&] {
        HEADDIM_SWITCH(head_size_tpl, [&] {
            HEADDIM_ROUNDED_SWITCH(head_size_rounded_tpl_runtime, kHeadDimRoundedCompileTime, [&] {
                compress_attn_combine_kernel_impl<elem_type, float, kHeadDim, kHeadDimRoundedCompileTime><<<num_blocks_combine, threads_per_block_combine, 0, stream>>>(
                    static_cast<const elem_type*>(params.o_temp_ptr),
                    static_cast<const float*>(params.m_prime_ptr),
                    static_cast<const float*>(params.l_prime_ptr),
                    static_cast<const float*>(params.s_k_prime_ptr),
                    static_cast<const elem_type*>(params.q_ptr),
                    static_cast<const elem_type*>(params.k_ptr),
                    static_cast<const elem_type*>(params.v_ptr),
                    params.inv_row_remapping_q_prime_ptr,
                    params.num_first_compress_ptr,
                    params.cu_seqlens_q,
                    params.cu_seqlens_k,
                    static_cast<elem_type*>(params.o_ptr),
                    static_cast<float*>(params.softmax_lse_ptr),
                    original_batch_size, num_heads_q_orig, num_heads_k_orig,
                    params.scale_softmax, total_q_elements_orig,
                    q_comp_stride_b_ele, q_comp_stride_h_ele, q_comp_stride_s_ele,
                    k_comp_stride_b_ele, k_comp_stride_h_ele, k_comp_stride_s_ele,
                    v_comp_stride_b_ele, v_comp_stride_h_ele, v_comp_stride_s_ele,
                    o_final_stride_b_ele, o_final_stride_h_ele, o_final_stride_s_ele
                );
                C10_CUDA_KERNEL_LAUNCH_CHECK();
            });
        });
    });
}

void compute_cumulative_lengths_from_per_bh_counts(
    const int* d_per_bh_counts_q, const int* d_per_bh_counts_kv,
    int* d_cu_seqlens_q_prime, int* d_cu_seqlens_kv_prime,
    int& h_total_q_prime, int& h_total_kv_prime,
    int& h_max_seqlen_q_prime, int& h_max_seqlen_kv_prime,
    int original_batch_size, int original_num_heads, cudaStream_t stream) {

    const int num_bh_items = original_batch_size * original_num_heads;
    if (num_bh_items == 0) {
        h_total_q_prime = 0; h_total_kv_prime = 0;
        h_max_seqlen_q_prime = 0; h_max_seqlen_kv_prime = 0;
        if (d_cu_seqlens_q_prime) C10_CUDA_CHECK(cudaMemsetAsync(d_cu_seqlens_q_prime, 0, (num_bh_items + 1) * sizeof(int), stream));
        if (d_cu_seqlens_kv_prime) C10_CUDA_CHECK(cudaMemsetAsync(d_cu_seqlens_kv_prime, 0, (num_bh_items + 1) * sizeof(int), stream));
        return;
    }
    std::vector<int> h_per_bh_counts_q(num_bh_items);
    std::vector<int> h_per_bh_counts_kv(num_bh_items);
    if (d_per_bh_counts_q) {
        C10_CUDA_CHECK(cudaMemcpyAsync(h_per_bh_counts_q.data(), d_per_bh_counts_q, num_bh_items * sizeof(int), cudaMemcpyDeviceToHost, stream));
    } else {
        std::fill(h_per_bh_counts_q.begin(), h_per_bh_counts_q.end(), 0);
    }
    if (d_per_bh_counts_kv) {
        C10_CUDA_CHECK(cudaMemcpyAsync(h_per_bh_counts_kv.data(), d_per_bh_counts_kv, num_bh_items * sizeof(int), cudaMemcpyDeviceToHost, stream));
    } else {
         std::fill(h_per_bh_counts_kv.begin(), h_per_bh_counts_kv.end(), 0);
    }

    C10_CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<int> h_cu_seqlens_q_prime(num_bh_items + 1, 0);
    std::vector<int> h_cu_seqlens_kv_prime(num_bh_items + 1, 0);

    h_max_seqlen_q_prime = 0;
    if (!h_per_bh_counts_q.empty()) {
        auto max_it = std::max_element(h_per_bh_counts_q.begin(), h_per_bh_counts_q.end());
        if (max_it != h_per_bh_counts_q.end()) h_max_seqlen_q_prime = *max_it;
    }
    std::partial_sum(h_per_bh_counts_q.begin(), h_per_bh_counts_q.end(), h_cu_seqlens_q_prime.begin() + 1);
    if (!h_cu_seqlens_q_prime.empty()) h_total_q_prime = h_cu_seqlens_q_prime.back();
    else h_total_q_prime = 0;

    h_max_seqlen_kv_prime = 0;
    if (!h_per_bh_counts_kv.empty()) {
        auto max_it = std::max_element(h_per_bh_counts_kv.begin(), h_per_bh_counts_kv.end());
        if (max_it != h_per_bh_counts_kv.end()) h_max_seqlen_kv_prime = *max_it;
    }
    std::partial_sum(h_per_bh_counts_kv.begin(), h_per_bh_counts_kv.end(), h_cu_seqlens_kv_prime.begin() + 1);
    if (!h_cu_seqlens_kv_prime.empty()) h_total_kv_prime = h_cu_seqlens_kv_prime.back();
    else h_total_kv_prime = 0;

    if (d_cu_seqlens_q_prime) {
        C10_CUDA_CHECK(cudaMemcpyAsync(d_cu_seqlens_q_prime, h_cu_seqlens_q_prime.data(), (num_bh_items + 1) * sizeof(int), cudaMemcpyHostToDevice, stream));
    }
    if (d_cu_seqlens_kv_prime) {
        C10_CUDA_CHECK(cudaMemcpyAsync(d_cu_seqlens_kv_prime, h_cu_seqlens_kv_prime.data(), (num_bh_items + 1) * sizeof(int), cudaMemcpyHostToDevice, stream));
    }
}

__global__ void compute_lse_prime_kernel(
    const float* __restrict__ m_prime,
    const float* __restrict__ l_prime,
    float* __restrict__ lse_prime,
    int total_q_prime,
    const float softmax_scale
) {
    int idx = blockIdx.x * blockIdx.x + threadIdx.x;
    if (idx < total_q_prime) {
        float m_unscaled = m_prime[idx];
        float l = l_prime[idx];

        lse_prime[idx] = (l > 1e-10f) ? (m_unscaled * softmax_scale + logf(l)) : -std::numeric_limits<float>::infinity();
    }
}

void run_compute_lse_prime_kernel(
    const float* m_prime_ptr,
    const float* l_prime_ptr,
    float* lse_prime_ptr,
    int total_q_prime,
    cudaStream_t stream,
    const float softmax_scale
) {
    if (total_q_prime <= 0) return;
    const int threads = 256;
    const int blocks = (total_q_prime + threads - 1) / threads;
    compute_lse_prime_kernel<<<blocks, threads, 0, stream>>>(
        m_prime_ptr,
        l_prime_ptr,
        lse_prime_ptr,
        total_q_prime,
        softmax_scale
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

/**
 * @brief [实现] 这是 run_scan_for_prime_seqlens 的 CUDA C++ 实现。
 *        所有 CUB 调用都在这个函数内部，由 NVCC 正确编译。
 */
void run_scan_for_prime_seqlens(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    const int* d_per_bh_counts_q,
    const int* d_per_bh_counts_kv,
    int* d_cu_seqlens_q_prime,
    int* d_cu_seqlens_kv_prime,
    int* d_total_q_prime,
    int* d_total_kv_prime,
    int num_bh_items,
    cudaStream_t stream
) {
    if (num_bh_items == 0) {
        // 对于空输入，我们需要确保 cu_seqlens[0] 是 0，并且总数是 0
        C10_CUDA_CHECK(cudaMemsetAsync(d_cu_seqlens_q_prime, 0, sizeof(int), stream));
        C10_CUDA_CHECK(cudaMemsetAsync(d_cu_seqlens_kv_prime, 0, sizeof(int), stream));
        C10_CUDA_CHECK(cudaMemsetAsync(d_total_q_prime, 0, sizeof(int), stream));
        C10_CUDA_CHECK(cudaMemsetAsync(d_total_kv_prime, 0, sizeof(int), stream));
        temp_storage_bytes = 0;
        return;
    }

    // CUB 的两段式 API：第一次调用 (nullptr) 获取大小
    if (d_temp_storage == nullptr) {
        size_t bytes_scan_q = 0, bytes_scan_kv = 0, bytes_reduce_q = 0, bytes_reduce_kv = 0;
        // 注意：输出指针现在直接指向数组开头，不再 +1
        cub::DeviceScan::ExclusiveSum(nullptr, bytes_scan_q, d_per_bh_counts_q, d_cu_seqlens_q_prime, num_bh_items, stream);
        cub::DeviceScan::ExclusiveSum(nullptr, bytes_scan_kv, d_per_bh_counts_kv, d_cu_seqlens_kv_prime, num_bh_items, stream);
        cub::DeviceReduce::Sum(nullptr, bytes_reduce_q, d_per_bh_counts_q, d_total_q_prime, num_bh_items, stream);
        cub::DeviceReduce::Sum(nullptr, bytes_reduce_kv, d_per_bh_counts_kv, d_total_kv_prime, num_bh_items, stream);
        temp_storage_bytes = std::max({bytes_scan_q, bytes_scan_kv, bytes_reduce_q, bytes_reduce_kv});
        return;
    }

    // 执行 Q' 的扫描和归约
    // ExclusiveSum 会自动处理第一个元素为 0 的情况，我们直接从 index 0 开始写
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_per_bh_counts_q, d_cu_seqlens_q_prime, num_bh_items, stream);
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_per_bh_counts_q, d_total_q_prime, num_bh_items, stream);
    // 将总和写入 cu_seqlens 的最后一个位置 (index num_bh_items)
    C10_CUDA_CHECK(cudaMemcpyAsync(d_cu_seqlens_q_prime + num_bh_items, d_total_q_prime, sizeof(int), cudaMemcpyDeviceToDevice, stream));

    // 执行 K'/V' 的扫描和归约
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_per_bh_counts_kv, d_cu_seqlens_kv_prime, num_bh_items, stream);
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_per_bh_counts_kv, d_total_kv_prime, num_bh_items, stream);
    C10_CUDA_CHECK(cudaMemcpyAsync(d_cu_seqlens_kv_prime + num_bh_items, d_total_kv_prime, sizeof(int), cudaMemcpyDeviceToDevice, stream));
}

/**
 * @brief [新实现-内核] 轻量级内核，用于并行计算每个 batch-head 的 Q' 和 K'/V' 长度。
 *        每个线程块只使用一个线程来处理一个 batch-head，避免了块内同步的开销。
 */
__global__ void precompute_prime_lengths_kernel_new(
    const int* __restrict__ cu_seqlens_q_orig_ptr,
    const int* __restrict__ cu_seqlens_k_orig_ptr,
    const bool* __restrict__ merge_indices_ptr,
    const int* __restrict__ num_first_compress_ptr,
    int* __restrict__ d_per_bh_counts_q_prime_ptr,
    int* __restrict__ d_per_bh_counts_kv_prime_ptr,
    const int original_batch_size,
    const int original_num_heads,
    const int64_t merge_indices_stride_b,
    const int64_t merge_indices_stride_h,
    const int max_seq_len_k_orig_for_merge_idx
) {
    const int b_orig = blockIdx.x;
    const int h_orig = blockIdx.y;
    const int bh_flat_idx = b_orig * original_num_heads + h_orig;

    const int q_seq_len = cu_seqlens_q_orig_ptr[b_orig + 1] - cu_seqlens_q_orig_ptr[b_orig];
    const int k_seq_len = cu_seqlens_k_orig_ptr[b_orig + 1] - cu_seqlens_k_orig_ptr[b_orig];

    const int nfc_val = num_first_compress_ptr[bh_flat_idx];

    // 计算 Q' 长度
    d_per_bh_counts_q_prime_ptr[bh_flat_idx] = max(0, q_seq_len - (nfc_val + 1));

    // 计算 K'/V' 长度
    if (k_seq_len == 0) {
        d_per_bh_counts_kv_prime_ptr[bh_flat_idx] = 0;
        return;
    }

    const bool* current_merge_indices_bh_start = merge_indices_ptr
                                         + (int64_t)b_orig * merge_indices_stride_b
                                         + (int64_t)h_orig * merge_indices_stride_h;
    int survivor_k_count = 0;
    for (int i = 0; i < k_seq_len; ++i) {
        bool is_compressed = (i < nfc_val);
        if (!is_compressed && i < k_seq_len - 1) {
            // 修正: 确保 i 不会访问 merge_indices 的边界之外
            // merge_indices 的大小是 (..., S_k - 1)
            // 因此 i 必须严格小于 S_k - 1
            if (i < max_seq_len_k_orig_for_merge_idx - 1) {
                 if (current_merge_indices_bh_start[i]) {
                    is_compressed = true;
                }
            }
        }
        if (!is_compressed) {
            survivor_k_count++;
        }
    }
    d_per_bh_counts_kv_prime_ptr[bh_flat_idx] = max(0, survivor_k_count - 1);
}

/**
 * @brief [新实现-启动器] 启动 precompute_prime_lengths_kernel_new。
 */
void run_precompute_prime_lengths(Flash_fwd_params &params, cudaStream_t stream,
                                  int* d_per_bh_counts_q_prime, int* d_per_bh_counts_kv_prime) {
    dim3 grid(params.b, params.h);
    dim3 block(1); // 每个线程块只用一个线程处理一个 batch-head
    precompute_prime_lengths_kernel_new<<<grid, block, 0, stream>>>(
        params.cu_seqlens_q, params.cu_seqlens_k,
        static_cast<const bool*>(params.merge_indices_ptr),
        params.num_first_compress_ptr,
        d_per_bh_counts_q_prime, d_per_bh_counts_kv_prime,
        params.b, params.h,
        params.merge_indices_batch_stride, params.merge_indices_head_stride,
        params.seqlen_k // host 端的 max_seqlen_k
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


/**
 * @brief [最终修复版本 V3] 融合的 Gather 内核。
 *        该版本恢复使用共享内存存储幸存者索引列表，但与旧的低效内核不同，
 *        此列表的构建和使用都是完全并行化的，从而在保证逻辑绝对正确的同时，
 *        维持了极高的计算效率。
 *        1. Pass 1: 所有线程并行计算，使用原子操作将幸存者的原始索引无冲突地填入共享内存数组。
 *        2. Pass 2: 所有线程并行从共享内存索引数组中读取，执行K'和V'的Gather操作。
 *           对于V'，通过读取 smem_survivor_indices[i + 1] 来实现完美的、无歧义的“错位”采集。
 */
template <typename Element, int BLOCK_THREADS, int HEAD_SIZE, int MAX_SEQLEN_K_TPL>
__global__ void fused_gather_kernel_final_impl(Flash_fwd_params params) {
    // K'/V' Gather 使用的共享内存
    extern __shared__ char smem[];

    // --- 高效的共享内存布局 ---
    // 1. 用于并行扫描的临时存储
    __align__(alignof(typename cub::BlockScan<int, BLOCK_THREADS>::TempStorage))
    char* temp_storage_raw = smem;
    typename cub::BlockScan<int, BLOCK_THREADS>::TempStorage& temp_storage = *reinterpret_cast<typename cub::BlockScan<int, BLOCK_THREADS>::TempStorage*>(temp_storage_raw);

    // 2. 存储幸存者在原始序列中的局部索引 (0 to k_seq_len-1)
    int* smem_survivor_indices = (int*)(temp_storage_raw + sizeof(typename cub::BlockScan<int, BLOCK_THREADS>::TempStorage));

    const int b_orig = blockIdx.x;
    const int h_orig = blockIdx.y;
    const int tid = threadIdx.x;

    // --- 元数据准备 ---
    __shared__ int smem_metadata[3];
    if (tid == 0) {
        const int bh_flat_idx = b_orig * params.h + h_orig;
        smem_metadata[0] = params.num_first_compress_ptr[bh_flat_idx];
        smem_metadata[1] = params.cu_seqlens_q_prime_ptr[bh_flat_idx];
        smem_metadata[2] = params.cu_seqlens_kv_prime_ptr[bh_flat_idx];
    }
    __syncthreads();

    const int nfc_val = smem_metadata[0];
    const int q_prime_base_addr = smem_metadata[1];
    const int kv_prime_base_addr = smem_metadata[2];
    const int h_k_orig = h_orig / params.h_h_k_ratio;
    const int q_seq_len = params.cu_seqlens_q[b_orig + 1] - params.cu_seqlens_q[b_orig];
    const int k_seq_len = params.cu_seqlens_k[b_orig + 1] - params.cu_seqlens_k[b_orig];

    if (q_seq_len == 0 && k_seq_len == 0) return;

    // --- Q' Gather (此部分逻辑正确，保持不变) ---
    const int tokens_to_skip_q = nfc_val + 1;
    for (int i_q_local = tid; i_q_local < q_seq_len; i_q_local += BLOCK_THREADS) {
        const bool should_include_in_q_prime = i_q_local >= tokens_to_skip_q;
        const int original_q_global_flat_idx = params.cu_seqlens_q[b_orig] + i_q_local;
        const int inv_remap_idx = original_q_global_flat_idx * params.h + h_orig;

        if (should_include_in_q_prime) {
            const int q_prime_local_idx = i_q_local - tokens_to_skip_q;
            const int q_prime_global_write_idx = q_prime_base_addr + q_prime_local_idx;
            const Element* src_q_token_head_ptr = (const Element*)params.q_ptr + (int64_t)original_q_global_flat_idx * params.q_row_stride + (int64_t)h_orig * params.q_head_stride;
            Element* dst_q_prime_token_ptr = (Element*)params.q_prime_ptr + (int64_t)q_prime_global_write_idx * HEAD_SIZE;
            #pragma unroll
            for (int d = 0; d < HEAD_SIZE / (128 / (sizeof(Element) * 8)); ++d) {
                reinterpret_cast<uint128_t*>(dst_q_prime_token_ptr)[d] = reinterpret_cast<const uint128_t*>(src_q_token_head_ptr)[d];
            }
            params.row_remapping_q_prime_ptr[q_prime_global_write_idx] = original_q_global_flat_idx;
            params.inv_row_remapping_q_prime_ptr[inv_remap_idx] = q_prime_global_write_idx;
            params.q_is_participated_mask_ptr[inv_remap_idx] = true;
        } else {
             params.inv_row_remapping_q_prime_ptr[inv_remap_idx] = -1;
             params.q_is_participated_mask_ptr[inv_remap_idx] = false;
        }
    }

    // --- K'/V' Gather (高效且正确的修复版) ---
    if (k_seq_len > 0) {
        const bool* current_merge_indices_bh_start = (const bool*)params.merge_indices_ptr
                                             + (int64_t)b_orig * params.merge_indices_batch_stride
                                             + (int64_t)h_orig * params.merge_indices_head_stride;

        __shared__ int num_survivors_shared;

        // Pass 1 & 2: 并行构建有序的幸存者索引列表
        int chunk_base_rank = 0;
        for (int i_chunk_start = 0; i_chunk_start < k_seq_len; i_chunk_start += BLOCK_THREADS) {
            const int i_k_local = i_chunk_start + tid;
            bool is_survivor = false;
            if (i_k_local < k_seq_len) {
                bool is_compressed = (i_k_local < nfc_val);
                if (!is_compressed && i_k_local < k_seq_len - 1) {
                    if (i_k_local < params.seqlen_k - 1) {
                        if (current_merge_indices_bh_start[i_k_local]) {
                            is_compressed = true;
                        }
                    }
                }
                is_survivor = !is_compressed;
            }

            int rank_in_chunk;
            int total_in_chunk;
            cub::BlockScan<int, BLOCK_THREADS>(temp_storage).ExclusiveSum(is_survivor ? 1 : 0, rank_in_chunk, total_in_chunk);

            if (is_survivor) {
                const int survivor_rank = chunk_base_rank + rank_in_chunk;
                if (survivor_rank < MAX_SEQLEN_K_TPL) {
                    smem_survivor_indices[survivor_rank] = i_k_local;
                }
            }
            __syncthreads();
            chunk_base_rank += total_in_chunk;
            __syncthreads();
        }

        if (tid == 0) {
            num_survivors_shared = chunk_base_rank;
        }
        __syncthreads();

        // Pass 3: 并行采集 K' 和 V'
        const int num_survivors = num_survivors_shared;
        for (int i = tid; i < num_survivors; i += BLOCK_THREADS) {
            // --- Gather K' ---
            if (i < num_survivors - 1) {
                const int k_prime_local_idx = i;
                const int original_k_local_idx = smem_survivor_indices[i];
                const int k_prime_global_write_idx = kv_prime_base_addr + k_prime_local_idx;
                const int original_k_global_flat_idx = params.cu_seqlens_k[b_orig] + original_k_local_idx;

                const Element* src_k_token_head_ptr = (const Element*)params.k_ptr + (int64_t)original_k_global_flat_idx * params.k_row_stride + (int64_t)h_k_orig * params.k_head_stride;
                Element* dst_k_prime_token_ptr = (Element*)params.k_prime_ptr + (int64_t)k_prime_global_write_idx * HEAD_SIZE;
                #pragma unroll
                for (int d = 0; d < HEAD_SIZE / (128 / (sizeof(Element) * 8)); ++d) {
                    reinterpret_cast<uint128_t*>(dst_k_prime_token_ptr)[d] = reinterpret_cast<const uint128_t*>(src_k_token_head_ptr)[d];
                }
                params.col_remapping_k_prime_ptr[k_prime_global_write_idx] = original_k_global_flat_idx;
            }

            // --- Gather V' ---
            if (i < num_survivors - 1) {
                const int v_prime_local_idx = i;
                // **关键修复**: 从有序的共享索引数组中读取下一个幸存者的索引
                const int original_v_local_idx = smem_survivor_indices[i + 1];
                const int v_prime_global_write_idx = kv_prime_base_addr + v_prime_local_idx;
                const int original_v_global_flat_idx = params.cu_seqlens_k[b_orig] + original_v_local_idx;

                const Element* src_v_token_head_ptr = (const Element*)params.v_ptr + (int64_t)original_v_global_flat_idx * params.v_row_stride + (int64_t)h_k_orig * params.v_head_stride;
                Element* dst_v_prime_token_ptr = (Element*)params.v_prime_ptr + (int64_t)v_prime_global_write_idx * HEAD_SIZE;
                #pragma unroll
                for (int d = 0; d < HEAD_SIZE / (128 / (sizeof(Element) * 8)); ++d) {
                    reinterpret_cast<uint128_t*>(dst_v_prime_token_ptr)[d] = reinterpret_cast<const uint128_t*>(src_v_token_head_ptr)[d];
                }
                params.v_remapping_prime_ptr[v_prime_global_write_idx] = original_v_global_flat_idx;
            }
        }
    }
}


/**
 * @brief [最终修复版本] 启动器，为修复后的内核计算正确的共享内存大小。
 */
void run_fused_gather_kernel(Flash_fwd_params &params, cudaStream_t stream) {
    dim3 grid(params.b, params.h);
    const int block_threads = 256;
    dim3 block(block_threads);

    const int max_k_len = params.seqlen_k;

    auto launch_kernel = [&](auto max_len_tpl) {
        constexpr int MAX_LEN = decltype(max_len_tpl)::value;

        auto align_to = [] (size_t ptr_offset, size_t alignment) {
            return (ptr_offset + alignment - 1) & ~(alignment - 1);
        };

        size_t smem_scan_storage = sizeof(typename cub::BlockScan<int, block_threads>::TempStorage);
        size_t smem_indices_storage = sizeof(int) * MAX_LEN;

        size_t total_smem_size = align_to(smem_scan_storage, alignof(int)) + smem_indices_storage;


        FP16_SWITCH(!params.is_bf16, [&] {
            HEADDIM_SWITCH(params.d, [&] {
                auto kernel = &fused_gather_kernel_final_impl<elem_type, block_threads, kHeadDim, MAX_LEN>;

                if (total_smem_size >= 48 * 1024) {
                     C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, total_smem_size));
                }

                kernel<<<grid, block, total_smem_size, stream>>>(params);
            });
        });
    };

    // 根据序列长度选择模板实例，以优化共享内存使用
    if (max_k_len <= 1024) { launch_kernel(std::integral_constant<int, 1024>{}); }
    else if (max_k_len <= 2048) { launch_kernel(std::integral_constant<int, 2048>{}); }
    else if (max_k_len <= 4096) { launch_kernel(std::integral_constant<int, 4096>{}); }
    else {
        if (max_k_len <= 8192) {
             launch_kernel(std::integral_constant<int, 8192>{});
        } else {
            TORCH_CHECK(false, "DCAttention fused gather kernel does not support sequence length > 8192");
        }
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void run_reduce_max_for_prime_seqlens(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    const int* d_per_bh_counts_q,
    const int* d_per_bh_counts_kv,
    int* d_max_seqlen_q_prime,
    int* d_max_seqlen_k_prime,
    int num_bh_items,
    cudaStream_t stream
) {
    if (num_bh_items == 0) {
        C10_CUDA_CHECK(cudaMemsetAsync(d_max_seqlen_q_prime, 0, sizeof(int), stream));
        C10_CUDA_CHECK(cudaMemsetAsync(d_max_seqlen_k_prime, 0, sizeof(int), stream));
        temp_storage_bytes = 0;
        return;
    }

    // CUB 的两段式 API
    if (d_temp_storage == nullptr) {
        size_t bytes_max_q = 0, bytes_max_k = 0;
        cub::DeviceReduce::Max(nullptr, bytes_max_q, d_per_bh_counts_q, d_max_seqlen_q_prime, num_bh_items, stream);
        cub::DeviceReduce::Max(nullptr, bytes_max_k, d_per_bh_counts_kv, d_max_seqlen_k_prime, num_bh_items, stream);
        temp_storage_bytes = std::max(bytes_max_q, bytes_max_k);
        return;
    }

    cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_per_bh_counts_q, d_max_seqlen_q_prime, num_bh_items, stream);
    cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_per_bh_counts_kv, d_max_seqlen_k_prime, num_bh_items, stream);
}

// ==================================================================================
// V2 优化版本内核实现 (新增内容)
// ==================================================================================

// ------------------- calculate_sk_prime_kernel 的 V2 版本 (已向量化) -------------------
template <typename Element, typename ElementAccum, int HEAD_SIZE_TPL>
__global__ void calculate_sk_prime_kernel_impl_v2(
    const Flash_fwd_params params
) {
    const int idx_flat_q_x_head = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx_flat_q_x_head >= params.total_q * params.h) {
        return;
    }

    const int original_q_global_flat_idx = idx_flat_q_x_head / params.h;
    const int h_q_idx_orig = idx_flat_q_x_head % params.h;
    const int h_k_idx_orig = h_q_idx_orig / params.h_h_k_ratio;

    const int current_batch_idx_orig = params.q_orig_to_batch_idx_ptr_fwd[original_q_global_flat_idx];
    const int local_row_idx_in_batch_orig = original_q_global_flat_idx - params.cu_seqlens_q[current_batch_idx_orig];

    const int current_k_seq_len = params.cu_seqlens_k[current_batch_idx_orig + 1] - params.cu_seqlens_k[current_batch_idx_orig];
    if (local_row_idx_in_batch_orig >= current_k_seq_len) {
        static_cast<ElementAccum*>(params.s_k_prime_ptr)[idx_flat_q_x_head] = -std::numeric_limits<ElementAccum>::infinity();
        return;
    }

    // --- 向量化修改 ---
    constexpr int ELEMENTS_PER_VEC = sizeof(uint128_t) / sizeof(Element); // 16 bytes / 2 bytes = 8 elements
    constexpr int VECTORS_PER_HEAD = HEAD_SIZE_TPL / ELEMENTS_PER_VEC;

    const uint128_t* q_vec_ptr = reinterpret_cast<const uint128_t*>(
        static_cast<const Element*>(params.q_ptr)
        + (int64_t)original_q_global_flat_idx * params.q_row_stride
        + (int64_t)h_q_idx_orig * params.q_head_stride
    );

    const uint128_t* k_vec_ptr = reinterpret_cast<const uint128_t*>(
        static_cast<const Element*>(params.k_ptr)
        + (int64_t)original_q_global_flat_idx * params.k_row_stride
        + (int64_t)h_k_idx_orig * params.k_head_stride
    );

    ElementAccum score = ElementAccum(0.0f);

    #pragma unroll
    for (int i = 0; i < VECTORS_PER_HEAD; ++i) {
        uint128_t q_u128 = q_vec_ptr[i];
        uint128_t k_u128 = k_vec_ptr[i];

        cutlass::Array<Element, ELEMENTS_PER_VEC> q_arr;
        cutlass::Array<Element, ELEMENTS_PER_VEC> k_arr;
        *reinterpret_cast<uint128_t*>(&q_arr) = q_u128;
        *reinterpret_cast<uint128_t*>(&k_arr) = k_u128;

        #pragma unroll
        for (int j = 0; j < ELEMENTS_PER_VEC; ++j) {
            score += static_cast<ElementAccum>(q_arr[j]) * static_cast<ElementAccum>(k_arr[j]);
        }
    }

    score *= params.scale_softmax;
    static_cast<ElementAccum*>(params.s_k_prime_ptr)[idx_flat_q_x_head] = score;
}

void run_calculate_sk_prime_kernel_v2(Flash_fwd_params &params, cudaStream_t stream) {
    const int total_s_k_elements = params.total_q * params.h;
    if (total_s_k_elements == 0) return;

    const int threads_per_block_sk = 256;
    const int num_blocks_sk = (total_s_k_elements + threads_per_block_sk - 1) / threads_per_block_sk;

    FP16_SWITCH(!params.is_bf16, [&] {
        HEADDIM_SWITCH(params.d, [&] {
            calculate_sk_prime_kernel_impl_v2<elem_type, float, kHeadDim><<<num_blocks_sk, threads_per_block_sk, 0, stream>>>(params);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        });
    });
}


// ------------------- compress_attn_combine_kernel 的 V2 版本 (已向量化) -------------------
template <typename Element, typename ElementAccum, int HEAD_SIZE_TPL, int HEAD_SIZE_ROUNDED_TPL>
__global__ void compress_attn_combine_kernel_impl_v2(const Flash_fwd_params params) {
    const int idx_flat_orig_q_x_head = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx_flat_orig_q_x_head >= params.total_q * params.h) { return; }

    // --- 1. 数据准备 (与之前相同) ---
    const int original_global_q_flat_idx = idx_flat_orig_q_x_head / params.h;
    const int h_q_idx_orig = idx_flat_orig_q_x_head % params.h;
    const int h_k_idx_orig = h_q_idx_orig / params.h_h_k_ratio;

    const int current_batch_idx_orig = params.q_orig_to_batch_idx_ptr_fwd[original_global_q_flat_idx];
    const int local_row_idx_in_batch_orig = original_global_q_flat_idx - params.cu_seqlens_q[current_batch_idx_orig];

    constexpr int ELEMENTS_PER_VEC = sizeof(uint128_t) / sizeof(Element);
    constexpr int VECTORS_PER_HEAD = HEAD_SIZE_TPL / ELEMENTS_PER_VEC;

    uint128_t* o_final_vec_ptr = reinterpret_cast<uint128_t*>(
        static_cast<Element*>(params.o_ptr)
        + (int64_t)current_batch_idx_orig * params.o_batch_stride
        + (int64_t)h_q_idx_orig * params.o_head_stride
        + (int64_t)local_row_idx_in_batch_orig * params.o_row_stride
    );

    ElementAccum* lse_final_scalar_ptr = static_cast<ElementAccum*>(params.softmax_lse_ptr) + idx_flat_orig_q_x_head;

    // --- 2. 计算谓词 (Predicates) ---
    const int inv_remap_read_idx = original_global_q_flat_idx * params.h + h_q_idx_orig;
    const int q_prime_global_idx = params.inv_row_remapping_q_prime_ptr[inv_remap_read_idx];

    const int k_seq_len = params.cu_seqlens_k[current_batch_idx_orig + 1] - params.cu_seqlens_k[current_batch_idx_orig];
    const bool has_diagonal_kv = (local_row_idx_in_batch_orig < k_seq_len);
    const bool is_rect_participant = q_prime_global_idx >= 0;

    // --- 3. 无条件获取所有计算所需的数据 ---
    const ElementAccum m_p_unscaled = is_rect_participant ? static_cast<const ElementAccum*>(params.m_prime_ptr)[q_prime_global_idx] : 0.0f;
    const ElementAccum l_p_val = is_rect_participant ? static_cast<const ElementAccum*>(params.l_prime_ptr)[q_prime_global_idx] : 0.0f;
    const ElementAccum sk_val = has_diagonal_kv ? static_cast<const ElementAccum*>(params.s_k_prime_ptr)[idx_flat_orig_q_x_head] : -std::numeric_limits<ElementAccum>::infinity();

    const bool can_use_rect_data = is_rect_participant && (l_p_val > 1e-10f);

    // --- 4. 无分支的 LSE 和权重计算 ---
    const ElementAccum lse_rect = can_use_rect_data ? (m_p_unscaled * params.scale_softmax + logf(l_p_val)) : -std::numeric_limits<ElementAccum>::infinity();
    const ElementAccum lse_diag = sk_val;

    const ElementAccum m_final = fmaxf(lse_rect, lse_diag);

    const ElementAccum exp_rect = __expf(lse_rect - m_final);
    const ElementAccum exp_diag = __expf(lse_diag - m_final);
    const ElementAccum l_final = exp_rect + exp_diag;

    const bool is_l_final_valid = l_final > 1e-10f;
    const ElementAccum inv_l_final = 1.0f / l_final;

    const ElementAccum alpha = is_l_final_valid ? exp_rect * inv_l_final : 0.0f;
    const ElementAccum beta = is_l_final_valid ? exp_diag * inv_l_final : (has_diagonal_kv ? 1.0f : 0.0f);

    // --- 5. 无分支的向量化输出计算 ---
    const uint128_t* o_temp_row_ptr = reinterpret_cast<const uint128_t*>(
        static_cast<const Element*>(params.o_temp_ptr) + (int64_t)q_prime_global_idx * HEAD_SIZE_ROUNDED_TPL
    );
    const uint128_t* v_comp_vec_ptr = reinterpret_cast<const uint128_t*>(
        static_cast<const Element*>(params.v_ptr)
        + (int64_t)original_global_q_flat_idx * params.v_row_stride
        + (int64_t)h_k_idx_orig * params.v_head_stride
    );

    const uint128_t zero_vec{};

    #pragma unroll
    for (int i = 0; i < VECTORS_PER_HEAD; ++i) {
        uint128_t o_temp_u128 = can_use_rect_data ? o_temp_row_ptr[i] : zero_vec;
        uint128_t v_comp_u128 = has_diagonal_kv ? v_comp_vec_ptr[i] : zero_vec;

        cutlass::Array<Element, ELEMENTS_PER_VEC> o_temp_arr;
        *reinterpret_cast<uint128_t*>(&o_temp_arr) = o_temp_u128;
        cutlass::Array<Element, ELEMENTS_PER_VEC> v_comp_arr;
        *reinterpret_cast<uint128_t*>(&v_comp_arr) = v_comp_u128;

        cutlass::Array<Element, ELEMENTS_PER_VEC> final_arr;

        #pragma unroll
        for(int j=0; j<ELEMENTS_PER_VEC; ++j) {
            // ======================= 最终修复点 =======================
            // 使用三元运算符确保在 l_p_val 无效时，o_rect_normalized 为 0，而不是 nan
            ElementAccum o_rect_normalized = can_use_rect_data ? (static_cast<ElementAccum>(o_temp_arr[j]) / l_p_val) : 0.0f;
            // ======================= 修复结束 =======================
            ElementAccum o_diag = static_cast<ElementAccum>(v_comp_arr[j]);

            final_arr[j] = static_cast<Element>(alpha * o_rect_normalized + beta * o_diag);
        }

        o_final_vec_ptr[i] = is_l_final_valid ? *reinterpret_cast<uint128_t*>(&final_arr) : zero_vec;
    }

    *lse_final_scalar_ptr = is_l_final_valid ? (m_final + logf(l_final)) : -std::numeric_limits<ElementAccum>::infinity();
}

void run_compress_attn_combine_kernel_v2(Flash_fwd_params &params, cudaStream_t stream) {
    const int total_combine_items = params.total_q * params.h;
    if (total_combine_items == 0) return;

    const int threads_per_block_combine = 256;
    const int num_blocks_combine = (total_combine_items + threads_per_block_combine - 1) / threads_per_block_combine;

    FP16_SWITCH(!params.is_bf16, [&] {
        HEADDIM_SWITCH(params.d, [&] {
            HEADDIM_ROUNDED_SWITCH(params.d_rounded, kHeadDimRoundedCompileTime, [&] {
                compress_attn_combine_kernel_impl_v2<elem_type, float, kHeadDim, kHeadDimRoundedCompileTime><<<num_blocks_combine, threads_per_block_combine, 0, stream>>>(params);
                C10_CUDA_KERNEL_LAUNCH_CHECK();
            });
        });
    });
}

__global__ void map_q_indices_to_batch_kernel(
    const int* __restrict__ cu_seqlens_q_orig,
    int* __restrict__ q_orig_to_batch_idx_map,
    const int batch_size,
    const int total_q_orig
) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total_q_orig;
         idx += blockDim.x * gridDim.x)
    {
        int left = 0;
        int right = batch_size;
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (cu_seqlens_q_orig[mid + 1] > idx) {
                right = mid;
            } else {
                left = mid + 1;
            }
        }
        q_orig_to_batch_idx_map[idx] = left;
    }
}

void run_map_q_indices_to_batch(
    const int* cu_seqlens_q_orig_ptr,
    int* q_orig_to_batch_idx_map_ptr,
    const int batch_size,
    const int total_q_orig,
    cudaStream_t stream
) {
    if (total_q_orig == 0) return;

    const int threads = 256;
    const int blocks = (total_q_orig + threads - 1) / threads;

    map_q_indices_to_batch_kernel<<<blocks, threads, 0, stream>>>(
        cu_seqlens_q_orig_ptr,
        q_orig_to_batch_idx_map_ptr,
        batch_size,
        total_q_orig
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

