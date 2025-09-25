/******************************************************************************
 * Copyright (c) 2025, xxx.
 ******************************************************************************/

#pragma once

#include <cuda.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>

constexpr int TOTAL_DIM = 0;
constexpr int H_DIM = 1;
constexpr int D_DIM = 2;

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Qkv_params {
    using index_t = int64_t;
    void *__restrict__ q_ptr;
    void *__restrict__ k_ptr;
    void *__restrict__ v_ptr;
    index_t q_batch_stride;
    index_t k_batch_stride;
    index_t v_batch_stride;
    index_t q_row_stride;
    index_t k_row_stride;
    index_t v_row_stride;
    index_t q_head_stride;
    index_t k_head_stride;
    index_t v_head_stride;
    int h, h_k;
    int h_h_k_ratio;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Flash_fwd_params : public Qkv_params {
    void * __restrict__ o_ptr;
    void * __restrict__ oaccum_ptr;
    index_t o_batch_stride;
    index_t o_row_stride;
    index_t o_head_stride;
    void * __restrict__ p_ptr; // Dropout probability mask tensor
    void * __restrict__ softmax_lse_ptr;
    void * __restrict__ softmax_lseaccum_ptr;
    int b, seqlen_q, seqlen_k, seqlen_knew, d, seqlen_q_rounded, seqlen_k_rounded, d_rounded, rotary_dim, total_q;
    float scale_softmax;
    float scale_softmax_log2;
    int * __restrict__ cu_seqlens_q;
    int * __restrict__ cu_seqlens_k;
    int * __restrict__ leftpad_k;
    int * __restrict__ seqused_k;
    int *__restrict__ blockmask; // Not used in current compress path
    void * __restrict__ knew_ptr;
    void * __restrict__ vnew_ptr;
    index_t knew_batch_stride;
    index_t vnew_batch_stride;
    index_t knew_row_stride;
    index_t vnew_row_stride;
    index_t knew_head_stride;
    index_t vnew_head_stride;
    void * __restrict__ rotary_cos_ptr;
    void * __restrict__ rotary_sin_ptr;
    int * __restrict__ cache_batch_idx;
    int * __restrict__ block_table;
    index_t block_table_batch_stride;
    int page_block_size;
    float p_dropout; // Probability of *dropping* an element
    uint8_t p_dropout_in_uint8_t;
    float rp_dropout; // 1.0 / (1.0 - p_dropout)
    float scale_softmax_rp_dropout;
    at::PhiloxCudaState philox_args; // For dropout RNG state
    uint64_t * rng_state; // For dropout custom RNG state saving
    int window_size_left, window_size_right;
    float softcap;
    bool is_bf16;
    bool is_causal; // Controls causal masking behavior
    bool is_seqlens_k_cumulative;
    bool is_rotary_interleaved;
    int num_splits; // For split-K/V original FlashAttention
    void * __restrict__ alibi_slopes_ptr; // Keep as void* for flexibility if type can vary
    index_t alibi_slopes_batch_stride;
    bool unpadded_lse;
    bool seqlenq_ngroups_swapped;

    // --- Members for Compressed Attention ---
    bool compress_attention; // If true, triggers compressed attention path

    // ======================= 最终修复 V3 =======================
    // 为前向 V2 优化内核专用的指针，使用新名称 _fwd
    const int *q_orig_to_batch_idx_ptr_fwd;
    // ======================= 修复结束 =======================


    void *merge_indices_ptr;            // bool*, but void* for flexibility if Flash_fwd_params is generic
    int64_t merge_indices_batch_stride; // Element stride for batch dim of merge_indices
    int64_t merge_indices_head_stride;  // Element stride for head dim of merge_indices
    const int *num_first_compress_ptr;  // int*

    void *q_prime_ptr;    // Element* (e.g., half*, bfloat16*)
    int64_t q_prime_row_stride; // 新增: Q'的行步长 (以元素为单位)
    void *k_prime_ptr;    // Element*
    int64_t k_prime_row_stride; // 新增: K'的行步长
    void *v_prime_ptr;    // Element*
    int64_t v_prime_row_stride; // 新增: V'的行步长
    void *o_temp_ptr;     // ElementAccum* (float*) for intermediate O_temp
    int64_t o_temp_row_stride;  // 新增: O_temp的行步长
    void *m_prime_ptr;    // ElementAccum* (float*) for m'
    void *l_prime_ptr;    // ElementAccum* (float*) for l'
    void *s_k_prime_ptr;  // ElementAccum* (float*) for ALL diagonal scores (indexed by original Q)

    int *row_remapping_q_prime_ptr; // int* mapping Q'_idx_global -> Q_orig_idx_flat_global
    int *col_remapping_k_prime_ptr; // int* mapping K'_idx_global -> K_orig_idx_flat_global
    int *v_remapping_prime_ptr;
    int *inv_row_remapping_q_prime_ptr; // int* mapping Q_orig_idx_flat_global -> Q'_idx_global (-1 if not in Q')

    int *cu_seqlens_q_prime_ptr;  // int* cumulative seq lengths for Q' (flat over pseudo_batch_size)
    int *cu_seqlens_kv_prime_ptr; // int* cumulative seq lengths for K'/V' (flat over pseudo_batch_size)

    int total_q_prime;    // Total number of tokens in Q' across all pseudo-batch items
    int total_k_prime;    // Total number of tokens in K' across all pseudo-batch items
    int max_seqlen_q_prime; // Max length of a Q' sequence in any pseudo-batch item
    int max_seqlen_k_prime; // Max length of a K' sequence in any pseudo-batch item


    //bool* q_is_participated_mask_ptr; // 指向"决策图"的指针
    bool* q_is_participated_mask_ptr; // 移除 const，因为 gather 内核需要写入它
};
/*struct Flash_bwd_params : public Flash_fwd_params {
    void *__restrict__ do_ptr;
    void *__restrict__ dq_ptr;
    void *__restrict__ dk_ptr;
    void *__restrict__ dv_ptr;
    void *__restrict__ dq_accum_ptr;
     // ======================= START MODIFICATION: Add dq_accum_batch_stride =======================
    int64_t dq_accum_batch_stride; // Stride for batch dimension of dq_accum
    // =======================  END MODIFICATION: Add dq_accum_batch_stride  =======================
    int64_t dq_accum_row_stride;   // 新增：为dq_accum_ptr添加行步长
    void *__restrict__ dk_accum_ptr;
    void *__restrict__ dv_accum_ptr;
    index_t do_batch_stride;
    index_t do_row_stride;
    index_t do_head_stride;
    index_t dq_batch_stride;
    index_t dk_batch_stride;
    index_t dv_batch_stride;
    index_t dq_row_stride;
    index_t dk_row_stride;
    index_t dv_row_stride;
    index_t dq_head_stride;
    index_t dk_head_stride;
    index_t dv_head_stride;
    void *__restrict__ dsoftmax_sum;
    // START: MODIFICATION - 为矩形注意力的dsoftmax_sum添加专用指针
    void *__restrict__ dsoftmax_sum_prime_ptr; // 用于存储 dot(dO', O'_normalized) 的结果
    // END: MODIFICATION
    // START: MODIFICATION - Add pointer for correction factors
    const float *__restrict__ alpha_ptr; // NEW: Pointer to per-row alpha values
    // END: MODIFICATION
    bool deterministic;
    index_t dq_accum_split_stride;
    
    // --- Additional fields for Compressed Attention Backward ---
    
    // Gradient of diagonal scores s_k (output from stage 1)
    void *__restrict__ s_k_grad_ptr;  // float*
    
    // Final output gradients for compressed tensors
    void *__restrict__ dq_comp_ptr;   // Element* - final gradient of Q_comp
    int64_t dq_comp_row_stride;   // 新增: dq_comp的行步长
    void *__restrict__ dk_comp_ptr;   // Element* - final gradient of K_comp  
    int64_t dk_comp_row_stride;   // 新增: dk_comp的行步长
    void *__restrict__ dv_comp_ptr;   // Element* - final gradient of V_comp
    int64_t dv_comp_row_stride;   // 新增: dv_comp的行步长

    // ======================= START MODIFICATION =======================
    // Stride information for intermediate padded tensors (dQ', dK', dV')
    int64_t dq_prime_batch_stride;
    int64_t dk_prime_batch_stride;
    int64_t dv_prime_batch_stride;
    // =======================  END MODIFICATION  =======================

    // ======================= START MODIFICATION: Add Missing Row Strides =======================
    int64_t dq_prime_row_stride;
    int64_t dk_prime_row_stride;
    int64_t dv_prime_row_stride;
    // =======================  END MODIFICATION: Add Missing Row Strides  =======================
    
    // Pointers to forward-pass intermediate tensors (READ-ONLY in bwd)
    void *__restrict__ o_temp_ptr;      // Rectangular FA output O' - Element*
    void *__restrict__ m_prime_ptr;     // Rectangular FA max statistics m' - float*
    void *__restrict__ l_prime_ptr;     // Rectangular FA sum statistics l' - float*
    
    // Pointer to the FULL diagonal scores tensor from the forward pass
    const void *__restrict__ s_k_fwd_ptr;       // float*
    // 新增：指向 LSE' = m' + log(l') 张量的指针
    const void *__restrict__ lse_prime_fwd_ptr; 
    
    // Pointer to precomputed mapping tables (READ-ONLY in bwd)
    const int *__restrict__ q_prime_to_pseudo_batch_idx_ptr;   // [total_q_prime]
    const int *__restrict__ kv_prime_to_pseudo_batch_idx_ptr;  // [total_k_prime]
    const int *__restrict__ q_orig_to_batch_idx_ptr;           // 新增: [total_q_orig] - a map from original q index to its batch index


    // 只需添加新的成员即可
    //const bool* q_is_participated_mask_ptr;
    int max_seqlen_q_prime_rounded; // 用于正确索引 dsoftmax_sum

    float* debug_ptr;

};*/

////////////////////////////////////////////////////////////////////////////////////////////////////
// Function declarations
template<typename T, int Headdim, bool Is_causal_tpl> // Renamed template parameter
void run_mha_fwd_(Flash_fwd_params &params, cudaStream_t stream);

//template<typename T, int Headdim, bool Is_causal>
//void run_mha_bwd_(Flash_bwd_params &params, cudaStream_t stream);
