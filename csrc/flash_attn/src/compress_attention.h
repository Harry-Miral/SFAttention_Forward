/******************************************************************************
 * Copyright (c) 2025, xxx.
 ******************************************************************************/

#pragma once

#include "flash.h" // For Flash_fwd_params
#include <cuda_runtime.h>

// ======================= 新增 V8.0 =======================
// 声明一个新的CUDA核函数启动器，用于计算 LSE' = m' + log(l')
void run_compute_lse_prime_kernel(
    const float* m_prime_ptr,
    const float* l_prime_ptr,
    float* lse_prime_ptr,
    int total_q_prime,
    cudaStream_t stream,
    const float softmax_scale // ADD this parameter
);
// ======================= 修复结束 =======================
// ======================= 新增：高效预计算与融合 Gather 内核 =======================

// ======================= 用下面的声明替换旧的 run_scan_for_prime_seqlens =======================
/**
 * @brief [主机端包装函数] 调用 CUB 来对 per-bh-counts 数组进行并行扫描和归约。
 *        这是一个纯 CUDA C++ 函数，不依赖任何 PyTorch API。
 * @param d_temp_storage 指向 CUB 所需的临时设备内存的指针。
 * @param temp_storage_bytes 临时内存的大小。
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
);
// ======================= 修改结束 =======================

/**
 * @brief [主机端包装函数] 调用 CUB 在 GPU 上计算 per-bh-counts 数组的最大值。
 */
void run_reduce_max_for_prime_seqlens(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    const int* d_per_bh_counts_q,
    const int* d_per_bh_counts_kv,
    int* d_max_seqlen_q_prime, // 改为设备指针
    int* d_max_seqlen_k_prime, // 改为设备指针
    int num_bh_items,
    cudaStream_t stream
);

/**
 * @brief [新实现] 步骤 1: 并行计算每个 batch-head 中 Q' 和 K'/V' 的长度。
 *        这是一个高度优化的轻量级内核，每个线程块处理一个 batch-head。
 */
void run_precompute_prime_lengths(Flash_fwd_params &params, cudaStream_t stream,
                                  int* d_per_bh_counts_q_prime, int* d_per_bh_counts_kv_prime);

/**
 * @brief [新实现] 步骤 2: 融合的 Gather 内核的启动器。
 *        它会根据 max_seqlen_k 选择一个合适的模板化内核实例来启动，以优化共享内存。
 *        取代了旧的 gather_qkv_prime_data 函数。
 */
void run_fused_gather_kernel(Flash_fwd_params &params, cudaStream_t stream);

// ======================= 修改结束 =======================

// Stage 1: Generate Q', K', V' data, their mappings, and cumulative sequence lengths.
// This function will also update params.total_q_prime, params.total_k_prime,
// params.max_seqlen_q_prime, params.max_seqlen_k_prime.
//void run_qkv_prime_generation_stage(Flash_fwd_params &params, cudaStream_t stream);

// Stage 1 - Part A: Calculate metadata for Q', K', V'.
// This function will update params.total_q_prime, params.total_k_prime,
// params.max_seqlen_q_prime, params.max_seqlen_k_prime and fill
// params.cu_seqlens_q_prime_ptr, params.cu_seqlens_kv_prime_ptr.
// It DOES NOT perform data gathering.
void calculate_qkv_prime_metadata(Flash_fwd_params &params, cudaStream_t stream);

// Stage 1 - Part B: Gather Q', K', V' data and their mappings.
// Assumes metadata in params (total_q_prime etc.) is correctly populated
// and q_prime_ptr, k_prime_ptr etc. point to correctly sized allocated memory.
void gather_qkv_prime_data(Flash_fwd_params &params, cudaStream_t stream);

// Stage 3: Calculate diagonal scores (s_k_prime) for rows corresponding to Q'.
void run_calculate_sk_prime_kernel(Flash_fwd_params &params, cudaStream_t stream);

// Stage 4: Combine results from rectangular FlashAttention and diagonal scores.
void run_compress_attn_combine_kernel(Flash_fwd_params &params, cudaStream_t stream);

// ======================= 新增内容 (V2 优化版本) =======================
// Stage 3 的优化版本，使用预计算的 token->batch 映射
void run_calculate_sk_prime_kernel_v2(Flash_fwd_params &params, cudaStream_t stream);

// Stage 4 的优化版本，使用预计算的 token->batch 映射
void run_compress_attn_combine_kernel_v2(Flash_fwd_params &params, cudaStream_t stream);
// ======================= 新增结束 =======================


// Forward pass for rectangular FlashAttention that outputs intermediate values m' and l'
// This declaration assumes it will be defined elsewhere, possibly by modifying

// Helper function (typically on host) to compute cumulative sums.
// This might be called inside run_qkv_prime_generation_stage or separately.
// For now, declaring it here if it's a utility for compress_attention.cu.
void compute_cumulative_lengths_from_per_bh_counts(
    const int* d_per_bh_counts_q, // device pointer to per-batch-head Q' lengths
    const int* d_per_bh_counts_kv, // device pointer to per-batch-head K'/V' lengths
    int* d_cu_seqlens_q_prime,    // device pointer to output cumulative Q' lengths
    int* d_cu_seqlens_kv_prime,   // device pointer to output cumulative K'/V' lengths
    int& h_total_q_prime,         // host reference to store total Q' elements
    int& h_total_kv_prime,        // host reference to store total K'/V' elements
    int& h_max_seqlen_q_prime,    // host reference to store max Q' length per BH
    int& h_max_seqlen_kv_prime,   // host reference to store max K'/V' length per BH
    int batch_size,
    int num_heads,
    cudaStream_t stream
);

void run_map_q_indices_to_batch(
    const int* cu_seqlens_q_orig_ptr,
    int* q_orig_to_batch_idx_map_ptr,
    const int batch_size,
    const int total_q_orig,
    cudaStream_t stream
);

