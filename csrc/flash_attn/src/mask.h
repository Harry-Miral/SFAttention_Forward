/******************************************************************************
 * Copyright (c) 2025, xxx.
 ******************************************************************************/

#pragma once

#include <cute/tensor.hpp>

namespace flash {

using namespace cute;

enum class AttentionPass {
    Forward,
    Backward
};

template <bool Is_causal, bool Is_local, bool Has_alibi>
struct Mask {

    const int max_seqlen_k, max_seqlen_q;
    const int window_size_left, window_size_right;
    const float alibi_slope;
    // Compressed attention member variables
    const int* row_remapping_q_prime_ptr;
    const int* col_remapping_k_prime_ptr;
    const bool compress_attention;
    // Global total token counts
    const int total_q_prime;
    const int total_k_prime;

    __forceinline__ __device__ Mask(const int max_seqlen_k, const int max_seqlen_q,
                                    const int window_size_left, const int window_size_right,
                                    const float alibi_slope=0.f,
                                    const int* row_remapping_q_prime_ptr=nullptr,
                                    const int* col_remapping_k_prime_ptr=nullptr,
                                    const bool compress_attention=false,
                                    const int total_q_prime=0,
                                    const int total_k_prime=0)
        : max_seqlen_k(max_seqlen_k)
        , max_seqlen_q(max_seqlen_q)
        , window_size_left(window_size_left)
        , window_size_right(window_size_right)
        , alibi_slope(!Has_alibi ? 0.0 : alibi_slope)
        , row_remapping_q_prime_ptr(row_remapping_q_prime_ptr)
        , col_remapping_k_prime_ptr(col_remapping_k_prime_ptr)
        , compress_attention(compress_attention)
        , total_q_prime(total_q_prime)
        , total_k_prime(total_k_prime)
    {};

    template <bool Causal_mask=false, bool Is_even_MN=true, typename Engine, typename Layout>
    __forceinline__ __device__ void apply_mask(Tensor<Engine, Layout> &tensor_,
                                            const int col_idx_offset_,
                                            const int row_idx_offset,
                                            const int warp_row_stride,
                                            const int global_q_prime_offset = 0,
                                            const int global_k_prime_offset = 0) {
        static_assert(!(Is_causal && Is_local), "Cannot be both causal and local");
        static_assert(Layout::rank == 3, "Only support 3D Tensor");
        static_assert(decltype(size<0>(tensor_))::value == 4, "First dimension must be 4");
        constexpr bool Need_masking = Has_alibi || Is_causal || Is_local || !Is_even_MN;

        if constexpr (Need_masking) {
            Tensor tensor = make_tensor(tensor_.data(), flash::convert_layout_acc_rowcol(tensor_.layout()));

            if (this->compress_attention && this->row_remapping_q_prime_ptr != nullptr && this->col_remapping_k_prime_ptr != nullptr) {

                const int lane_id = threadIdx.x % 32;
                const int col_idx_offset = col_idx_offset_ + (lane_id % 4) * 2;

                #pragma unroll
                for (int mi = 0; mi < size<0, 1>(tensor); ++mi) {
                    const int row_idx_base = row_idx_offset + mi * warp_row_stride;

                    #pragma unroll
                    for (int i = 0; i < size<0, 0>(tensor); ++i) {
                        const int row_idx_prime_local = row_idx_base + i * 8;

                        #pragma unroll
                        for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
                            const int col_idx_base = col_idx_offset + nj * 8;

                            #pragma unroll
                            for (int j = 0; j < size<1, 0>(tensor); ++j) {
                                const int col_idx_prime_local = col_idx_base + j;

                                const int row_idx_prime_global = global_q_prime_offset + row_idx_prime_local;
                                const int col_idx_prime_global = global_k_prime_offset + col_idx_prime_local;

                                if (row_idx_prime_local < this->max_seqlen_q &&
                                    col_idx_prime_local < this->max_seqlen_k &&
                                    row_idx_prime_global >= 0 &&
                                    row_idx_prime_global < this->total_q_prime &&
                                    col_idx_prime_global >= 0 &&
                                    col_idx_prime_global < this->total_k_prime) {

                                    int original_row_idx = this->row_remapping_q_prime_ptr[row_idx_prime_global];
                                    int original_col_idx = this->col_remapping_k_prime_ptr[col_idx_prime_global];

                                    if (original_row_idx <= original_col_idx) {
                                        tensor(make_coord(i, mi), make_coord(j, nj)) = -INFINITY;
                                    }
                                } else {
                                    tensor(make_coord(i, mi), make_coord(j, nj)) = -INFINITY;
                                }
                            }
                        }
                    }
                }
            } else { // Standard Attention Path
                const int lane_id = threadIdx.x % 32;
                const int col_idx_offset = col_idx_offset_ + (lane_id % 4) * 2;
                #pragma unroll
                for (int mi = 0; mi < size<0, 1>(tensor); ++mi) {
                    const int row_idx_base = row_idx_offset + mi * warp_row_stride;
                    #pragma unroll
                    for (int i_loop = 0; i_loop < size<0, 0>(tensor); ++i_loop) {
                        const int row_idx = row_idx_base + i_loop * 8;
                        const int col_idx_limit_left = std::max(0, row_idx + max_seqlen_k - max_seqlen_q - window_size_left);
                        const int col_idx_limit_right = std::min(max_seqlen_k, row_idx + 1 + max_seqlen_k - max_seqlen_q + window_size_right);
                        #pragma unroll
                        for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
                            const int col_idx_base = col_idx_offset + nj * 8;
                            #pragma unroll
                            for (int j_loop = 0; j_loop < size<1, 0>(tensor); ++j_loop) {
                                const int col_idx = col_idx_base + j_loop;
                                if constexpr (Has_alibi) {
                                    if constexpr (Is_causal) {
                                        tensor(make_coord(i_loop, mi), make_coord(j_loop, nj)) += alibi_slope * col_idx;
                                    } else {
                                        tensor(make_coord(i_loop, mi), make_coord(j_loop, nj)) -= alibi_slope * abs(row_idx + max_seqlen_k - max_seqlen_q - col_idx);
                                    }
                                }
                                if constexpr (Causal_mask) {
                                    if (col_idx >= col_idx_limit_right) {
                                        tensor(make_coord(i_loop, mi), make_coord(j_loop, nj)) = -INFINITY;
                                    }
                                }
                                if constexpr (Is_local) {
                                    if (col_idx >= col_idx_limit_right || col_idx < col_idx_limit_left) {
                                        tensor(make_coord(i_loop, mi), make_coord(j_loop, nj)) = -INFINITY;
                                    }
                                }
                                if constexpr (!Causal_mask && !Is_local && !Is_even_MN) {
                                    if (col_idx >= max_seqlen_k) {
                                        tensor(make_coord(i_loop, mi), make_coord(j_loop, nj)) = -INFINITY;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    };
};

} // namespace flash