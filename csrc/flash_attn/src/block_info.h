/******************************************************************************
 * Copyright (c) 2025, xxx.
 ******************************************************************************/

#pragma once

namespace flash {

////////////////////////////////////////////////////////////////////////////////////////////////////

template<bool Varlen=true>
struct BlockInfo {

    const int sum_s_q;
    const int sum_s_k;
    const int actual_seqlen_q;
    const int actual_seqlen_k;
    
    // Standard FlashAttention members, initialized to 0 for compress_attention path
    const int leftpad_k;
    const int seqlen_k_cache;

    template<typename Params>
    __device__ BlockInfo(const Params &params, const int bidb)
        : sum_s_q([&] {
            if (!Varlen) return -1;
            // START: FINAL SIMPLIFICATION - Unify Varlen logic
            // For ALL Varlen modes (compressed or not), sum_s_q comes from cu_seqlens_q.
            // In the compress path, flash_api.cpp ensures params.cu_seqlens_q points to cu_seqlens_q_prime.
            return params.cu_seqlens_q == nullptr ? -1 : params.cu_seqlens_q[bidb];
            // END: FINAL SIMPLIFICATION
        }())
        , sum_s_k([&] {
            if (!Varlen) return -1;
            // START: FINAL SIMPLIFICATION - Unify Varlen logic
            // For ALL Varlen modes (compressed or not), sum_s_k comes from cu_seqlens_k.
            // In the compress path, flash_api.cpp ensures params.cu_seqlens_k points to cu_seqlens_kv_prime.
            return params.cu_seqlens_k == nullptr ? -1 : params.cu_seqlens_k[bidb];
            // END: FINAL SIMPLIFICATION
        }())
        , actual_seqlen_q([&] {
            if (!Varlen) return params.seqlen_q;
            // START: FINAL SIMPLIFICATION - Unify Varlen logic
            // The logic for actual sequence length is identical for all Varlen cases.
            return params.cu_seqlens_q == nullptr ? params.seqlen_q : (params.cu_seqlens_q[bidb + 1] - sum_s_q);
            // END: FINAL SIMPLIFICATION
        }())
        , actual_seqlen_k([&] {
            if (!Varlen) return params.seqlen_k;
            if (params.compress_attention) {
                // START: FINAL SIMPLIFICATION - Unify Varlen logic
                // In compress mode, the logic is the same as standard Varlen.
                return params.cu_seqlens_k == nullptr ? params.seqlen_k : (params.cu_seqlens_k[bidb + 1] - sum_s_k);
                // END: FINAL SIMPLIFICATION
            } else {
                const int cache_len = (params.cu_seqlens_k == nullptr ? params.seqlen_k : (params.is_seqlens_k_cumulative ? params.cu_seqlens_k[bidb + 1] - sum_s_k : params.cu_seqlens_k[bidb])) - (params.leftpad_k == nullptr ? 0 : params.leftpad_k[bidb]);
                return params.seqused_k ? params.seqused_k[bidb] - (params.leftpad_k == nullptr ? 0 : params.leftpad_k[bidb]) : cache_len + (params.knew_ptr == nullptr ? 0 : params.seqlen_knew);
            }
        }())
        , leftpad_k(params.compress_attention ? 0 : (params.leftpad_k == nullptr ? 0 : params.leftpad_k[bidb]))
        , seqlen_k_cache(params.compress_attention ? 0 : ((!Varlen || params.cu_seqlens_k == nullptr ? params.seqlen_k : (params.is_seqlens_k_cumulative ? params.cu_seqlens_k[bidb + 1] - sum_s_k : params.cu_seqlens_k[bidb])) - leftpad_k))
    {}

    template <typename index_t>
    __forceinline__ __device__ index_t q_offset(const index_t batch_stride, const index_t row_stride, const int bidb) const {
        // sum_s_q == -1 indicates non-varlen mode.
        // In this case, we use the batch stride.
        // Otherwise, for varlen (both standard and compressed), we use the cumulative sum as the row offset.
        return sum_s_q == -1 ? bidb * batch_stride : uint32_t(sum_s_q) * row_stride;
    }

    template <typename index_t>
    __forceinline__ __device__ index_t k_offset(const index_t batch_stride, const index_t row_stride, const int bidb) const {
        // sum_s_k == -1 indicates non-varlen mode.
        return sum_s_k == -1 ? bidb * batch_stride + leftpad_k * row_stride : uint32_t(sum_s_k + leftpad_k) * row_stride;
    }

};

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace flash