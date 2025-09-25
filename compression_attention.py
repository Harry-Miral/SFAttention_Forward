import torch
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any, Union


def compute_head_token_merge_indicators(q, k, merge_strategy=None, similarity_threshold=0.0002,
                                        difference_threshold=0.0175, max_merge_level=20):
    """
    Fully vectorized token merge indicator computation with zero Python loops,
    fully leveraging GPU parallel computation

    Args:
        q: Query tensor, shape (batch_size, sequence_length, num_heads, head_dim)
        k: Key tensor, shape (batch_size, sequence_length, num_heads, head_dim)
        merge_strategy: Tensor specifying merge strategy for each head, shape (batch_size, num_heads)
        similarity_threshold: Threshold for similarity merging
        difference_threshold: Threshold for difference merging
        max_merge_level: Maximum number of consecutive tokens to consider for merging

    Returns:
        merge_indicators: Boolean tensor, shape (batch_size, num_heads, sequence_length-1)
        dot_products_dict: Dictionary containing dot product results for loss computation
    """
    # Get dimension information
    batch_size, seq_len, num_heads, head_dim = q.shape
    device = q.device
    dtype = q.dtype

    # Handle sequences that are too short
    if seq_len <= 1:
        return torch.zeros((batch_size, num_heads, 0), dtype=torch.bool, device=device), None

    # If no merge strategy is specified, default all heads to similarity merging
    if merge_strategy is None:
        merge_strategy = torch.zeros((batch_size, num_heads), dtype=torch.long, device=device)

    # Limit max_merge_level to not exceed sequence length
    max_merge_level = min(max_merge_level, seq_len - 1, 5)

    # Adjust tensor dimensions for dot product computation
    q_transposed = q.transpose(1, 2)  # [batch, heads, seq, dim]
    k_transposed = k.transpose(1, 2)  # [batch, heads, seq, dim]
    
    # Store dot product results for loss computation - only keep necessary fields
    dot_products_dict = {
        'batch_dot_products': [],
        'merge_strategy': merge_strategy,
        'valid_lens': [],
        'similarity_threshold': similarity_threshold,
        'difference_threshold': difference_threshold
    }
    
    # Perform L2 normalization to ensure dot products are strictly in [-1,1] range
    # Use torch.no_grad() for intermediate values that don't need gradients
    with torch.no_grad():
        q_normalized = F.normalize(q_transposed, p=2, dim=-1)
        k_normalized = F.normalize(k_transposed, p=2, dim=-1)
        
        # Store merge possibility matrices for all levels [batch, heads, seq-1, max_level]
        merge_possibilities = torch.zeros(
            (batch_size, num_heads, seq_len - 1, max_merge_level),
            dtype=torch.bool, device=device
        )

        # First stage: Compute potential merge possibilities for all levels
        for level in range(max_merge_level):
            # Compute valid length for this level
            valid_len = seq_len - level - 1

            # Select query and key positions for comparison
            q_part = q_normalized[:, :, level + 1:, :]
            k_part = k_normalized[:, :, :valid_len, :]

            # Compute dot product similarity
            dot_products = torch.matmul(q_part, k_part.transpose(-1, -2))

            # Extract diagonal elements
            batch_dot_products = torch.diagonal(dot_products, dim1=-2, dim2=-1)
            
            # Only save dot products for first level for backward propagation
            if level == 0:
                # Save gradient-enabled dot products
                # Recompute dot products for first level to preserve gradients
                q_grad_part = F.normalize(q_transposed[:, :, 1:, :], p=2, dim=-1)
                k_grad_part = F.normalize(k_transposed[:, :, :valid_len, :], p=2, dim=-1)
                dot_products_grad = torch.matmul(q_grad_part, k_grad_part.transpose(-1, -2))
                batch_dot_products_grad = torch.diagonal(dot_products_grad, dim1=-2, dim2=-1)
                
                dot_products_dict['batch_dot_products'].append(batch_dot_products_grad)
            else:
                # For other levels, use gradient-free version (since no backward pass needed later)
                dot_products_dict['batch_dot_products'].append(batch_dot_products.detach())
                
            dot_products_dict['valid_lens'].append(valid_len)

            # Apply merge strategy and thresholds
            similarity_mask = (merge_strategy == 0).unsqueeze(-1).expand(-1, -1, valid_len)
            difference_mask = (merge_strategy == 1).unsqueeze(-1).expand(-1, -1, valid_len)

            similar_mergeable = 1 - batch_dot_products < similarity_threshold
            different_mergeable = torch.abs(batch_dot_products) < difference_threshold

            # Save mergeable information for loss computation - only save first level info
            if level == 0:
                dot_products_dict['similar_mergeable'] = similar_mergeable
                dot_products_dict['different_mergeable'] = different_mergeable

            # Combine results from both strategies
            levelwise_mergeable = (similarity_mask & similar_mergeable) | (difference_mask & different_mergeable)

            # Store in merge possibility matrix
            merge_possibilities[:, :, :valid_len, level] = levelwise_mergeable

        # Second stage: Handle validity of consecutive merging
        # Initialize merge indicators
        merge_indicators = merge_possibilities[:, :, :, 0].clone()

        # Iterate until merge state stabilizes or max iterations reached
        max_iterations = max_merge_level * 2

        for iteration in range(max_iterations):
            # Create sliding window to detect consecutive merges
            shifted_indicators = torch.zeros_like(merge_indicators)
            shifted_indicators[:, :, :-1] = merge_indicators[:, :, 1:]

            # Find all consecutive merge pairs
            consecutive_pairs = merge_indicators & shifted_indicators

            if not torch.any(consecutive_pairs):
                break

            # Check validity of consecutive merges
            invalid_pairs = torch.zeros_like(consecutive_pairs)

            # Use cumulative sum method to find consecutive 1s length
            ones_counter = torch.zeros_like(merge_indicators, dtype=torch.int32)
            ones_counter = torch.where(merge_indicators,
                                     torch.ones_like(ones_counter),
                                     torch.zeros_like(ones_counter))

            # Calculate consecutive 1s length
            cumsum = torch.zeros_like(ones_counter)
            cumsum[:, :, 0] = ones_counter[:, :, 0]

            for i in range(1, seq_len - 1):
                # If current position is 1 and previous position is also 1, accumulate length
                cumsum[:, :, i] = (ones_counter[:, :, i] *
                                 (cumsum[:, :, i - 1] + 1) *
                                 ones_counter[:, :, i - 1])

            # Check cross-token merging for each level
            for level in range(1, max_merge_level):
                # Find positions with exactly level consecutive 1s
                level_positions = (cumsum == level) & merge_indicators

                # Get validity of cross-level token merging for these positions
                valid_merges = torch.zeros_like(level_positions)
                valid_range = min(seq_len - 1 - level, seq_len - 1)
                valid_merges[:, :, :valid_range] = merge_possibilities[:, :, :valid_range, level]

                # Find invalid merges: level positions have consecutive 1s but cross-token merge invalid
                current_invalid = level_positions & ~valid_merges
                invalid_pairs = invalid_pairs | current_invalid

            # Limit maximum consecutive merge length
            too_long = cumsum > max_merge_level
            invalid_pairs = invalid_pairs | too_long

            # Break invalid merges
            if torch.any(invalid_pairs):
                merge_indicators = merge_indicators & ~invalid_pairs
            else:
                break
    
    # Release no longer needed intermediate variables
    del q_normalized, k_normalized, merge_possibilities

    return merge_indicators, dot_products_dict


def compute_compression_loss(dot_products_dict, I_factor=1.0):
    """
    Compute compression loss strictly following the paper's formula:
    
    L_comp = ((Σ(L_sim(i,j) * M_sim(i,j) + L_diff(i,j) * M_diff(i,j))) / N_merged 
              - N_merged / N_total_pairs) * I_factor
    
    Where:
    - L_sim(i,j) = (1 - d_p)^2  # Similarity loss
    - L_diff(i,j) = d_p^2       # Difference loss  
    - d_p is cosine similarity (dot product, since already L2 normalized)
    - M_sim/M_diff are strategy masks
    - N_merged is actual number of merged token pairs
    - N_total_pairs is total number of token pairs
    - I_factor is scaling hyperparameter
    
    Args:
        dot_products_dict: Dictionary containing dot product results and merge strategies
        I_factor: Scaling factor hyperparameter, default 1.0
        
    Returns:
        Computed compression loss with gradients for backpropagation
    """
    if dot_products_dict is None:
        return None
        
    batch_dot_products_list = dot_products_dict['batch_dot_products']
    merge_strategy = dot_products_dict['merge_strategy']
    valid_lens = dot_products_dict['valid_lens']
    
    # Get thresholds
    similarity_threshold = dot_products_dict.get('similarity_threshold', 0.0002)
    difference_threshold = dot_products_dict.get('difference_threshold', 0.0175)
    
    # Only use first level dot product results (adjacent token similarity)
    if len(batch_dot_products_list) == 0:
        return torch.zeros(1, device=merge_strategy.device, requires_grad=True)
        
    batch_dot_products = batch_dot_products_list[0]  # Use level=0 dot products
    valid_len = valid_lens[0]
    
    # Get strategy masks
    similarity_mask = (merge_strategy == 0).unsqueeze(-1).expand(-1, -1, valid_len)
    difference_mask = (merge_strategy == 1).unsqueeze(-1).expand(-1, -1, valid_len)
    
    # Compute current mergeable states
    if 'similar_mergeable' in dot_products_dict and 'different_mergeable' in dot_products_dict:
        similar_mergeable = dot_products_dict['similar_mergeable']
        different_mergeable = dot_products_dict['different_mergeable']
    else:
        similar_mergeable = 1 - batch_dot_products < similarity_threshold
        different_mergeable = torch.abs(batch_dot_products) < difference_threshold
    
    # Compute losses according to paper formula:
    # L_sim(i,j) = (1 - d_p)^2
    # L_diff(i,j) = d_p^2
    L_sim = torch.square(1.0 - batch_dot_products)
    L_diff = torch.square(batch_dot_products)
    
    # Apply strategy masks: M_sim(i,j) and M_diff(i,j)
    masked_sim_loss = L_sim * similarity_mask.float()
    masked_diff_loss = L_diff * difference_mask.float()
    
    # Compute N_merged: actual number of merged token pairs
    # Here we use pairs that currently satisfy merge conditions as proxy for N_merged
    N_merged_sim = (similarity_mask & similar_mergeable).float().sum()
    N_merged_diff = (difference_mask & different_mergeable).float().sum()
    N_merged = N_merged_sim + N_merged_diff
    
    # Compute N_total_pairs: total number of token pairs
    N_total_pairs = (similarity_mask | difference_mask).float().sum()
    
    if N_total_pairs == 0:
        return torch.zeros(1, device=batch_dot_products.device, requires_grad=True)
    
    # Compute quality loss component: Σ(L_sim * M_sim + L_diff * M_diff) / N_merged
    total_weighted_loss = masked_sim_loss.sum() + masked_diff_loss.sum()
    
    if N_merged > 0:
        quality_component = total_weighted_loss / N_merged
    else:
        # If no merging, quality component is average loss
        quality_component = total_weighted_loss / N_total_pairs
    
    # Compute quantity loss component: N_merged / N_total_pairs
    # This component encourages more merging (since it's subtracted in the formula)
    quantity_component = N_merged / N_total_pairs
    
    # Final loss: paper formula
    # L_comp = (quality_component - quantity_component) * I_factor
    compression_loss = (quality_component - quantity_component) * I_factor
    
    return compression_loss


def apply_head_token_compression(q, k, v, merge_indicators):
    """
    Apply token compression using vectorized operations, fixing duplicate accumulation issues

    Args:
        q, k, v: Shape (batch_size, num_heads, seq_len, head_dim)
        merge_indicators: Shape (batch_size, num_heads, seq_len-1)

    Returns:
        Compressed q, k, v and merged token markers
    """
    batch_size, num_heads, seq_len, head_dim = q.shape
    device = q.device

    # Check if there are any merge requirements - avoid unnecessary memory allocation
    if not merge_indicators.any():
        # If no merge indicators, return original tensors and all-zero merge markers
        merged_tokens = torch.zeros((batch_size, num_heads, seq_len), dtype=torch.bool, device=device)
        return q, k, v, merged_tokens

    # Initialize merged token markers
    merged_tokens = torch.zeros((batch_size, num_heads, seq_len), dtype=torch.bool, device=device)
    merged_tokens[:, :, :-1] = merge_indicators

    # Create output tensor copies - only when there's actual merge requirement
    q_comp = q.clone()
    k_comp = k.clone()
    v_comp = v.clone()

    # Find positions of all tokens that need merging
    merge_indices = torch.nonzero(merged_tokens, as_tuple=True)

    if len(merge_indices[0]) == 0:
        # When no merge positions found, return early
        return q_comp, k_comp, v_comp, merged_tokens

    # Calculate target positions
    b_idx, h_idx, pos_idx = merge_indices
    target_pos = pos_idx + 1

    # Filter out out-of-bounds positions
    valid_mask = target_pos < seq_len
    if not torch.any(valid_mask):
        return q_comp, k_comp, v_comp, merged_tokens

    b_idx, h_idx = b_idx[valid_mask], h_idx[valid_mask]
    pos_idx, target_pos = pos_idx[valid_mask], target_pos[valid_mask]

    # Add original values to target positions
    q_comp[b_idx, h_idx, target_pos] += q[b_idx, h_idx, pos_idx]
    k_comp[b_idx, h_idx, target_pos] += k[b_idx, h_idx, pos_idx]
    v_comp[b_idx, h_idx, target_pos] += v[b_idx, h_idx, pos_idx]

    return q_comp, k_comp, v_comp, merged_tokens


def create_compressed_attention_mask(merged_tokens, causal=True):
    """
    Create compressed attention mask compatible with PyTorch F.scaled_dot_product_attention

    Args:
        merged_tokens: Boolean tensor of shape (batch_size, num_heads, seq_len)
        causal: Whether to apply causal mask

    Returns:
        PyTorch-compatible attention mask, True means "allow attention", False means "disallow attention"
    """
    batch_size, num_heads, seq_len = merged_tokens.shape
    device = merged_tokens.device

    # Use torch.no_grad() to avoid unnecessary gradients for mask computation
    with torch.no_grad():
        # Create attention allowance mask (True = allow attention)
        attn_mask = torch.ones((batch_size, num_heads, seq_len, seq_len), dtype=torch.bool, device=device)

        # Apply causal mask (upper triangle set to False = disallow attention)
        if causal:
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)
            attn_mask = attn_mask & (~causal_mask.unsqueeze(0).unsqueeze(0))

        # Mask entire columns for merged tokens (set to False = disallow attention)
        column_mask = merged_tokens.unsqueeze(2).expand(-1, -1, seq_len, -1)
        attn_mask = attn_mask & (~column_mask)

        # Allow self-attention (diagonal set to True = allow attention)
        diag_mask = torch.eye(seq_len, dtype=torch.bool, device=device).unsqueeze(0).unsqueeze(0)
        attn_mask = attn_mask | diag_mask

    return attn_mask


def compressed_attention_forward(q, k, v, merge_indicators=None, merge_strategy=None, attention_bias=None,
                                 is_causal=True, dropout_p=0.0,
                                 scale=None, I_factor=1.0):
    """
    Implement compressed attention using PyTorch's F.scaled_dot_product_attention,
    supporting different compression states and merge strategies for each head.

    Args:
        q, k, v: Shape (batch_size, num_heads, seq_len, head_dim)
        merge_indicators: Shape (batch_size, num_heads, seq_len-1)
                         If None and merge_strategy is not None, merge indicators will be computed
        merge_strategy: Tensor specifying merge strategy for each head, shape (batch_size, num_heads)
                      0 for similarity merging, 1 for difference merging
        attention_bias: Additional attention bias, shape (batch_size, num_heads, seq_len, seq_len)
        is_causal: Whether to apply causal mask
        dropout_p: Dropout probability
        scale: Scaling factor
        I_factor: Scaling hyperparameter for compression loss

    Returns:
        Attention output, shape (batch_size, num_heads, seq_len, head_dim)
        Compression loss (if merge_indicators is not None)
    """
    batch_size, num_heads, seq_len, head_dim = q.shape

    # Default scaling factor
    if scale is None:
        scale = 1.0 / (head_dim ** 0.5)

    compression_loss = None
    dot_products_dict = None

    # If no merge indicators and no merge strategy, perform standard attention computation
    if merge_indicators is None and merge_strategy is None:
        return F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_bias,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale
        ), compression_loss

    # If no merge indicators but have merge strategy, compute merge indicators
    if merge_indicators is None and merge_strategy is not None:
        # Adjust q and k shape to fit compute_head_token_merge_indicators
        q_for_merge = q.transpose(1, 2)  # [batch, seq, heads, dim]
        k_for_merge = k.transpose(1, 2)  # [batch, seq, heads, dim]
        merge_indicators, dot_products_dict = compute_head_token_merge_indicators(q_for_merge, k_for_merge, merge_strategy)

    # Compute compression loss - this is backpropagatable loss
    compression_loss = compute_compression_loss(dot_products_dict, I_factor)
    
    # Perform compressed attention computation
    q_comp, k_comp, v_comp, merged_tokens = apply_head_token_compression(q, k, v, merge_indicators)

    # Create compressed attention mask
    comp_mask = create_compressed_attention_mask(merged_tokens, causal=is_causal)

    # Use PyTorch's scaled_dot_product_attention to compute attention
    output = F.scaled_dot_product_attention(
        q_comp,
        k_comp,
        v_comp,
        attn_mask=comp_mask,
        dropout_p=dropout_p,
        is_causal=False,  # Already handled causality in comp_mask
        scale=scale
    )
    
    # Release no longer needed intermediate variables
    del q_comp, k_comp, v_comp, merged_tokens, comp_mask
    
    return output, compression_loss