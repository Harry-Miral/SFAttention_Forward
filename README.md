# SFAttention

An efficient attention mechanism implementation for Transformer architectures.

## System Requirements

- **Python**: 3.9
- **CUDA**: 12.1  
- **PyTorch**: 2.3.1
- **Hardware**: Tested on A100 GPU

### Hardware Compatibility

- **Recommended**: A100 GPU with SM_80 architecture or higher  
- **Lower architectures**: Devices below A100's SM_80 architecture are untested and may encounter limitations. Manual configuration adjustments may be required.
- **Higher architectures**: Should work theoretically, but may require configuration tweaks if issues arise.

**Note**: The Python implementation should work on most CUDA-capable GPUs. The specific architecture requirements mentioned above primarily apply to the CUDA implementation.

## Quick Start

### Python Implementation

**Usage:**
Replace attention calls in your Transformer architecture with the `compressed_attention_forward` function from `compression_attention.py`:

```python
from compression_attention import compressed_attention_forward

# Basic usage - replace your existing attention mechanism
# Note: Input tensors should have shape (batch_size, num_heads, seq_len, head_dim)
output, compression_loss = compressed_attention_forward(q, k, v)

# Advanced usage with compression strategies
merge_strategy = torch.zeros((batch_size, num_heads), dtype=torch.long)  # 0: similarity, 1: difference
output, compression_loss = compressed_attention_forward(
    q, k, v, 
    merge_strategy=merge_strategy,
    is_causal=True,  # For causal attention
    dropout_p=0.1    # Dropout probability
)

# If you need to use the compression loss for training
if compression_loss is not None:
    total_loss = your_main_loss + compression_loss
```

**Return Values:**
- `output`: Compressed attention output with shape (batch_size, num_heads, seq_len, head_dim)  
- `compression_loss`: Compression loss for training (can be None)

**Note on Additional Parameters**: You may notice that some internal functions return multiple values that were originally intended for backward propagation. These are preserved for learning and research purposes, but can be safely ignored in forward-only usage.

**Features:**
- Intuitive and easy to understand implementation
- Ready for learning and basic testing
- Slower execution compared to CUDA implementation
- No compilation required

### CUDA Implementation

The CUDA implementation offers better performance but requires longer compilation time.

**Installation:**
Navigate to the project folder and run:
```bash
pip install -e .
```
**Note**: Compilation time is considerably long.

- See the appendix for detailed CUDA implementation information
- Performance optimized for production use

## Current Release Notes

**Forward-only Implementation**: This repository currently provides the complete forward propagation implementation.

**Why Forward-only?**: To protect against academic misconduct and plagiarism during the review process (we have experienced article theft previously). This is a necessary protective measure.

**Research Parameters**: You may notice additional return values and intermediate computations in the code that were originally designed for backward propagation research. These are preserved for:
- Educational purposes and algorithm understanding
- Research analysis and debugging
- Future compatibility when full implementation is released

**For Current Users**: The forward implementation provides:
- Full compression attention functionality
- Training-ready compression loss
- Performance evaluation capabilities
- Algorithm learning and understanding

**Full Release**: Complete implementation including optimized backward propagation will be open-sourced immediately upon paper acceptance.

## Configuration Parameters

The implementation provides several tunable parameters for optimization:

```python
# Key parameters in compute_head_token_merge_indicators()
similarity_threshold = 0.0002      # Threshold for similarity-based merging
difference_threshold = 0.0175      # Threshold for difference-based merging  
max_merge_level = 20              # Maximum consecutive tokens to consider
```

**Parameter Tuning Tips:**
- Lower `similarity_threshold` = more aggressive similarity merging
- Lower `difference_threshold` = more aggressive difference merging  
- Higher `max_merge_level` = longer compression chains (more memory usage)
- Strategy selection per head allows fine-grained control

## Contributing

We welcome contributions and feedback from the community. Please feel free to open issues or submit pull requests.

## Citation

If you use this work, please cite our paper (citation will be updated upon publication):

```bibtex
@article{your_paper,
  title={Your Paper Title},
  author={Your Name},
  journal={Conference/Journal Name},
  year={2024}
}
```

## License
