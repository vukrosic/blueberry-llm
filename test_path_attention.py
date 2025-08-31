#!/usr/bin/env python3
"""
Quick test to verify PaTH attention integration works correctly
"""

import torch
import torch.nn as nn
from train_distributed_path_llm import MultiHeadAttention, ModelConfig

def test_path_attention():
    """Test that PaTH attention works with our wrapper"""
    print("🧪 Testing PaTH attention integration...")
    
    # Create a small test configuration
    config = ModelConfig()
    config.d_model = 256
    config.n_heads = 4
    config.max_seq_len = 128
    config.dropout = 0.1
    
    # Initialize the attention layer
    attention = MultiHeadAttention(
        d_model=config.d_model,
        n_heads=config.n_heads,
        max_seq_len=config.max_seq_len,
        dropout=config.dropout
    )
    
    # Create test input
    batch_size = 2
    seq_len = 32
    x = torch.randn(batch_size, seq_len, config.d_model)
    
    print(f"  Input shape: {x.shape}")
    
    # Test forward pass
    with torch.no_grad():
        output = attention(x)
        print(f"  Output shape: {output.shape}")
        
        # Verify output shape matches input
        assert output.shape == x.shape, f"Shape mismatch: {output.shape} != {x.shape}"
        
        # Verify output is not NaN or Inf
        assert torch.isfinite(output).all(), "Output contains NaN or Inf values"
        
    print("  ✅ Forward pass successful!")
    
    # Test gradient computation
    x.requires_grad_(True)
    output = attention(x)
    loss = output.sum()
    loss.backward()
    
    assert x.grad is not None, "Gradients not computed"
    assert torch.isfinite(x.grad).all(), "Gradients contain NaN or Inf"
    
    print("  ✅ Backward pass successful!")
    
    # Print some statistics
    print(f"  Output mean: {output.mean().item():.6f}")
    print(f"  Output std: {output.std().item():.6f}")
    print(f"  Gradient norm: {x.grad.norm().item():.6f}")
    
    print("🎉 PaTH attention integration test passed!")

if __name__ == "__main__":
    test_path_attention()