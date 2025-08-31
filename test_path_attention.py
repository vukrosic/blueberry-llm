#!/usr/bin/env python3
"""
Quick test to verify PaTH attention integration works correctly
"""

import torch
import torch.nn as nn
from fla.layers.path_attn import PaTHAttention

class MultiHeadAttention(nn.Module):
    """Wrapper around PaTHAttention for compatibility with existing code"""
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        
        # Use PaTHAttention from flash-linear-attention
        self.path_attention = PaTHAttention(
            hidden_size=d_model,
            num_heads=n_heads,
            use_forget_gate=True,  # Enable forget gate for better performance
            use_qk_norm=True,      # Enable QK normalization
            use_low_rank_w=True,   # Use low-rank parameterization
            use_w_shortconv=True,  # Use short convolution
            conv_size=3,           # Convolution kernel size
            conv_bias=False        # No bias in convolution
        )

    def forward(self, x):
        # PaTHAttention expects (batch, seq_len, hidden_size) and returns (output, None, None)
        output, _, _ = self.path_attention(x)
        return output

def test_path_attention():
    """Test that PaTH attention works with our wrapper"""
    print("🧪 Testing PaTH attention integration...")
    
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Using device: {device}")
    
    if not torch.cuda.is_available():
        print("  ⚠️ CUDA not available, PaTH attention requires GPU. Skipping test.")
        return
    
    # Test configuration
    d_model = 256
    n_heads = 4
    max_seq_len = 128
    dropout = 0.1
    
    # Initialize the attention layer
    attention = MultiHeadAttention(
        d_model=d_model,
        n_heads=n_heads,
        max_seq_len=max_seq_len,
        dropout=dropout
    ).to(device)
    
    # Create test input
    batch_size = 2
    seq_len = 32
    x = torch.randn(batch_size, seq_len, d_model).to(device)
    
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