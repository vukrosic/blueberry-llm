#!/usr/bin/env python3
"""
Compare the old MultiHeadAttention with PaTH attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from fla.layers.path_attn import PaTHAttention
import time

class OldMultiHeadAttention(nn.Module):
    """Original attention implementation"""
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout

    def forward(self, x):
        batch_size, seq_len = x.size(0), x.size(1)

        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.n_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        Q, K, V = qkv[0], qkv[1], qkv[2]

        attn_output = F.scaled_dot_product_attention(
            Q, K, V, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        return self.w_o(attn_output)

class NewMultiHeadAttention(nn.Module):
    """PaTH attention implementation"""
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        
        self.path_attention = PaTHAttention(
            hidden_size=d_model,
            num_heads=n_heads,
            use_forget_gate=True,
            use_qk_norm=True,
            use_low_rank_w=True,
            use_w_shortconv=True,
            conv_size=3,
            conv_bias=False
        )

    def forward(self, x):
        output, _, _ = self.path_attention(x)
        return output

def compare_attentions():
    """Compare old vs new attention mechanisms"""
    print("🔍 Comparing Old vs PaTH Attention...")
    
    if not torch.cuda.is_available():
        print("  ⚠️ CUDA not available, skipping comparison.")
        return
    
    device = torch.device('cuda:0')
    
    # Configuration
    d_model = 512
    n_heads = 8
    max_seq_len = 1024
    batch_size = 4
    seq_len = 256
    
    print(f"  Config: {d_model}d, {n_heads}h, seq_len={seq_len}, batch={batch_size}")
    
    # Create models
    old_attn = OldMultiHeadAttention(d_model, n_heads, max_seq_len).to(device)
    new_attn = NewMultiHeadAttention(d_model, n_heads, max_seq_len).to(device)
    
    # Count parameters
    old_params = sum(p.numel() for p in old_attn.parameters())
    new_params = sum(p.numel() for p in new_attn.parameters())
    
    print(f"  Old attention parameters: {old_params:,}")
    print(f"  PaTH attention parameters: {new_params:,}")
    print(f"  Parameter ratio: {new_params/old_params:.2f}x")
    
    # Create test input
    x = torch.randn(batch_size, seq_len, d_model).to(device)
    
    # Warm up
    with torch.no_grad():
        _ = old_attn(x)
        _ = new_attn(x)
    
    # Benchmark forward pass
    torch.cuda.synchronize()
    
    # Old attention timing
    start_time = time.time()
    with torch.no_grad():
        for _ in range(10):
            old_output = old_attn(x)
    torch.cuda.synchronize()
    old_time = (time.time() - start_time) / 10
    
    # New attention timing
    start_time = time.time()
    with torch.no_grad():
        for _ in range(10):
            new_output = new_attn(x)
    torch.cuda.synchronize()
    new_time = (time.time() - start_time) / 10
    
    print(f"  Old attention time: {old_time*1000:.2f}ms")
    print(f"  PaTH attention time: {new_time*1000:.2f}ms")
    print(f"  Speed ratio: {old_time/new_time:.2f}x")
    
    # Check output shapes
    print(f"  Old output shape: {old_output.shape}")
    print(f"  PaTH output shape: {new_output.shape}")
    
    # Basic statistics
    print(f"  Old output mean: {old_output.mean().item():.6f}, std: {old_output.std().item():.6f}")
    print(f"  PaTH output mean: {new_output.mean().item():.6f}, std: {new_output.std().item():.6f}")
    
    print("✅ Comparison completed!")

if __name__ == "__main__":
    compare_attentions()