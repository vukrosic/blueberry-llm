#!/usr/bin/env python3
"""
Simple verification script to test PaTH attention integration
Run this in your environment to verify everything works
"""

import torch
import os

def main():
    print("🔍 Verifying PaTH attention integration...")
    
    # Check CUDA
    if torch.cuda.is_available():
        print(f"  ✅ CUDA available: {torch.cuda.get_device_name(0)}")
        device = torch.device('cuda:0')
    else:
        print("  ⚠️ CUDA not available, using CPU")
        device = torch.device('cpu')
    
    # Test 1: Import PaTH attention
    try:
        from fla.layers.path_attn import PaTHAttention
        print("  ✅ PaTHAttention import successful")
    except ImportError as e:
        print(f"  ❌ PaTHAttention import failed: {e}")
        return False
    
    # Test 2: Import training script components
    try:
        # Set minimal environment
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")
        
        from train_distributed_path_llm import ModelConfig, MultiHeadAttention
        print("  ✅ Training script imports successful")
    except Exception as e:
        print(f"  ❌ Training script import failed: {e}")
        return False
    
    # Test 3: Create PaTH attention layer
    try:
        config = ModelConfig()
        config.d_model = 256
        config.n_heads = 8
        config.vocab_size = 1000
        
        attention = MultiHeadAttention(
            d_model=config.d_model,
            n_heads=config.n_heads,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout
        )
        
        if torch.cuda.is_available():
            attention = attention.to(device)
        
        print(f"  ✅ PaTH attention layer created ({config.d_model}d, {config.n_heads}h)")
    except Exception as e:
        print(f"  ❌ PaTH attention creation failed: {e}")
        return False
    
    # Test 4: Forward pass
    try:
        batch_size, seq_len = 2, 64
        x = torch.randn(batch_size, seq_len, config.d_model)
        
        if torch.cuda.is_available():
            x = x.to(device)
        
        with torch.no_grad():
            output = attention(x)
        
        assert output.shape == x.shape, f"Shape mismatch: {output.shape} != {x.shape}"
        assert torch.isfinite(output).all(), "Output contains NaN or Inf"
        
        print(f"  ✅ Forward pass successful: {x.shape} -> {output.shape}")
    except Exception as e:
        print(f"  ❌ Forward pass failed: {e}")
        return False
    
    # Test 5: Check parameter count
    try:
        param_count = sum(p.numel() for p in attention.parameters())
        print(f"  ✅ PaTH attention has {param_count:,} parameters")
    except Exception as e:
        print(f"  ❌ Parameter count failed: {e}")
        return False
    
    print("\n🎉 All verification tests passed!")
    print("✅ PaTH attention is successfully integrated and working")
    print("\nYou can now run your training script:")
    print("  torchrun --nproc_per_node=2 train_distributed_path_llm.py")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n❌ Verification failed. Please check the errors above.")
        exit(1)