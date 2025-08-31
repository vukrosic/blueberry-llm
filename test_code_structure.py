#!/usr/bin/env python3
"""
Test that the code structure is correct and imports work
"""

import torch
import sys
import os

def test_imports():
    """Test that all imports work correctly"""
    print("🧪 Testing imports...")
    
    try:
        from fla.layers.path_attn import PaTHAttention
        print("  ✅ PaTHAttention import successful")
    except ImportError as e:
        print(f"  ❌ PaTHAttention import failed: {e}")
        return False
    
    try:
        # Set minimal environment to avoid distributed errors
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")
        
        from train_distributed_path_llm import (
            ModelConfig, MinimalLLM, MultiHeadAttention, 
            TransformerBlock, TextTokenDataset, DistributedSampler
        )
        print("  ✅ Training script imports successful")
    except ImportError as e:
        print(f"  ❌ Training script import failed: {e}")
        return False
    
    return True

def test_model_creation():
    """Test that we can create the model structure"""
    print("🧪 Testing model creation...")
    
    try:
        from train_distributed_path_llm import ModelConfig, MinimalLLM, MultiHeadAttention
        
        # Create minimal config
        config = ModelConfig()
        config.d_model = 64
        config.n_heads = 4
        config.n_layers = 2
        config.d_ff = 128
        config.vocab_size = 100
        
        # Test attention layer
        attention = MultiHeadAttention(
            d_model=config.d_model,
            n_heads=config.n_heads,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout
        )
        print(f"  ✅ PaTH attention layer created")
        
        # Test full model
        model = MinimalLLM(config)
        param_count = sum(p.numel() for p in model.parameters())
        print(f"  ✅ Full model created with {param_count:,} parameters")
        
        # Test forward pass with CPU tensors
        batch_size, seq_len = 2, 16
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        
        with torch.no_grad():
            logits = model(input_ids)
            expected_shape = (batch_size, seq_len, config.vocab_size)
            assert logits.shape == expected_shape, f"Shape mismatch: {logits.shape} != {expected_shape}"
            print(f"  ✅ Forward pass successful: {input_ids.shape} -> {logits.shape}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_sampler_structure():
    """Test that the sampler and dataset work correctly"""
    print("🧪 Testing data structures...")
    
    try:
        from train_distributed_path_llm import TextTokenDataset, DistributedSampler
        
        # Create dummy data
        tokens = list(range(100))
        dataset = TextTokenDataset(tokens, seq_len=16)
        
        # Create sampler
        sampler = DistributedSampler(dataset, rank=0, world_size=1, shuffle=True)
        
        # Test sampler methods
        sampler.set_epoch(0)  # This was the original error
        print(f"  ✅ Sampler created with {len(sampler)} samples")
        
        # Test dataset
        x, y = dataset[0]
        print(f"  ✅ Dataset working: sample shape {x.shape}, {y.shape}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Data structure test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🔍 Running code structure tests...\n")
    
    tests = [
        ("Imports", test_imports),
        ("Model Creation", test_model_creation), 
        ("Data Structures", test_sampler_structure)
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\n--- {name} ---")
        success = test_func()
        results.append((name, success))
    
    print(f"\n{'='*50}")
    print("📊 Test Results:")
    all_passed = True
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {name}: {status}")
        if not success:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All tests passed! The PaTH attention integration looks good.")
    else:
        print("\n⚠️ Some tests failed. Please check the errors above.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)