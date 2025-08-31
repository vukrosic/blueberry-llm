#!/usr/bin/env python3
"""
Test the training script with PaTH attention integration
"""

import torch
import torch.nn as nn
import sys
import os

# Set environment variables for single GPU testing
os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1" 
os.environ["LOCAL_RANK"] = "0"
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12355"

def test_model_creation():
    """Test that we can create the model with PaTH attention"""
    print("🧪 Testing model creation with PaTH attention...")
    
    if not torch.cuda.is_available():
        print("  ⚠️ CUDA not available, skipping test.")
        return
    
    # Import after setting environment variables
    from train_distributed_path_llm import ModelConfig, MinimalLLM
    
    # Create a minimal config for testing
    config = ModelConfig()
    config.d_model = 128  # Small for testing
    config.n_heads = 4
    config.n_layers = 2   # Just 2 layers for testing
    config.d_ff = 256     # Small feed-forward
    config.max_seq_len = 64  # Short sequences
    config.vocab_size = 1000  # Small vocab
    
    print(f"  Config: {config.d_model}d, {config.n_heads}h, {config.n_layers}L")
    
    # Create model
    model = MinimalLLM(config)
    device = torch.device('cuda:0')
    model = model.to(device)
    
    print(f"  Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Test forward pass
    batch_size = 2
    seq_len = 32
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len)).to(device)
    
    print(f"  Input shape: {input_ids.shape}")
    
    with torch.no_grad():
        logits = model(input_ids)
        print(f"  Output shape: {logits.shape}")
        
        # Verify output shape
        expected_shape = (batch_size, seq_len, config.vocab_size)
        assert logits.shape == expected_shape, f"Shape mismatch: {logits.shape} != {expected_shape}"
        
        # Verify output is finite
        assert torch.isfinite(logits).all(), "Output contains NaN or Inf values"
    
    print("  ✅ Forward pass successful!")
    
    # Test backward pass
    input_ids.requires_grad_(False)  # Input IDs don't need gradients
    logits = model(input_ids)
    loss = logits.sum()
    loss.backward()
    
    # Check that some parameters have gradients
    grad_params = [p for p in model.parameters() if p.grad is not None]
    print(f"  Parameters with gradients: {len(grad_params)}/{len(list(model.parameters()))}")
    
    assert len(grad_params) > 0, "No parameters received gradients"
    
    # Check gradients are finite
    for p in grad_params:
        assert torch.isfinite(p.grad).all(), "Gradients contain NaN or Inf"
    
    print("  ✅ Backward pass successful!")
    print("🎉 Model integration test passed!")

if __name__ == "__main__":
    test_model_creation()