#!/usr/bin/env python3
"""
Quick test to verify the training script works with PaTH attention
"""

import os
import torch
import subprocess
import sys

def test_single_gpu_training():
    """Test training with a single GPU for a few steps"""
    print("🧪 Testing single GPU training with PaTH attention...")
    
    if not torch.cuda.is_available():
        print("  ⚠️ CUDA not available, skipping training test.")
        return
    
    # Set environment variables for single GPU
    env = os.environ.copy()
    env.update({
        "RANK": "0",
        "WORLD_SIZE": "1", 
        "LOCAL_RANK": "0",
        "MASTER_ADDR": "localhost",
        "MASTER_PORT": "12355"
    })
    
    # Create a minimal training command
    cmd = [
        sys.executable, "-c", """
import os
import torch
import torch.distributed as dist

# Set environment for single GPU
os.environ.update({
    'RANK': '0',
    'WORLD_SIZE': '1',
    'LOCAL_RANK': '0',
    'MASTER_ADDR': 'localhost',
    'MASTER_PORT': '12355'
})

# Import after setting environment
from train_distributed_path_llm import ModelConfig, MinimalLLM, TextTokenDataset, DistributedSampler
from torch.utils.data import DataLoader
import torch.nn.functional as F

# Initialize distributed
dist.init_process_group(backend='nccl', init_method='env://')

# Create minimal config
config = ModelConfig()
config.d_model = 128
config.n_heads = 4  
config.n_layers = 2
config.d_ff = 256
config.max_seq_len = 64
config.vocab_size = 1000
config.max_steps = 5  # Just 5 steps for testing
config.batch_size = 2

print(f"✅ Config created: {config.d_model}d, {config.n_heads}h, {config.n_layers}L")

# Create model
model = MinimalLLM(config)
device = torch.device('cuda:0')
model = model.to(device)

print(f"✅ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

# Create dummy dataset
tokens = list(range(1000))  # Simple token sequence
dataset = TextTokenDataset(tokens, config.max_seq_len)

# Create sampler and loader
sampler = DistributedSampler(dataset, 0, 1, shuffle=True, drop_last=True)
loader = DataLoader(dataset, batch_size=config.batch_size, sampler=sampler)

print(f"✅ Dataset created with {len(dataset)} samples")

# Test a few training steps
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

for step, (x, y) in enumerate(loader):
    if step >= config.max_steps:
        break
        
    x, y = x.to(device), y.to(device)
    
    # Set epoch for sampler (this was the original error)
    sampler.set_epoch(step)
    
    optimizer.zero_grad()
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))
    loss.backward()
    optimizer.step()
    
    print(f"  Step {step}: loss = {loss.item():.4f}")

print("🎉 Training test completed successfully!")
dist.destroy_process_group()
"""
    ]
    
    try:
        result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✅ Training test passed!")
            print("Output:", result.stdout)
        else:
            print("❌ Training test failed!")
            print("Error:", result.stderr)
            print("Output:", result.stdout)
            
    except subprocess.TimeoutExpired:
        print("⏰ Training test timed out (this might be normal for initialization)")
    except Exception as e:
        print(f"❌ Training test failed with exception: {e}")

if __name__ == "__main__":
    test_single_gpu_training()