#!/usr/bin/env python3
"""
Quick HellaSwag Benchmark - Simplified version for fast testing
"""

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer
import json
import os
from tqdm import tqdm
import random

# Try to import from both possible training scripts
try:
    from train_distributed_llm import MinimalLLM, ModelConfig
    print("📦 Using distributed training model")
except ImportError:
    try:
        from train_llm import MinimalLLM, ModelConfig
        print("📦 Using regular training model")
    except ImportError:
        print("❌ Could not import model classes")
        exit(1)

def load_model_from_checkpoint(checkpoint_path):
    """Load model from checkpoint"""
    config_path = os.path.join(checkpoint_path, "config.json")
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    config = ModelConfig()
    for key, value in config_dict.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    model = MinimalLLM(config)
    model_path = os.path.join(checkpoint_path, "pytorch_model.bin")
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    
    return model, config

def get_completion_logprob(model, tokenizer, context, completion, device):
    """Get log probability of completion given context"""
    full_text = context + " " + completion
    
    # Tokenize
    tokens = tokenizer.encode(full_text, return_tensors='pt').to(device)
    context_tokens = tokenizer.encode(context, return_tensors='pt').to(device)
    
    if tokens.size(1) <= context_tokens.size(1):
        return float('-inf')
    
    with torch.no_grad():
        logits = model(tokens)
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Get log prob of completion tokens only
        completion_start = context_tokens.size(1) - 1
        completion_tokens = tokens[:, completion_start + 1:]
        completion_logits = log_probs[0, completion_start:-1]
        
        if completion_tokens.size(1) == 0:
            return float('-inf')
        
        # Calculate average log probability
        token_log_probs = completion_logits.gather(1, completion_tokens.T).T
        avg_log_prob = token_log_probs.mean().item()
    
    return avg_log_prob

def evaluate_hellaswag_sample(model, tokenizer, device, num_samples=100):
    """Quick evaluation on a sample of HellaSwag"""
    print(f"🔍 Loading HellaSwag validation set...")
    
    try:
        dataset = load_dataset("hellaswag", split="validation")
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return None
    
    # Sample random examples
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    sample_data = [dataset[i] for i in indices]
    
    print(f"📊 Evaluating {len(sample_data)} examples...")
    
    correct = 0
    total = 0
    
    for example in tqdm(sample_data, desc="Evaluating"):
        context = example['ctx']
        endings = example['endings']
        correct_idx = int(example['label'])
        
        # Get log probabilities for each ending
        log_probs = []
        for ending in endings:
            log_prob = get_completion_logprob(model, tokenizer, context, ending, device)
            log_probs.append(log_prob)
        
        # Predict highest probability ending
        predicted_idx = max(range(len(log_probs)), key=lambda i: log_probs[i])
        
        if predicted_idx == correct_idx:
            correct += 1
        total += 1
    
    accuracy = correct / total if total > 0 else 0
    
    print(f"\n🎯 Quick HellaSwag Results:")
    print(f"📊 Accuracy: {accuracy:.3f} ({correct}/{total})")
    print(f"🎲 Random baseline: 0.250")
    
    if accuracy > 0.25:
        improvement = (accuracy - 0.25) / 0.25 * 100
        print(f"📈 Improvement over random: +{improvement:.1f}%")
    else:
        print(f"📉 Below random baseline")
    
    return accuracy

def main():
    import glob
    
    # Find checkpoints (handle both naming patterns)
    checkpoints = glob.glob("checkpoints/checkpoint-*") + glob.glob("checkpoints/checkpoint_step_*")
    if not checkpoints:
        print("❌ No checkpoints found in 'checkpoints/' directory")
        return
    
    # Sort by step number (handle both patterns)
    def extract_step(path):
        basename = os.path.basename(path)
        if 'checkpoint-' in basename:
            return int(basename.split('-')[-1])
        elif 'checkpoint_step_' in basename:
            return int(basename.split('_')[-1])
        return 0
    
    checkpoints.sort(key=extract_step)
    
    print("📁 Available checkpoints:")
    for i, checkpoint in enumerate(checkpoints):
        basename = os.path.basename(checkpoint)
        if 'checkpoint-' in basename:
            step = basename.split('-')[-1]
        elif 'checkpoint_step_' in basename:
            step = basename.split('_')[-1]
        else:
            step = "unknown"
        print(f"  {i+1}. Step {step} ({checkpoint})")
    
    # Let user choose
    while True:
        try:
            choice = int(input(f"\nChoose checkpoint (1-{len(checkpoints)}): ")) - 1
            if 0 <= choice < len(checkpoints):
                selected_checkpoint = checkpoints[choice]
                break
            else:
                print("Invalid choice, try again.")
        except (ValueError, KeyboardInterrupt):
            print("\nExiting...")
            return
    
    print(f"🚀 Loading checkpoint: {selected_checkpoint}")
    
    # Load model
    model, config = load_model_from_checkpoint(selected_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    print(f"✅ Model loaded on {device}")
    print(f"📊 Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Run quick evaluation
    num_samples = int(input("Number of samples to evaluate (default 100): ") or "100")
    evaluate_hellaswag_sample(model, tokenizer, device, num_samples)

if __name__ == "__main__":
    main()