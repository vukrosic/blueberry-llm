#!/usr/bin/env python3
"""
HellaSwag Benchmark for Distributed LLM
Evaluates common sense reasoning capabilities of your trained model
"""

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer
import json
import os
import glob
import re
from tqdm import tqdm
import argparse
from typing import Dict, List, Tuple
import numpy as np

# Import your model classes
from train_distributed_llm import MinimalLLM, ModelConfig

class HellaSwagEvaluator:
    def __init__(self, model_path: str, device: str = 'auto'):
        """Initialize the HellaSwag evaluator"""
        self.device = self._setup_device(device)
        self.model, self.config = self._load_model(model_path)
        self.tokenizer = self._load_tokenizer()
        
        print(f"✅ Model loaded on {self.device}")
        print(f"📊 Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def _setup_device(self, device: str) -> torch.device:
        """Setup compute device"""
        if device == 'auto':
            if torch.cuda.is_available():
                device = 'cuda'
            else:
                device = 'cpu'
        return torch.device(device)
    
    def _load_model(self, checkpoint_path: str) -> Tuple[MinimalLLM, ModelConfig]:
        """Load model from checkpoint"""
        # Load config
        config_path = os.path.join(checkpoint_path, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        # Create config object
        config = ModelConfig()
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        # Load model
        model = MinimalLLM(config)
        model_path = os.path.join(checkpoint_path, "pytorch_model.bin")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model = model.to(self.device)
        model.eval()
        
        return model, config
    
    def _load_tokenizer(self):
        """Load tokenizer"""
        tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    
    def _get_logprobs(self, text: str) -> float:
        """Get log probability of a text sequence"""
        # Tokenize
        tokens = self.tokenizer.encode(text, return_tensors='pt').to(self.device)
        
        if tokens.size(1) == 0:
            return float('-inf')
        
        with torch.no_grad():
            # Get logits
            logits = self.model(tokens)
            
            # Calculate log probabilities
            log_probs = F.log_softmax(logits, dim=-1)
            
            # Get log prob of actual next tokens
            target_tokens = tokens[:, 1:]  # Shift by 1 for next token prediction
            input_tokens = tokens[:, :-1]  # Remove last token from input
            
            if target_tokens.size(1) == 0:
                return float('-inf')
            
            # Gather log probs for target tokens
            gathered_log_probs = log_probs[0, :-1].gather(1, target_tokens.T).T
            
            # Average log probability per token
            avg_log_prob = gathered_log_probs.mean().item()
            
        return avg_log_prob
    
    def _evaluate_example(self, example: Dict) -> Dict:
        """Evaluate a single HellaSwag example"""
        context = example['ctx']
        endings = example['endings']
        correct_idx = int(example['label'])
        
        # Calculate log probabilities for each ending
        log_probs = []
        for ending in endings:
            full_text = context + " " + ending
            log_prob = self._get_logprobs(full_text)
            log_probs.append(log_prob)
        
        # Predict the ending with highest log probability
        predicted_idx = np.argmax(log_probs)
        is_correct = predicted_idx == correct_idx
        
        return {
            'correct': is_correct,
            'predicted_idx': predicted_idx,
            'correct_idx': correct_idx,
            'log_probs': log_probs,
            'context': context,
            'endings': endings
        }
    
    def evaluate(self, split: str = 'validation', max_examples: int = None, 
                 save_results: bool = True) -> Dict:
        """Evaluate on HellaSwag dataset"""
        print(f"🔍 Loading HellaSwag {split} dataset...")
        
        # Load dataset
        try:
            dataset = load_dataset("hellaswag", split=split)
        except Exception as e:
            print(f"❌ Failed to load HellaSwag dataset: {e}")
            print("💡 Make sure you have internet connection and datasets library installed")
            return {}
        
        if max_examples:
            dataset = dataset.select(range(min(max_examples, len(dataset))))
        
        print(f"📊 Evaluating {len(dataset)} examples...")
        
        results = []
        correct_count = 0
        
        # Evaluate each example
        for i, example in enumerate(tqdm(dataset, desc="Evaluating")):
            result = self._evaluate_example(example)
            results.append(result)
            
            if result['correct']:
                correct_count += 1
            
            # Print progress every 100 examples
            if (i + 1) % 100 == 0:
                current_accuracy = correct_count / (i + 1)
                print(f"Progress: {i+1}/{len(dataset)}, Accuracy so far: {current_accuracy:.3f}")
        
        # Calculate final metrics
        total_examples = len(results)
        accuracy = correct_count / total_examples if total_examples > 0 else 0.0
        
        # Calculate per-position accuracy (which choice position is most often correct)
        position_stats = {i: {'correct': 0, 'predicted': 0} for i in range(4)}
        for result in results:
            position_stats[result['correct_idx']]['correct'] += 1
            position_stats[result['predicted_idx']]['predicted'] += 1
        
        evaluation_results = {
            'accuracy': accuracy,
            'total_examples': total_examples,
            'correct_count': correct_count,
            'position_stats': position_stats,
            'examples': results[:10] if len(results) > 10 else results  # Save first 10 for inspection
        }
        
        # Save results
        if save_results:
            results_file = f"hellaswag_results_{split}.json"
            with open(results_file, 'w') as f:
                json.dump(evaluation_results, f, indent=2)
            print(f"💾 Results saved to {results_file}")
        
        return evaluation_results
    
    def print_results(self, results: Dict):
        """Print evaluation results in a nice format"""
        if not results:
            return
        
        print("\n" + "="*60)
        print("🎯 HELLASWAG BENCHMARK RESULTS")
        print("="*60)
        print(f"📊 Overall Accuracy: {results['accuracy']:.3f} ({results['correct_count']}/{results['total_examples']})")
        print(f"🎲 Random Baseline: 0.250 (25%)")
        
        if results['accuracy'] > 0.25:
            improvement = (results['accuracy'] - 0.25) / 0.25 * 100
            print(f"📈 Improvement over random: +{improvement:.1f}%")
        else:
            print("📉 Performance below random baseline")
        
        print("\n📍 Position Statistics:")
        for pos, stats in results['position_stats'].items():
            correct_pct = stats['correct'] / results['total_examples'] * 100
            predicted_pct = stats['predicted'] / results['total_examples'] * 100
            print(f"  Position {pos}: {correct_pct:.1f}% correct, {predicted_pct:.1f}% predicted")
        
        print("\n🔍 Sample Examples:")
        for i, example in enumerate(results['examples'][:3]):
            print(f"\nExample {i+1}:")
            print(f"Context: {example['context'][:100]}...")
            print(f"Correct ending ({example['correct_idx']}): {example['endings'][example['correct_idx']]}")
            print(f"Predicted ending ({example['predicted_idx']}): {example['endings'][example['predicted_idx']]}")
            print(f"Result: {'✅ Correct' if example['correct'] else '❌ Wrong'}")
        
        print("="*60)

def find_checkpoints(checkpoint_dir: str = "checkpoints") -> List[str]:
    """Find available model checkpoints"""
    if not os.path.exists(checkpoint_dir):
        return []
    
    # Handle both naming patterns
    checkpoints = (glob.glob(os.path.join(checkpoint_dir, "checkpoint-*")) + 
                  glob.glob(os.path.join(checkpoint_dir, "checkpoint_step_*")))
    
    # Sort by step number (handle both patterns)
    def extract_step(path):
        basename = os.path.basename(path)
        if 'checkpoint-' in basename:
            step_part = basename.split('-')[-1]
            return int(step_part) if step_part.isdigit() else 0
        elif 'checkpoint_step_' in basename:
            return int(basename.split('_')[-1])
        return 0
    
    checkpoints.sort(key=extract_step)
    return checkpoints

def main():
    parser = argparse.ArgumentParser(description="HellaSwag Benchmark for Distributed LLM")
    parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument("--split", type=str, default="validation", 
                       choices=["validation", "train"], help="Dataset split to evaluate")
    parser.add_argument("--max-examples", type=int, help="Maximum number of examples to evaluate")
    parser.add_argument("--device", type=str, default="auto", help="Device to use (auto, cuda, cpu)")
    parser.add_argument("--no-save", action="store_true", help="Don't save results to file")
    
    args = parser.parse_args()
    
    # Find checkpoint if not specified
    if not args.checkpoint:
        checkpoints = find_checkpoints()
        if not checkpoints:
            print("❌ No checkpoints found in 'checkpoints/' directory")
            print("💡 Train a model first or specify --checkpoint path")
            return
        
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
        
        while True:
            try:
                choice = int(input(f"\nChoose checkpoint (1-{len(checkpoints)}): ")) - 1
                if 0 <= choice < len(checkpoints):
                    args.checkpoint = checkpoints[choice]
                    break
                else:
                    print("Invalid choice, try again.")
            except (ValueError, KeyboardInterrupt):
                print("\nExiting...")
                return
    
    print(f"🚀 Starting HellaSwag evaluation...")
    print(f"📂 Checkpoint: {args.checkpoint}")
    print(f"📊 Split: {args.split}")
    if args.max_examples:
        print(f"🔢 Max examples: {args.max_examples}")
    
    try:
        # Initialize evaluator
        evaluator = HellaSwagEvaluator(args.checkpoint, device=args.device)
        
        # Run evaluation
        results = evaluator.evaluate(
            split=args.split,
            max_examples=args.max_examples,
            save_results=not args.no_save
        )
        
        # Print results
        evaluator.print_results(results)
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()