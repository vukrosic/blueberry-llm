#!/usr/bin/env python3
"""
Quick test script to compare specific attention mechanisms
"""
import torch
import argparse
from configs.base_config import ExperimentConfig, AttentionConfig
from experiments import ExperimentRunner

def create_quick_test_experiments(attention_types: list, max_steps: int = 1000) -> list:
    """Create quick test experiments for specific attention types"""
    experiments = []
    
    # Base configuration for quick testing
    base_config = {
        "d_model": 384,
        "n_layers": 4,  # Smaller for quick testing
        "n_heads": 8,
        "d_ff": 1536,
        "max_steps": max_steps,
        "batch_size": 8,  # Smaller batch for quick testing
        "eval_every": 200,  # More frequent evaluation
        "save_every": 1000,  # Don't save checkpoints for quick tests
    }
    
    for attention_type in attention_types:
        config = ExperimentConfig(
            name=f"quick_test_{attention_type}",
            description=f"Quick test of {attention_type} attention",
            hypothesis=f"Testing {attention_type} attention mechanism",
            attention_config=AttentionConfig(attention_type=attention_type),
            **base_config
        )
        experiments.append(config)
    
    return experiments

def main():
    parser = argparse.ArgumentParser(description="Quick attention mechanism test")
    parser.add_argument("--attention_types", type=str, required=True,
                       help="Comma-separated list of attention types (e.g., 'gla,retnet,based')")
    parser.add_argument("--max_steps", type=int, default=1000,
                       help="Maximum training steps for quick test")
    parser.add_argument("--results_dir", type=str, default="quick_test_results",
                       help="Directory to save results")
    
    args = parser.parse_args()
    
    # Parse attention types
    attention_types = [t.strip() for t in args.attention_types.split(',')]
    
    print("🚀 QUICK ATTENTION MECHANISM TEST")
    print("=" * 50)
    print(f"Testing attention types: {', '.join(attention_types)}")
    print(f"Max steps per experiment: {args.max_steps}")
    print()
    
    # Check GPU
    if not torch.cuda.is_available():
        print("❌ CUDA required for FLA testing")
        return
    
    print(f"🔍 Using GPU: {torch.cuda.get_device_name(0)}")
    
    # Create experiments
    experiments = create_quick_test_experiments(attention_types, args.max_steps)
    
    print(f"\n📋 EXPERIMENTS TO RUN:")
    for i, exp in enumerate(experiments):
        print(f"  {i+1}. {exp.name} - {exp.description}")
    
    # Create runner
    runner = ExperimentRunner(
        results_dir=args.results_dir,
        use_distributed=False  # Single GPU for quick tests
    )
    
    try:
        print(f"\n🏃 Starting experiments...")
        results = runner.run_experiment_suite(experiments)
        
        print(f"\n✅ QUICK TEST COMPLETED!")
        print(f"📁 Results saved in: {args.results_dir}/")
        
        # Print quick summary
        print(f"\n📊 QUICK RESULTS SUMMARY:")
        for result in results:
            if result.get('success', False):
                final_loss = result.get('final_loss', 'N/A')
                tokens_per_sec = result.get('tokens_per_sec', 'N/A')
                print(f"  ✅ {result['experiment_name']}: Loss={final_loss}, Speed={tokens_per_sec} tok/s")
            else:
                print(f"  ❌ {result['experiment_name']}: Failed")
                
    except KeyboardInterrupt:
        print(f"\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()