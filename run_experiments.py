#!/usr/bin/env python3
"""
Main script to run research experiments
"""
import argparse
import sys
import torch
from typing import List

from experiments import (
    ExperimentRunner,
    create_baseline_experiments,
    create_attention_experiments,
    create_architecture_experiments,
    create_training_experiments
)
from experiments.experiment_definitions import create_efficiency_experiments
from configs.base_config import ExperimentConfig

def main():
    parser = argparse.ArgumentParser(description="Run LLM Research Experiments")
    parser.add_argument("--experiment-set", type=str, default="baseline",
                       choices=["baseline", "attention", "architecture", "training", "efficiency", "all"],
                       help="Which set of experiments to run")
    parser.add_argument("--results-dir", type=str, default="results",
                       help="Directory to save results")
    parser.add_argument("--distributed", action="store_true",
                       help="Use distributed training")
    parser.add_argument("--single-gpu", action="store_true",
                       help="Force single GPU training")
    
    args = parser.parse_args()
    
    print("🔬 LLM RESEARCH EXPERIMENT RUNNER")
    print("=" * 60)
    print("Based on Flash Linear Attention architectures")
    print()
    
    # Check GPU availability - REQUIRED for FLA
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"🔍 Available GPUs: {gpu_count}")
        for i in range(min(gpu_count, 4)):  # Show first 4 GPUs
            gpu_name = torch.cuda.get_device_name(i)
            memory_gb = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"  GPU {i}: {gpu_name} ({memory_gb:.1f} GB)")
    else:
        print("❌ No CUDA GPUs available!")
        print("💡 Flash Linear Attention requires CUDA for Triton kernels")
        print("   Only standard attention will work on CPU")
        
        # Check if any experiments use FLA
        from experiments.experiment_definitions import create_attention_experiments
        attention_experiments = create_attention_experiments()
        fla_experiments = [exp for exp in attention_experiments if exp.attention_config.attention_type != "standard"]
        
        if fla_experiments:
            print(f"⚠️ {len(fla_experiments)} experiments require CUDA but none available")
            confirm = input("Continue anyway? (y/N): ").strip().lower()
            if confirm != 'y':
                print("Aborted. Please run on a machine with CUDA GPUs.")
                return
    
    # Determine distributed training
    use_distributed = False
    if args.single_gpu:
        use_distributed = False
        print("🔧 Forced single GPU mode")
    elif args.distributed:
        use_distributed = True
        print("🔧 Forced distributed mode")
    else:
        # Auto-detect
        use_distributed = torch.cuda.device_count() > 1
        mode = "distributed" if use_distributed else "single GPU"
        print(f"🔧 Auto-detected: {mode} mode")
    
    # Create experiment runner
    runner = ExperimentRunner(
        results_dir=args.results_dir,
        use_distributed=use_distributed
    )
    
    # Select experiments
    experiments = []
    
    if args.experiment_set == "baseline":
        experiments = create_baseline_experiments()
        print(f"📋 Running baseline experiments ({len(experiments)} experiments)")
    elif args.experiment_set == "attention":
        experiments = create_attention_experiments()
        print(f"📋 Running attention mechanism experiments ({len(experiments)} experiments)")
    elif args.experiment_set == "architecture":
        experiments = create_architecture_experiments()
        print(f"📋 Running architecture experiments ({len(experiments)} experiments)")
    elif args.experiment_set == "training":
        experiments = create_training_experiments()
        print(f"📋 Running training configuration experiments ({len(experiments)} experiments)")
    elif args.experiment_set == "efficiency":
        experiments = create_efficiency_experiments()
        print(f"📋 Running efficiency comparison experiments ({len(experiments)} experiments)")
    elif args.experiment_set == "all":
        experiments = (
            create_baseline_experiments() +
            create_attention_experiments() +
            create_architecture_experiments() +
            create_training_experiments() +
            create_efficiency_experiments()
        )
        print(f"📋 Running ALL experiments ({len(experiments)} experiments)")
    
    if not experiments:
        print("❌ No experiments selected!")
        return
    
    # Print experiment overview
    print(f"\n📊 EXPERIMENT OVERVIEW:")
    for i, exp in enumerate(experiments):
        attention_type = exp.attention_config.attention_type
        print(f"  {i+1:2d}. {exp.name:<25} ({attention_type:<10}) - {exp.description}")
    
    # Confirm before starting
    print(f"\n🚀 Ready to run {len(experiments)} experiments")
    if not args.single_gpu and torch.cuda.device_count() > 1:
        print(f"⚡ Using {torch.cuda.device_count()} GPUs with distributed training")
    
    confirm = input("Continue? (y/N): ").strip().lower()
    if confirm != 'y':
        print("Aborted.")
        return
    
    try:
        # Run experiments
        results = runner.run_experiment_suite(experiments)
        
        print(f"\n✅ EXPERIMENT SUITE COMPLETED!")
        print(f"📁 Results saved in: {args.results_dir}/")
        print(f"📊 {len([r for r in results if r.get('success', False)])} successful experiments")
        
    except KeyboardInterrupt:
        print(f"\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"\n❌ Experiment suite failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        runner.cleanup()

if __name__ == "__main__":
    main()