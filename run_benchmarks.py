#!/usr/bin/env python3
"""
Main script to run benchmarks on trained models
"""
import argparse
from utils.benchmarking import BenchmarkRunner

def main():
    parser = argparse.ArgumentParser(description="Benchmark Trained LLM Models")
    parser.add_argument("--results-dir", type=str, default="results",
                       help="Directory containing experiment results")
    parser.add_argument("--experiment", type=str, default=None,
                       help="Specific experiment to benchmark (default: all)")
    parser.add_argument("--max-examples", type=int, default=500,
                       help="Maximum examples per benchmark")
    
    args = parser.parse_args()
    
    print("📊 LLM MODEL BENCHMARK RUNNER")
    print("=" * 60)
    
    # Create benchmark runner
    runner = BenchmarkRunner(results_dir=args.results_dir)
    
    # Find available experiments
    checkpoints = runner.find_experiment_checkpoints()
    
    if not checkpoints:
        print(f"❌ No experiment checkpoints found in {args.results_dir}")
        print("💡 Make sure you've run experiments first using run_experiments.py")
        return
    
    print(f"📂 Found {len(checkpoints)} experiments with checkpoints:")
    for i, exp_name in enumerate(checkpoints.keys()):
        print(f"  {i+1}. {exp_name}")
    
    # Run benchmarks
    if args.experiment:
        # Benchmark specific experiment
        if args.experiment in checkpoints:
            checkpoint_dir = checkpoints[args.experiment]
            result = runner.benchmark_experiment(args.experiment, checkpoint_dir)
            
            # Save result
            import json
            import os
            result_file = os.path.join(args.results_dir, f"{args.experiment}_benchmark_results.json")
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"💾 Results saved to: {result_file}")
        else:
            print(f"❌ Experiment '{args.experiment}' not found!")
            print(f"Available experiments: {', '.join(checkpoints.keys())}")
    else:
        # Benchmark all experiments
        print(f"\n🚀 Benchmarking all {len(checkpoints)} experiments...")
        confirm = input("Continue? (y/N): ").strip().lower()
        if confirm != 'y':
            print("Aborted.")
            return
        
        results = runner.benchmark_all_experiments()
        print(f"\n✅ Benchmarking completed!")
        print(f"📁 Results saved in: {args.results_dir}/")

if __name__ == "__main__":
    main()