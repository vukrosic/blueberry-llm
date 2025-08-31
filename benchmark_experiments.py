#!/usr/bin/env python3
"""
Benchmark Runner for Research Experiments
Evaluates trained models from research_experiments.py using existing benchmark tools
"""

import os
import json
import glob
from typing import Dict, List, Any
from tiny_llm_benchmarks import TinyLLMBenchmark
from hellaswag_benchmark import HellaSwagBenchmark

def find_experiment_checkpoints(results_dir: str = "research_results") -> Dict[str, str]:
    """Find all experiment checkpoints"""
    checkpoints = {}
    
    if not os.path.exists(results_dir):
        print(f"❌ Results directory not found: {results_dir}")
        return checkpoints
    
    # Look for experiment directories
    for exp_dir in glob.glob(f"{results_dir}/*/"):
        exp_name = os.path.basename(exp_dir.rstrip('/'))
        
        # Look for checkpoints in this experiment
        checkpoint_dir = os.path.join(exp_dir, "checkpoints")
        if os.path.exists(checkpoint_dir):
            # Find the latest checkpoint
            checkpoint_files = glob.glob(f"{checkpoint_dir}/**/model.pt", recursive=True)
            if checkpoint_files:
                # Get the most recent checkpoint
                latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
                checkpoints[exp_name] = latest_checkpoint
                print(f"📂 Found checkpoint for {exp_name}: {latest_checkpoint}")
    
    return checkpoints

def run_benchmarks_for_experiment(exp_name: str, checkpoint_path: str) -> Dict[str, Any]:
    """Run all benchmarks for a single experiment"""
    print(f"\n🧪 BENCHMARKING: {exp_name}")
    print("=" * 50)
    
    results = {
        'experiment_name': exp_name,
        'checkpoint_path': checkpoint_path,
        'benchmarks': {}
    }
    
    try:
        # Run comprehensive benchmarks
        print("📊 Running comprehensive benchmark suite...")
        benchmark = TinyLLMBenchmark(checkpoint_path)
        comprehensive_results = benchmark.run_all_benchmarks(save_results=False)
        results['benchmarks']['comprehensive'] = comprehensive_results
        
        # Print summary
        if 'overall' in comprehensive_results:
            overall = comprehensive_results['overall']
            print(f"📈 Overall accuracy: {overall.get('accuracy', 'N/A'):.3f}")
        
        # Show individual task results
        for task_name, task_result in comprehensive_results.items():
            if task_name != 'overall' and isinstance(task_result, dict):
                if 'accuracy' in task_result:
                    print(f"  {task_name}: {task_result['accuracy']:.3f}")
                elif 'exact_match_score' in task_result:
                    print(f"  {task_name}: EM={task_result['exact_match_score']:.3f}")
        
    except Exception as e:
        print(f"❌ Comprehensive benchmarks failed: {e}")
        results['benchmarks']['comprehensive'] = {'error': str(e)}
    
    try:
        # Run HellaSwag specifically
        print("\n🎯 Running HellaSwag benchmark...")
        hellaswag = HellaSwagBenchmark(checkpoint_path)
        hellaswag_results = hellaswag.evaluate(max_examples=200)  # Quick evaluation
        results['benchmarks']['hellaswag'] = hellaswag_results
        
        if 'accuracy' in hellaswag_results:
            print(f"🎯 HellaSwag accuracy: {hellaswag_results['accuracy']:.3f}")
        
    except Exception as e:
        print(f"❌ HellaSwag benchmark failed: {e}")
        results['benchmarks']['hellaswag'] = {'error': str(e)}
    
    return results

def compare_experiments(all_results: List[Dict[str, Any]]) -> None:
    """Compare results across experiments"""
    print(f"\n📊 EXPERIMENT COMPARISON")
    print("=" * 60)
    
    # Extract key metrics for comparison
    comparison_data = []
    
    for result in all_results:
        exp_name = result['experiment_name']
        benchmarks = result['benchmarks']
        
        row = {'experiment': exp_name}
        
        # Get comprehensive benchmark overall accuracy
        if 'comprehensive' in benchmarks and 'overall' in benchmarks['comprehensive']:
            overall = benchmarks['comprehensive']['overall']
            row['overall_accuracy'] = overall.get('accuracy', 0)
        else:
            row['overall_accuracy'] = 0
        
        # Get HellaSwag accuracy
        if 'hellaswag' in benchmarks and 'accuracy' in benchmarks['hellaswag']:
            row['hellaswag_accuracy'] = benchmarks['hellaswag']['accuracy']
        else:
            row['hellaswag_accuracy'] = 0
        
        # Get specific task accuracies
        if 'comprehensive' in benchmarks:
            comp = benchmarks['comprehensive']
            row['lambada'] = comp.get('lambada', {}).get('accuracy', 0)
            row['arithmetic'] = comp.get('arithmetic', {}).get('accuracy', 0)
            row['piqa'] = comp.get('piqa', {}).get('accuracy', 0)
        
        comparison_data.append(row)
    
    # Sort by overall accuracy
    comparison_data.sort(key=lambda x: x['overall_accuracy'], reverse=True)
    
    # Print comparison table
    print(f"{'Experiment':<20} {'Overall':<8} {'HellaSwag':<10} {'LAMBADA':<8} {'Arith':<6} {'PIQA':<6}")
    print("-" * 70)
    
    for row in comparison_data:
        print(f"{row['experiment']:<20} "
              f"{row['overall_accuracy']:<8.3f} "
              f"{row['hellaswag_accuracy']:<10.3f} "
              f"{row.get('lambada', 0):<8.3f} "
              f"{row.get('arithmetic', 0):<6.3f} "
              f"{row.get('piqa', 0):<6.3f}")
    
    # Find best performing experiments
    if comparison_data:
        best_overall = comparison_data[0]
        best_hellaswag = max(comparison_data, key=lambda x: x['hellaswag_accuracy'])
        
        print(f"\n🏆 BEST PERFORMERS:")
        print(f"  Overall: {best_overall['experiment']} ({best_overall['overall_accuracy']:.3f})")
        print(f"  HellaSwag: {best_hellaswag['experiment']} ({best_hellaswag['hellaswag_accuracy']:.3f})")

def main():
    """Main benchmark runner"""
    print("📊 RESEARCH EXPERIMENT BENCHMARK RUNNER")
    print("=" * 60)
    
    # Find all experiment checkpoints
    checkpoints = find_experiment_checkpoints()
    
    if not checkpoints:
        print("❌ No experiment checkpoints found!")
        print("💡 Make sure you've run research_experiments.py first")
        return
    
    print(f"\n📂 Found {len(checkpoints)} experiments to benchmark")
    
    # Ask user which experiments to benchmark
    print("\nAvailable experiments:")
    exp_names = list(checkpoints.keys())
    for i, name in enumerate(exp_names):
        print(f"  {i+1}. {name}")
    
    print(f"  {len(exp_names)+1}. All experiments")
    
    choice = input(f"\nSelect experiments to benchmark (1-{len(exp_names)+1}): ").strip()
    
    selected_experiments = []
    if choice == str(len(exp_names)+1):
        selected_experiments = exp_names
    else:
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(exp_names):
                selected_experiments = [exp_names[idx]]
            else:
                print("Invalid choice, benchmarking all experiments")
                selected_experiments = exp_names
        except ValueError:
            print("Invalid choice, benchmarking all experiments")
            selected_experiments = exp_names
    
    # Run benchmarks
    all_results = []
    
    for i, exp_name in enumerate(selected_experiments):
        print(f"\n{'='*60}")
        print(f"BENCHMARKING {i+1}/{len(selected_experiments)}")
        print(f"{'='*60}")
        
        checkpoint_path = checkpoints[exp_name]
        
        try:
            result = run_benchmarks_for_experiment(exp_name, checkpoint_path)
            all_results.append(result)
            
            # Save individual result
            result_file = f"research_results/{exp_name}_benchmark_results.json"
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"💾 Results saved to: {result_file}")
            
        except Exception as e:
            print(f"❌ Benchmarking failed for {exp_name}: {e}")
            continue
    
    # Save combined results
    if all_results:
        combined_file = "research_results/all_benchmark_results.json"
        with open(combined_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\n💾 Combined results saved to: {combined_file}")
        
        # Compare experiments
        compare_experiments(all_results)
    
    print(f"\n✅ Benchmarking complete!")
    print(f"📁 Results saved in: research_results/")

if __name__ == "__main__":
    main()