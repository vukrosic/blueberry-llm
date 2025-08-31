"""
Benchmarking utilities that integrate with existing benchmark tools
"""
import os
import json
import glob
from typing import Dict, List, Any, Optional
import subprocess
import sys

class BenchmarkRunner:
    """Runner for benchmarking trained models"""
    
    def __init__(self, results_dir: str = "results"):
        self.results_dir = results_dir
    
    def find_experiment_checkpoints(self) -> Dict[str, str]:
        """Find all experiment checkpoints"""
        checkpoints = {}
        
        if not os.path.exists(self.results_dir):
            return checkpoints
        
        # Look for experiment directories
        for exp_dir in glob.glob(f"{self.results_dir}/*/"):
            exp_name = os.path.basename(exp_dir.rstrip('/'))
            
            # Look for checkpoints in this experiment
            checkpoint_dir = os.path.join(exp_dir, "checkpoints")
            if os.path.exists(checkpoint_dir):
                # Look for final model first, then latest checkpoint
                final_model = os.path.join(checkpoint_dir, "final_model.pt")
                if os.path.exists(final_model):
                    checkpoints[exp_name] = checkpoint_dir
                else:
                    # Find latest checkpoint
                    checkpoint_files = glob.glob(f"{checkpoint_dir}/checkpoint_step_*.pt")
                    if checkpoint_files:
                        # Get the most recent checkpoint
                        latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
                        checkpoints[exp_name] = checkpoint_dir
        
        return checkpoints
    
    def run_comprehensive_benchmark(self, checkpoint_dir: str, max_examples: int = 500) -> Dict[str, Any]:
        """Run comprehensive benchmark using existing tiny_llm_benchmarks.py"""
        try:
            # Import the existing benchmark
            from tiny_llm_benchmarks import TinyLLMBenchmark
            
            benchmark = TinyLLMBenchmark(checkpoint_dir)
            results = benchmark.run_all_benchmarks(save_results=False)
            
            return {
                'status': 'success',
                'results': results
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def run_hellaswag_benchmark(self, checkpoint_dir: str, max_examples: int = 200) -> Dict[str, Any]:
        """Run HellaSwag benchmark using existing hellaswag_benchmark.py"""
        try:
            # Import the existing benchmark
            from hellaswag_benchmark import HellaSwagEvaluator
            
            evaluator = HellaSwagEvaluator(checkpoint_dir)
            results = evaluator.evaluate(max_examples=max_examples, save_results=False)
            
            return {
                'status': 'success',
                'results': results
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def benchmark_experiment(self, exp_name: str, checkpoint_dir: str) -> Dict[str, Any]:
        """Run all benchmarks for a single experiment"""
        print(f"\n🧪 BENCHMARKING: {exp_name}")
        print("=" * 50)
        
        results = {
            'experiment_name': exp_name,
            'checkpoint_dir': checkpoint_dir,
            'benchmarks': {}
        }
        
        # Run comprehensive benchmarks
        print("📊 Running comprehensive benchmark suite...")
        comprehensive_results = self.run_comprehensive_benchmark(checkpoint_dir)
        results['benchmarks']['comprehensive'] = comprehensive_results
        
        if comprehensive_results['status'] == 'success':
            comp_results = comprehensive_results['results']
            if 'overall' in comp_results:
                overall = comp_results['overall']
                print(f"📈 Overall accuracy: {overall.get('accuracy', 'N/A'):.3f}")
        else:
            print(f"❌ Comprehensive benchmarks failed: {comprehensive_results.get('error', 'Unknown error')}")
        
        # Run HellaSwag benchmark
        print("\n🎯 Running HellaSwag benchmark...")
        hellaswag_results = self.run_hellaswag_benchmark(checkpoint_dir)
        results['benchmarks']['hellaswag'] = hellaswag_results
        
        if hellaswag_results['status'] == 'success':
            hella_results = hellaswag_results['results']
            if 'accuracy' in hella_results:
                print(f"🎯 HellaSwag accuracy: {hella_results['accuracy']:.3f}")
        else:
            print(f"❌ HellaSwag benchmark failed: {hellaswag_results.get('error', 'Unknown error')}")
        
        return results
    
    def benchmark_all_experiments(self) -> List[Dict[str, Any]]:
        """Benchmark all available experiments"""
        checkpoints = self.find_experiment_checkpoints()
        
        if not checkpoints:
            print("❌ No experiment checkpoints found!")
            return []
        
        print(f"📂 Found {len(checkpoints)} experiments to benchmark")
        
        all_results = []
        
        for i, (exp_name, checkpoint_dir) in enumerate(checkpoints.items()):
            print(f"\n{'='*60}")
            print(f"BENCHMARKING {i+1}/{len(checkpoints)}")
            print(f"{'='*60}")
            
            try:
                result = self.benchmark_experiment(exp_name, checkpoint_dir)
                all_results.append(result)
                
                # Save individual result
                result_file = os.path.join(self.results_dir, f"{exp_name}_benchmark_results.json")
                with open(result_file, 'w') as f:
                    json.dump(result, f, indent=2)
                print(f"💾 Results saved to: {result_file}")
                
            except Exception as e:
                print(f"❌ Benchmarking failed for {exp_name}: {e}")
                continue
        
        # Save combined results
        if all_results:
            combined_file = os.path.join(self.results_dir, "all_benchmark_results.json")
            with open(combined_file, 'w') as f:
                json.dump(all_results, f, indent=2)
            print(f"\n💾 Combined results saved to: {combined_file}")
            
            self._print_comparison(all_results)
        
        return all_results
    
    def _print_comparison(self, all_results: List[Dict[str, Any]]):
        """Print comparison of all experiments"""
        print(f"\n📊 EXPERIMENT COMPARISON")
        print("=" * 80)
        
        # Extract metrics for comparison
        comparison_data = []
        
        for result in all_results:
            exp_name = result['experiment_name']
            benchmarks = result['benchmarks']
            
            row = {'experiment': exp_name}
            
            # Get comprehensive benchmark overall accuracy
            if ('comprehensive' in benchmarks and 
                benchmarks['comprehensive']['status'] == 'success' and
                'overall' in benchmarks['comprehensive']['results']):
                overall = benchmarks['comprehensive']['results']['overall']
                row['overall_accuracy'] = overall.get('accuracy', 0)
            else:
                row['overall_accuracy'] = 0
            
            # Get HellaSwag accuracy
            if ('hellaswag' in benchmarks and 
                benchmarks['hellaswag']['status'] == 'success' and
                'accuracy' in benchmarks['hellaswag']['results']):
                row['hellaswag_accuracy'] = benchmarks['hellaswag']['results']['accuracy']
            else:
                row['hellaswag_accuracy'] = 0
            
            comparison_data.append(row)
        
        # Sort by overall accuracy
        comparison_data.sort(key=lambda x: x['overall_accuracy'], reverse=True)
        
        # Print comparison table
        print(f"{'Experiment':<30} {'Overall Acc':<12} {'HellaSwag Acc':<12}")
        print("-" * 80)
        
        for row in comparison_data:
            print(f"{row['experiment']:<30} "
                  f"{row['overall_accuracy']:<12.3f} "
                  f"{row['hellaswag_accuracy']:<12.3f}")
        
        # Find best performers
        if comparison_data:
            best_overall = comparison_data[0]
            best_hellaswag = max(comparison_data, key=lambda x: x['hellaswag_accuracy'])
            
            print(f"\n🏆 BEST PERFORMERS:")
            print(f"  Overall: {best_overall['experiment']} ({best_overall['overall_accuracy']:.3f})")
            print(f"  HellaSwag: {best_hellaswag['experiment']} ({best_hellaswag['hellaswag_accuracy']:.3f})")