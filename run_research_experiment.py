#!/usr/bin/env python3
"""
Research Experiment Runner
Integrates with existing training script to run systematic experiments
"""

import os
import sys
import json
import subprocess
import time
from research_framework import ExperimentConfig, ResearchTracker, ExperimentResult
from tiny_llm_benchmarks import TinyLLMBenchmark, find_checkpoints

def modify_training_config(config: ExperimentConfig, temp_file: str = "temp_train_config.py"):
    """Create a modified training script with the experiment configuration"""
    
    # Read the original training script
    with open("train_llm.py", 'r') as f:
        original_script = f.read()
    
    # Create modified version with new config
    modified_script = original_script.replace(
        "config = ModelConfig()",
        f"""config = ModelConfig(
    d_model={config.d_model},
    n_heads={config.n_heads}, 
    n_layers={config.n_layers},
    d_ff={config.d_ff},
    max_steps={config.max_steps},
    batch_size={config.batch_size},
    gradient_accumulation_steps={config.gradient_accumulation_steps},
    max_seq_len={config.max_seq_len},
    num_documents={config.num_documents},
    max_tokens={config.max_tokens},
    weight_decay={config.weight_decay},
    dropout={config.dropout},
    use_amp={config.use_amp}
)"""
    )
    
    # Write temporary script
    with open(temp_file, 'w') as f:
        f.write(modified_script)
    
    return temp_file

def run_training_experiment(config: ExperimentConfig) -> dict:
    """Run training with the given configuration"""
    print(f"🚀 Starting training with config: {config.experiment_name}")
    
    # Create temporary training script
    temp_script = modify_training_config(config)
    
    try:
        # Run training
        start_time = time.time()
        result = subprocess.run([sys.executable, temp_script], 
                              capture_output=True, text=True, timeout=3600)  # 1 hour timeout
        training_time = (time.time() - start_time) / 60
        
        if result.returncode != 0:
            print(f"❌ Training failed: {result.stderr}")
            return {
                'success': False,
                'error': result.stderr,
                'training_time': training_time
            }
        
        # Parse training output for metrics
        output_lines = result.stdout.split('\n')
        final_metrics = {}
        
        for line in output_lines:
            if "Final Results:" in line:
                # Look for the next few lines with metrics
                continue
            elif "Validation Loss:" in line:
                final_metrics['val_loss'] = float(line.split(':')[1].strip())
            elif "Validation Accuracy:" in line:
                final_metrics['val_accuracy'] = float(line.split(':')[1].strip())
            elif "Validation Perplexity:" in line:
                final_metrics['val_perplexity'] = float(line.split(':')[1].strip())
        
        return {
            'success': True,
            'training_time': training_time,
            'metrics': final_metrics,
            'output': result.stdout
        }
        
    except subprocess.TimeoutExpired:
        print("❌ Training timed out")
        return {
            'success': False,
            'error': 'Training timed out',
            'training_time': 60  # 1 hour
        }
    finally:
        # Clean up temporary script
        if os.path.exists(temp_script):
            os.remove(temp_script)

def run_benchmarks_on_latest_checkpoint() -> dict:
    """Run benchmarks on the most recent checkpoint"""
    checkpoints = find_checkpoints()
    if not checkpoints:
        print("❌ No checkpoints found")
        return {}
    
    latest_checkpoint = checkpoints[-1]
    print(f"📊 Running benchmarks on: {latest_checkpoint}")
    
    try:
        # Initialize benchmark
        benchmark = TinyLLMBenchmark(latest_checkpoint)
        
        # Run all benchmarks
        results = benchmark.run_all_benchmarks(save_results=False)
        
        # Extract just the accuracy scores for easier tracking
        simplified_results = {}
        for task_name, result in results.items():
            if task_name == 'overall':
                continue
            if 'error' in result:
                simplified_results[task_name] = {'error': result['error']}
            elif 'accuracy' in result:
                simplified_results[task_name] = {
                    'accuracy': result['accuracy'],
                    'correct': result['correct'],
                    'total': result['total']
                }
            elif 'exact_match_score' in result:
                simplified_results[task_name] = {
                    'exact_match': result['exact_match_score'],
                    'partial_match': result['partial_match_score'],
                    'total': result['total']
                }
        
        return simplified_results
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        return {'error': str(e)}

class IntegratedResearchRunner:
    """Integrates research framework with actual training and benchmarking"""
    
    def __init__(self, project_name: str = "tiny_llm_research"):
        self.tracker = ResearchTracker(project_name)
    
    def establish_baselines(self):
        """Establish baselines using actual training and benchmarking"""
        print("🎯 ESTABLISHING RESEARCH BASELINES")
        print("=" * 60)
        
        baselines = {}
        
        # 1. Theoretical random baselines
        print("📊 Theoretical Random Baselines:")
        random_baselines = {
            'hellaswag': 0.25,
            'piqa': 0.50,
            'siqa': 0.333,
            'lambada': 0.001,  # Very low for vocabulary-dependent tasks
            'arithmetic': 0.05,
            'sentence_completion': 0.1,
            'word_association': 0.1,
            'simple_qa': 0.1,
            'squad': 0.0
        }
        
        for task, baseline in random_baselines.items():
            print(f"  {task}: {baseline:.3f}")
        
        baselines['random'] = random_baselines
        
        # 2. Untrained model baseline (100 steps)
        print("\n🤖 Untrained Model Baseline (100 steps):")
        untrained_config = ExperimentConfig(
            max_steps=100,
            experiment_name="untrained_baseline",
            description="Minimal training to establish untrained model baseline",
            hypothesis="Untrained model should perform near random"
        )
        
        untrained_result = self.run_full_experiment(untrained_config, is_baseline=True)
        if untrained_result:
            baselines['untrained'] = untrained_result.benchmark_results
        
        # 3. Small trained model baseline (1000 steps)
        print("\n🚀 Small Trained Model Baseline (1000 steps):")
        small_trained_config = ExperimentConfig(
            max_steps=1000,
            experiment_name="small_trained_baseline",
            description="Small amount of training to establish trained model baseline", 
            hypothesis="Small training should show improvement over untrained"
        )
        
        small_trained_result = self.run_full_experiment(small_trained_config, is_baseline=True)
        if small_trained_result:
            baselines['small_trained'] = small_trained_result.benchmark_results
        
        # Save baselines
        self.tracker.baselines = baselines
        self.tracker._save_baselines()
        
        print("\n✅ Baselines established and saved!")
        self.tracker.print_baseline_summary()
        
        return baselines
    
    def run_full_experiment(self, config: ExperimentConfig, is_baseline: bool = False) -> ExperimentResult:
        """Run complete experiment: training + benchmarking"""
        experiment_id = f"{config.experiment_name}_{config.get_hash()}"
        
        print(f"\n🧪 Running Experiment: {experiment_id}")
        print(f"📝 Description: {config.description}")
        print(f"💡 Hypothesis: {config.hypothesis}")
        
        # Create experiment directory
        exp_dir = f"{self.tracker.results_dir}/experiments/{experiment_id}"
        os.makedirs(exp_dir, exist_ok=True)
        
        # Save config
        with open(f"{exp_dir}/config.json", 'w') as f:
            json.dump(config.to_dict(), f, indent=2)
        
        # Run training
        training_result = run_training_experiment(config)
        
        if not training_result['success']:
            print(f"❌ Training failed: {training_result['error']}")
            return None
        
        # Run benchmarks
        benchmark_results = run_benchmarks_on_latest_checkpoint()
        
        # Create result object
        import datetime
        import torch
        
        result = ExperimentResult(
            experiment_id=experiment_id,
            config=config,
            final_train_loss=training_result['metrics'].get('train_loss', 0),
            final_val_loss=training_result['metrics'].get('val_loss', 0),
            final_val_accuracy=training_result['metrics'].get('val_accuracy', 0),
            final_val_perplexity=training_result['metrics'].get('val_perplexity', 0),
            training_time_minutes=training_result['training_time'],
            benchmark_results=benchmark_results,
            timestamp=datetime.datetime.now().isoformat(),
            git_commit=self.tracker.get_git_commit(),
            cuda_device=torch.cuda.get_device_name() if torch.cuda.is_available() else None
        )
        
        # Save result
        with open(f"{exp_dir}/result.json", 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        
        # Add to experiments list
        if not is_baseline:
            self.tracker.experiments.append(result)
            self.tracker._save_experiments()
        
        print(f"✅ Experiment completed in {training_result['training_time']:.1f} minutes")
        
        # Print benchmark summary
        print("\n📊 Benchmark Results:")
        for task, result_data in benchmark_results.items():
            if 'error' in result_data:
                print(f"  {task}: ERROR - {result_data['error']}")
            elif 'accuracy' in result_data:
                print(f"  {task}: {result_data['accuracy']:.3f}")
            elif 'exact_match' in result_data:
                print(f"  {task}: EM={result_data['exact_match']:.3f}, PM={result_data['partial_match']:.3f}")
        
        return result

def main():
    """Main research workflow"""
    runner = IntegratedResearchRunner("tiny_llm_research")
    
    print("🔬 TINY LLM RESEARCH FRAMEWORK")
    print("=" * 50)
    
    # Check if baselines exist
    if not runner.tracker.baselines:
        print("📊 No baselines found. Establishing baselines...")
        runner.establish_baselines()
    else:
        print("📊 Baselines already established:")
        runner.tracker.print_baseline_summary()
        
        establish_new = input("\nRe-establish baselines? (y/n): ").strip().lower()
        if establish_new == 'y':
            runner.establish_baselines()
    
    # Interactive experiment runner
    while True:
        print("\n🧪 EXPERIMENT OPTIONS:")
        print("1. Run predefined experiment")
        print("2. Create custom experiment")
        print("3. View results")
        print("4. Generate report")
        print("5. Exit")
        
        choice = input("\nChoose option (1-5): ").strip()
        
        if choice == '1':
            # Predefined experiments
            experiments = [
                ExperimentConfig(
                    experiment_name="longer_training",
                    description="Train for 3000 steps instead of baseline 1000",
                    hypothesis="More training should improve all benchmark scores",
                    max_steps=3000
                ),
                ExperimentConfig(
                    experiment_name="larger_model",
                    description="Increase model size (512d, 8 layers)",
                    hypothesis="Larger model should have better capacity",
                    d_model=512,
                    n_layers=8,
                    d_ff=2048,
                    max_steps=2000
                ),
                ExperimentConfig(
                    experiment_name="higher_lr",
                    description="Test higher learning rate",
                    hypothesis="Higher LR might train faster",
                    learning_rate=0.02,
                    max_steps=2000
                )
            ]
            
            print("\nPredefined experiments:")
            for i, exp in enumerate(experiments):
                print(f"  {i+1}. {exp.experiment_name}: {exp.description}")
            
            exp_choice = input(f"Choose experiment (1-{len(experiments)}): ").strip()
            try:
                exp_idx = int(exp_choice) - 1
                if 0 <= exp_idx < len(experiments):
                    runner.run_full_experiment(experiments[exp_idx])
                else:
                    print("Invalid choice")
            except ValueError:
                print("Invalid input")
        
        elif choice == '2':
            print("Custom experiment creation not implemented yet")
            
        elif choice == '3':
            print(f"\n📊 Total experiments: {len(runner.tracker.experiments)}")
            for exp in runner.tracker.experiments:
                print(f"  - {exp.experiment_id}: {exp.config.description}")
                
        elif choice == '4':
            report = runner.tracker.generate_report()
            print(f"\n📋 Report generated: {runner.tracker.results_dir}/research_report.md")
            
        elif choice == '5':
            break
            
        else:
            print("Invalid choice")

if __name__ == "__main__":
    main()