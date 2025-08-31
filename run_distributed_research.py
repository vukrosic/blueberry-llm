#!/usr/bin/env python3
"""
Distributed Research Experiment Runner for 8x RTX 4090
Adapted for multi-GPU training with systematic experiment tracking
"""

import os
import sys
import json
import subprocess
import time
import torch
from research_framework import ExperimentConfig, ResearchTracker, ExperimentResult
from tiny_llm_benchmarks import TinyLLMBenchmark, find_checkpoints

class DistributedExperimentConfig(ExperimentConfig):
    """Extended config for distributed training"""
    def __init__(self, **kwargs):
        # Extract distributed-specific params
        num_gpus = kwargs.pop('num_gpus', 8)
        
        # Set distributed training defaults
        kwargs.setdefault('max_steps', 500)  # Short test runs
        kwargs.setdefault('eval_every', 100)  # Evaluate more frequently
        kwargs.setdefault('save_every', 500)  # Save at the end
        
        super().__init__(**kwargs)
        
        # Store num_gpus as instance variable
        self.num_gpus = num_gpus
    
    def to_dict(self):
        result = super().to_dict()
        result['num_gpus'] = self.num_gpus
        return result

def create_distributed_training_script(config: DistributedExperimentConfig, exp_dir: str):
    """Create a distributed training script for this experiment"""
    
    script_content = f'''#!/usr/bin/env python3
"""
Distributed training script for experiment: {config.experiment_name}
Auto-generated for 8x RTX 4090 setup
"""

import os
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from train_distributed_llm import *

def main():
    # Set distributed training parameters
    os.environ["WORLD_SIZE"] = "8"
    os.environ["MASTER_ADDR"] = "localhost" 
    os.environ["MASTER_PORT"] = "12355"
    
    # Override model config
    config = ModelConfig(
        # Architecture
        d_model={config.d_model},
        n_heads={config.n_heads},
        n_layers={config.n_layers},
        d_ff={config.d_ff},
        
        # Training
        max_steps={config.max_steps},
        batch_size={config.batch_size},
        gradient_accumulation_steps={config.gradient_accumulation_steps},
        
        # Data
        max_seq_len={config.max_seq_len},
        num_documents={config.num_documents},
        max_tokens={config.max_tokens},
        
        # Optimization
        weight_decay={config.weight_decay},
        dropout={config.dropout},
        use_amp={config.use_amp},
        
        # Evaluation
        eval_every={config.eval_every},
        save_every={config.save_every},
        
        # Checkpointing
        checkpoint_dir="{exp_dir}/checkpoints"
    )
    
    print(f"🚀 Starting distributed training: {config.experiment_name}")
    print(f"📊 Model: {{config.d_model}}d, {{config.n_layers}}L, {{config.n_heads}}H")
    print(f"🔧 Steps: {{config.max_steps}}, LR: {config.learning_rate}")
    
    # Run distributed training
    try:
        # Use the distributed training function from train_distributed_llm.py
        run_novita_4090_training()
    except Exception as e:
        print(f"❌ Training failed: {{e}}")
        sys.exit(1)
    
    print("✅ Training completed successfully")

if __name__ == "__main__":
    main()
'''
    
    script_path = f"{exp_dir}/train_experiment.py"
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    # Make executable
    os.chmod(script_path, 0o755)
    return script_path

def run_distributed_training(config: DistributedExperimentConfig, exp_dir: str):
    """Run distributed training for the experiment"""
    print(f"🚀 Starting distributed training: {config.experiment_name}")
    print(f"📊 Architecture: {config.d_model}d, {config.n_layers}L, {config.n_heads}H")
    print(f"🔧 Config: {config.max_steps} steps, LR {config.learning_rate}, seq_len {config.max_seq_len}")
    
    # Create training script
    script_path = create_distributed_training_script(config, exp_dir)
    
    # Run distributed training
    start_time = time.time()
    
    try:
        # Use torchrun for distributed training
        cmd = [
            "torchrun",
            "--nproc_per_node=8",
            "--master_port=12355",
            script_path
        ]
        
        print(f"🔄 Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30 min timeout
        
        training_time = (time.time() - start_time) / 60
        
        if result.returncode != 0:
            print(f"❌ Training failed:")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            return {
                'success': False,
                'error': result.stderr,
                'training_time': training_time,
                'stdout': result.stdout
            }
        
        # Parse training output for metrics
        output_lines = result.stdout.split('\n')
        final_metrics = {}
        
        for line in output_lines:
            if "Final - Loss:" in line:
                try:
                    parts = line.split(',')
                    for part in parts:
                        if "Loss:" in part:
                            final_metrics['val_loss'] = float(part.split(':')[1].strip())
                        elif "Acc:" in part:
                            final_metrics['val_accuracy'] = float(part.split(':')[1].strip())
                        elif "PPL:" in part:
                            final_metrics['val_perplexity'] = float(part.split(':')[1].strip())
                except:
                    pass
        
        return {
            'success': True,
            'training_time': training_time,
            'metrics': final_metrics,
            'output': result.stdout
        }
        
    except subprocess.TimeoutExpired:
        print("❌ Training timed out (30 minutes)")
        return {
            'success': False,
            'error': 'Training timed out after 30 minutes',
            'training_time': 30
        }
    except Exception as e:
        print(f"❌ Training failed with exception: {e}")
        return {
            'success': False,
            'error': str(e),
            'training_time': (time.time() - start_time) / 60
        }

class DistributedResearchRunner:
    """Research runner adapted for 8x RTX 4090 distributed training"""
    
    def __init__(self, project_name: str = "distributed_tiny_llm_research"):
        self.tracker = ResearchTracker(project_name)
        self.verify_gpu_setup()
    
    def verify_gpu_setup(self):
        """Verify 8x RTX 4090 setup"""
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available!")
        
        gpu_count = torch.cuda.device_count()
        print(f"🔍 Detected {gpu_count} GPUs")
        
        for i in range(min(gpu_count, 8)):
            gpu_name = torch.cuda.get_device_name(i)
            memory_gb = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"  GPU {i}: {gpu_name} ({memory_gb:.1f} GB)")
        
        if gpu_count < 8:
            print(f"⚠️ Warning: Only {gpu_count} GPUs available, expected 8")
    
    def run_test_experiments(self):
        """Run a small set of test experiments to verify everything works"""
        print("🧪 RUNNING TEST EXPERIMENTS (500 steps each)")
        print("=" * 60)
        
        # Define test experiments - just 3 to verify the system works
        test_experiments = [
            DistributedExperimentConfig(
                experiment_name="test_small_arch",
                description="Test small architecture (500 steps)",
                hypothesis="Small model should train quickly and show basic learning",
                d_model=256, n_layers=4, n_heads=4, d_ff=1024,
                learning_rate=0.01, max_seq_len=512,
                max_steps=500
            ),
            DistributedExperimentConfig(
                experiment_name="test_medium_arch", 
                description="Test medium architecture (500 steps)",
                hypothesis="Medium model should show better performance than small",
                d_model=384, n_layers=6, n_heads=8, d_ff=1536,
                learning_rate=0.01, max_seq_len=512,
                max_steps=500
            ),
            DistributedExperimentConfig(
                experiment_name="test_lr_ablation",
                description="Test different learning rate (500 steps)",
                hypothesis="Higher learning rate should train faster initially",
                d_model=384, n_layers=6, n_heads=8, d_ff=1536,
                learning_rate=0.02, max_seq_len=512,
                max_steps=500
            )
        ]
        
        results = []
        
        for i, config in enumerate(test_experiments):
            print(f"\n🔄 Running test experiment {i+1}/3: {config.experiment_name}")
            
            try:
                result = self.run_full_experiment(config, is_test=True)
                if result:
                    results.append(result)
                    print(f"✅ Test {i+1} completed successfully")
                else:
                    print(f"❌ Test {i+1} failed")
            except Exception as e:
                print(f"❌ Test {i+1} failed with exception: {e}")
                continue
        
        # Summary
        print(f"\n📊 TEST SUMMARY")
        print("=" * 40)
        print(f"Completed: {len(results)}/3 experiments")
        
        if results:
            print("\nResults:")
            for result in results:
                print(f"  {result.experiment_id}:")
                print(f"    Training time: {result.training_time_minutes:.1f} min")
                print(f"    Val loss: {result.final_val_loss:.4f}")
                print(f"    Val accuracy: {result.final_val_accuracy:.4f}")
        
        return results
    
    def run_full_experiment(self, config: DistributedExperimentConfig, is_test: bool = False):
        """Run a complete distributed experiment"""
        experiment_id = f"{config.experiment_name}_{config.get_hash()}"
        
        print(f"\n🧪 Running Experiment: {experiment_id}")
        print(f"📝 Description: {config.description}")
        print(f"💡 Hypothesis: {config.hypothesis}")
        
        # Create experiment directory
        exp_dir = f"{self.tracker.results_dir}/experiments/{experiment_id}"
        os.makedirs(exp_dir, exist_ok=True)
        os.makedirs(f"{exp_dir}/checkpoints", exist_ok=True)
        
        # Save config
        with open(f"{exp_dir}/config.json", 'w') as f:
            json.dump(config.to_dict(), f, indent=2)
        
        # Run distributed training
        training_result = run_distributed_training(config, exp_dir)
        
        if not training_result['success']:
            print(f"❌ Training failed: {training_result['error']}")
            # Save failed result
            with open(f"{exp_dir}/training_failed.json", 'w') as f:
                json.dump(training_result, f, indent=2)
            return None
        
        # Find the checkpoint that was created
        checkpoints = find_checkpoints(f"{exp_dir}/checkpoints")
        if not checkpoints:
            print("❌ No checkpoints found after training")
            return None
        
        latest_checkpoint = checkpoints[-1]
        print(f"📂 Found checkpoint: {latest_checkpoint}")
        
        # Run benchmarks (only if not a test run)
        benchmark_results = {}
        if not is_test:
            print("📊 Running benchmarks...")
            try:
                benchmark = TinyLLMBenchmark(latest_checkpoint)
                benchmark_results = benchmark.run_all_benchmarks(save_results=False)
                
                # Simplify results for storage
                simplified_results = {}
                for task_name, result in benchmark_results.items():
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
                benchmark_results = simplified_results
                
            except Exception as e:
                print(f"❌ Benchmark failed: {e}")
                benchmark_results = {'error': str(e)}
        else:
            print("⏭️ Skipping benchmarks for test run")
        
        # Create result object
        import datetime
        
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
            cuda_device=f"8x {torch.cuda.get_device_name(0)}"
        )
        
        # Save result
        with open(f"{exp_dir}/result.json", 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        
        # Save training output
        with open(f"{exp_dir}/training_output.txt", 'w') as f:
            f.write(training_result['output'])
        
        # Add to experiments list (only if not test)
        if not is_test:
            self.tracker.experiments.append(result)
            self.tracker._save_experiments()
        
        print(f"✅ Experiment completed in {training_result['training_time']:.1f} minutes")
        
        # Print benchmark summary (if available)
        if benchmark_results and 'error' not in benchmark_results:
            print("\n📊 Benchmark Results:")
            for task, result_data in benchmark_results.items():
                if 'error' in result_data:
                    print(f"  {task}: ERROR")
                elif 'accuracy' in result_data:
                    print(f"  {task}: {result_data['accuracy']:.3f}")
                elif 'exact_match' in result_data:
                    print(f"  {task}: EM={result_data['exact_match']:.3f}")
        
        return result

def main():
    """Main entry point for distributed research"""
    print("🔬 DISTRIBUTED TINY LLM RESEARCH (8x RTX 4090)")
    print("=" * 60)
    
    # Initialize runner
    try:
        runner = DistributedResearchRunner()
    except Exception as e:
        print(f"❌ Failed to initialize runner: {e}")
        return
    
    print("\n🧪 RUNNING TEST EXPERIMENTS")
    print("This will run 3 short experiments (500 steps each) to verify everything works")
    
    confirm = input("Continue with test experiments? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Exiting...")
        return
    
    # Run test experiments
    results = runner.run_test_experiments()
    
    if len(results) >= 2:
        print("\n✅ Test experiments successful!")
        print("🎯 System is ready for full research experiments")
        print("\nNext steps:")
        print("1. Review test results in research_results/")
        print("2. Run full experiments with longer training")
        print("3. Analyze scaling laws and performance")
    else:
        print("\n❌ Test experiments failed")
        print("🔧 Check GPU setup and training configuration")

if __name__ == "__main__":
    main()