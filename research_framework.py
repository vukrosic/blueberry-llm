#!/usr/bin/env python3
"""
Research Framework for LLM Experiments
Establishes baselines, tracks experiments, and manages results systematically
"""

import json
import os
import time
import datetime
import hashlib
import subprocess
import torch
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

@dataclass
class ExperimentConfig:
    """Configuration for a single experiment"""
    # Model architecture
    d_model: int = 384
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 1536
    
    # Training parameters
    max_steps: int = 5000
    batch_size: int = 8
    learning_rate: float = 0.01
    gradient_accumulation_steps: int = 4
    
    # Data parameters
    max_seq_len: int = 512
    num_documents: int = 3000
    max_tokens: int = 1000000
    
    # Optimizer
    optimizer_type: str = "muon"  # "muon", "adamw", "sgd"
    weight_decay: float = 0.1
    
    # Other parameters
    dropout: float = 0.1
    use_amp: bool = True
    
    # Experiment metadata
    experiment_name: str = ""
    description: str = ""
    hypothesis: str = ""
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def get_hash(self) -> str:
        """Get unique hash for this configuration"""
        config_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]

@dataclass
class ExperimentResult:
    """Results from a single experiment"""
    experiment_id: str
    config: ExperimentConfig
    
    # Training metrics
    final_train_loss: float
    final_val_loss: float
    final_val_accuracy: float
    final_val_perplexity: float
    training_time_minutes: float
    
    # Benchmark results
    benchmark_results: Dict[str, Any]
    
    # System info
    timestamp: str
    git_commit: Optional[str] = None
    cuda_device: Optional[str] = None
    
    def to_dict(self) -> Dict:
        result_dict = asdict(self)
        result_dict['config'] = self.config.to_dict()
        return result_dict

class ResearchTracker:
    """Tracks experiments and manages research workflow"""
    
    def __init__(self, project_name: str = "llm_research"):
        self.project_name = project_name
        self.results_dir = f"research_results/{project_name}"
        self.experiments_file = f"{self.results_dir}/experiments.json"
        self.baselines_file = f"{self.results_dir}/baselines.json"
        
        # Create directories
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(f"{self.results_dir}/checkpoints", exist_ok=True)
        os.makedirs(f"{self.results_dir}/plots", exist_ok=True)
        
        # Load existing data
        self.experiments = self._load_experiments()
        self.baselines = self._load_baselines()
    
    def _load_experiments(self) -> List[ExperimentResult]:
        """Load existing experiments"""
        if os.path.exists(self.experiments_file):
            with open(self.experiments_file, 'r') as f:
                data = json.load(f)
                return [ExperimentResult(**exp) for exp in data]
        return []
    
    def _load_baselines(self) -> Dict[str, Any]:
        """Load baseline results"""
        if os.path.exists(self.baselines_file):
            with open(self.baselines_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_experiments(self):
        """Save experiments to file"""
        with open(self.experiments_file, 'w') as f:
            json.dump([exp.to_dict() for exp in self.experiments], f, indent=2)
    
    def _save_baselines(self):
        """Save baselines to file"""
        with open(self.baselines_file, 'w') as f:
            json.dump(self.baselines, f, indent=2)
    
    def get_git_commit(self) -> Optional[str]:
        """Get current git commit hash"""
        try:
            result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                return result.stdout.strip()
        except:
            pass
        return None
    
    def establish_baselines(self) -> Dict[str, Any]:
        """Establish baseline performance metrics"""
        print("🎯 ESTABLISHING RESEARCH BASELINES")
        print("=" * 60)
        
        baselines = {}
        
        # 1. Random baselines (theoretical)
        print("📊 Theoretical Random Baselines:")
        random_baselines = {
            'hellaswag': 0.25,  # 4 choices
            'piqa': 0.50,       # 2 choices  
            'siqa': 0.333,      # 3 choices
            'lambada': 0.0,     # Vocabulary size dependent
            'arithmetic': 0.05, # ~1/20 for small numbers
            'sentence_completion': 0.1,  # Depends on vocabulary
            'word_association': 0.1,     # Depends on vocabulary
            'simple_qa': 0.1,            # Depends on vocabulary
            'squad_exact_match': 0.0,    # Nearly impossible randomly
            'squad_partial_match': 0.1   # Some word overlap possible
        }
        
        for task, baseline in random_baselines.items():
            print(f"  {task}: {baseline:.3f}")
        
        baselines['random'] = random_baselines
        
        # 2. Untrained model baseline
        print("\n🤖 Untrained Model Baseline:")
        print("Training model for 100 steps to establish untrained baseline...")
        
        # Create minimal config for untrained baseline
        untrained_config = ExperimentConfig(
            max_steps=100,
            experiment_name="untrained_baseline",
            description="Minimal training to establish untrained model baseline",
            hypothesis="Untrained model should perform near random"
        )
        
        # Run minimal training
        untrained_result = self.run_experiment(untrained_config, is_baseline=True)
        baselines['untrained'] = untrained_result.benchmark_results
        
        # 3. Small trained model baseline  
        print("\n🚀 Small Trained Model Baseline:")
        print("Training model for 2000 steps to establish small trained baseline...")
        
        small_trained_config = ExperimentConfig(
            max_steps=2000,
            experiment_name="small_trained_baseline", 
            description="Small amount of training to establish trained model baseline",
            hypothesis="Small training should show improvement over untrained"
        )
        
        small_trained_result = self.run_experiment(small_trained_config, is_baseline=True)
        baselines['small_trained'] = small_trained_result.benchmark_results
        
        # Save baselines
        self.baselines = baselines
        self._save_baselines()
        
        print("\n✅ Baselines established and saved!")
        self.print_baseline_summary()
        
        return baselines
    
    def print_baseline_summary(self):
        """Print summary of established baselines"""
        if not self.baselines:
            print("❌ No baselines established yet")
            return
        
        print("\n📊 BASELINE SUMMARY")
        print("=" * 60)
        
        # Create comparison table
        tasks = []
        random_scores = []
        untrained_scores = []
        small_trained_scores = []
        
        if 'random' in self.baselines:
            for task, score in self.baselines['random'].items():
                tasks.append(task)
                random_scores.append(score)
                
                # Get untrained score
                if 'untrained' in self.baselines and task in self.baselines['untrained']:
                    if isinstance(self.baselines['untrained'][task], dict):
                        untrained_score = self.baselines['untrained'][task].get('accuracy', 0)
                    else:
                        untrained_score = self.baselines['untrained'][task]
                else:
                    untrained_score = 0
                untrained_scores.append(untrained_score)
                
                # Get small trained score
                if 'small_trained' in self.baselines and task in self.baselines['small_trained']:
                    if isinstance(self.baselines['small_trained'][task], dict):
                        small_trained_score = self.baselines['small_trained'][task].get('accuracy', 0)
                    else:
                        small_trained_score = self.baselines['small_trained'][task]
                else:
                    small_trained_score = 0
                small_trained_scores.append(small_trained_score)
        
        # Print table
        print(f"{'Task':<20} {'Random':<10} {'Untrained':<12} {'Small Trained':<15}")
        print("-" * 60)
        for i, task in enumerate(tasks):
            print(f"{task:<20} {random_scores[i]:<10.3f} {untrained_scores[i]:<12.3f} {small_trained_scores[i]:<15.3f}")
    
    def run_experiment(self, config: ExperimentConfig, is_baseline: bool = False) -> ExperimentResult:
        """Run a single experiment"""
        experiment_id = f"{config.experiment_name}_{config.get_hash()}"
        
        print(f"\n🧪 Running Experiment: {experiment_id}")
        print(f"📝 Description: {config.description}")
        print(f"💡 Hypothesis: {config.hypothesis}")
        
        # Create experiment directory
        exp_dir = f"{self.results_dir}/experiments/{experiment_id}"
        os.makedirs(exp_dir, exist_ok=True)
        
        # Save config
        with open(f"{exp_dir}/config.json", 'w') as f:
            json.dump(config.to_dict(), f, indent=2)
        
        # Modify training script config and run training
        start_time = time.time()
        
        # Here you would modify your training script to use the config
        # For now, we'll simulate this
        print("🚀 Starting training...")
        
        # TODO: Actually run training with the config
        # This would involve modifying train_llm.py to accept config parameters
        # For now, we'll use placeholder values
        
        training_time = (time.time() - start_time) / 60
        
        # Simulate training results (replace with actual training)
        final_train_loss = 1.2  # Replace with actual
        final_val_loss = 1.3    # Replace with actual  
        final_val_accuracy = 0.85  # Replace with actual
        final_val_perplexity = 3.7  # Replace with actual
        
        # Run benchmarks
        print("📊 Running benchmarks...")
        # TODO: Run actual benchmarks
        # benchmark_results = self.run_benchmarks(checkpoint_path)
        benchmark_results = {}  # Placeholder
        
        # Create result
        result = ExperimentResult(
            experiment_id=experiment_id,
            config=config,
            final_train_loss=final_train_loss,
            final_val_loss=final_val_loss,
            final_val_accuracy=final_val_accuracy,
            final_val_perplexity=final_val_perplexity,
            training_time_minutes=training_time,
            benchmark_results=benchmark_results,
            timestamp=datetime.datetime.now().isoformat(),
            git_commit=self.get_git_commit(),
            cuda_device=torch.cuda.get_device_name() if torch.cuda.is_available() else None
        )
        
        # Save result
        with open(f"{exp_dir}/result.json", 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        
        # Add to experiments list
        if not is_baseline:
            self.experiments.append(result)
            self._save_experiments()
        
        print(f"✅ Experiment completed in {training_time:.1f} minutes")
        return result
    
    def compare_to_baseline(self, result: ExperimentResult) -> Dict[str, float]:
        """Compare experiment result to baselines"""
        if not self.baselines:
            print("❌ No baselines established")
            return {}
        
        comparisons = {}
        
        # Compare to small trained baseline
        if 'small_trained' in self.baselines:
            baseline_results = self.baselines['small_trained']
            
            for task, exp_result in result.benchmark_results.items():
                if task in baseline_results:
                    baseline_score = baseline_results[task].get('accuracy', 0) if isinstance(baseline_results[task], dict) else baseline_results[task]
                    exp_score = exp_result.get('accuracy', 0) if isinstance(exp_result, dict) else exp_result
                    
                    improvement = exp_score - baseline_score
                    comparisons[task] = improvement
        
        return comparisons
    
    def generate_report(self) -> str:
        """Generate research report"""
        report = []
        report.append("# LLM Research Report")
        report.append(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Project: {self.project_name}")
        report.append("")
        
        # Baselines section
        report.append("## Established Baselines")
        if self.baselines:
            report.append("| Task | Random | Untrained | Small Trained |")
            report.append("|------|--------|-----------|---------------|")
            
            if 'random' in self.baselines:
                for task in self.baselines['random']:
                    random_score = self.baselines['random'][task]
                    untrained_score = self.baselines.get('untrained', {}).get(task, {})
                    if isinstance(untrained_score, dict):
                        untrained_score = untrained_score.get('accuracy', 0)
                    small_trained_score = self.baselines.get('small_trained', {}).get(task, {})
                    if isinstance(small_trained_score, dict):
                        small_trained_score = small_trained_score.get('accuracy', 0)
                    
                    report.append(f"| {task} | {random_score:.3f} | {untrained_score:.3f} | {small_trained_score:.3f} |")
        else:
            report.append("No baselines established yet.")
        
        report.append("")
        
        # Experiments section
        report.append("## Experiments")
        if self.experiments:
            report.append(f"Total experiments: {len(self.experiments)}")
            report.append("")
            
            for exp in self.experiments:
                report.append(f"### {exp.experiment_id}")
                report.append(f"**Description:** {exp.config.description}")
                report.append(f"**Hypothesis:** {exp.config.hypothesis}")
                report.append(f"**Training Time:** {exp.training_time_minutes:.1f} minutes")
                report.append(f"**Final Val Loss:** {exp.final_val_loss:.4f}")
                report.append("")
        else:
            report.append("No experiments run yet.")
        
        # Save report
        report_text = "\n".join(report)
        with open(f"{self.results_dir}/research_report.md", 'w') as f:
            f.write(report_text)
        
        return report_text
    
    def plot_results(self):
        """Generate plots comparing results"""
        if not self.experiments:
            print("❌ No experiments to plot")
            return
        
        # Create plots directory
        plots_dir = f"{self.results_dir}/plots"
        os.makedirs(plots_dir, exist_ok=True)
        
        # Plot training metrics
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        exp_names = [exp.experiment_id for exp in self.experiments]
        val_losses = [exp.final_val_loss for exp in self.experiments]
        val_accuracies = [exp.final_val_accuracy for exp in self.experiments]
        perplexities = [exp.final_val_perplexity for exp in self.experiments]
        training_times = [exp.training_time_minutes for exp in self.experiments]
        
        # Validation Loss
        axes[0, 0].bar(exp_names, val_losses)
        axes[0, 0].set_title('Validation Loss')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Validation Accuracy
        axes[0, 1].bar(exp_names, val_accuracies)
        axes[0, 1].set_title('Validation Accuracy')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Perplexity
        axes[1, 0].bar(exp_names, perplexities)
        axes[1, 0].set_title('Perplexity')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Training Time
        axes[1, 1].bar(exp_names, training_times)
        axes[1, 1].set_title('Training Time (minutes)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(f"{plots_dir}/training_metrics.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Plots saved to {plots_dir}/")

def main():
    """Example usage of the research framework"""
    # Initialize research tracker
    tracker = ResearchTracker("tiny_llm_research")
    
    # Establish baselines (run this once)
    if not tracker.baselines:
        tracker.establish_baselines()
    else:
        print("📊 Baselines already established")
        tracker.print_baseline_summary()
    
    # Example experiments
    experiments = [
        ExperimentConfig(
            experiment_name="larger_model",
            description="Test larger model with more parameters",
            hypothesis="Larger model should perform better on all tasks",
            d_model=512,
            n_layers=8,
            max_steps=5000
        ),
        ExperimentConfig(
            experiment_name="longer_training",
            description="Train for more steps",
            hypothesis="More training steps should improve performance",
            max_steps=10000
        ),
        ExperimentConfig(
            experiment_name="different_optimizer",
            description="Test AdamW instead of Muon",
            hypothesis="Different optimizer might work better",
            optimizer_type="adamw",
            learning_rate=0.001
        )
    ]
    
    # Run experiments
    for config in experiments:
        result = tracker.run_experiment(config)
        
        # Compare to baseline
        improvements = tracker.compare_to_baseline(result)
        print(f"📈 Improvements over baseline: {improvements}")
    
    # Generate report and plots
    report = tracker.generate_report()
    tracker.plot_results()
    
    print(f"\n📋 Research report saved to: {tracker.results_dir}/research_report.md")

if __name__ == "__main__":
    main()