#!/usr/bin/env python3
"""
Research Experiments for Distributed LLM Training
Based on train_distributed_llm.py structure and functions

This script runs systematic experiments using the actual distributed training code.
"""

import os
import sys
import json
import time
import subprocess
import torch
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional
import datetime
import hashlib

# Import from the actual training script
from train_distributed_llm import (
    ModelConfig, 
    run_novita_4090_training,
    launch_distributed_ddp,
    launch_distributed,
    run_training_direct,
    NUM_GPUS,
    BASE_BATCH_SIZE,
    BASE_LR,
    SCALE_LR_WITH_GPUS
)

@dataclass
class ExperimentConfig:
    """Configuration for a research experiment"""
    # Experiment metadata
    name: str
    description: str
    hypothesis: str
    
    # Model architecture parameters (override ModelConfig defaults)
    d_model: Optional[int] = None
    n_heads: Optional[int] = None  
    n_layers: Optional[int] = None
    d_ff: Optional[int] = None
    
    # Training parameters
    max_steps: Optional[int] = None
    batch_size: Optional[int] = None
    muon_lr: Optional[float] = None
    gradient_accumulation_steps: Optional[int] = None
    
    # Data parameters
    max_seq_len: Optional[int] = None
    num_documents: Optional[int] = None
    max_tokens: Optional[int] = None
    
    # Evaluation
    eval_every: Optional[int] = None
    eval_steps: Optional[int] = None
    
    # Regularization
    weight_decay: Optional[float] = None
    dropout: Optional[float] = None
    grad_clip: Optional[float] = None
    
    # Training method
    training_method: str = "ddp"  # "ddp", "custom", "single"
    
    def get_hash(self) -> str:
        """Get unique hash for this configuration"""
        config_str = json.dumps(asdict(self), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]

class ResearchRunner:
    """Runs systematic experiments using the distributed training code"""
    
    def __init__(self, results_dir: str = "research_results"):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
        # Verify GPU setup
        self.verify_setup()
    
    def verify_setup(self):
        """Verify the distributed training setup"""
        print("🔍 VERIFYING DISTRIBUTED TRAINING SETUP")
        print("=" * 50)
        
        if not torch.cuda.is_available():
            print("❌ CUDA not available!")
            return False
        
        gpu_count = torch.cuda.device_count()
        print(f"📊 Available GPUs: {gpu_count}")
        
        for i in range(min(gpu_count, 8)):
            gpu_name = torch.cuda.get_device_name(i)
            memory_gb = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"  GPU {i}: {gpu_name} ({memory_gb:.1f} GB)")
        
        print(f"🔧 Configured for {NUM_GPUS} GPUs")
        print(f"📦 Base batch size: {BASE_BATCH_SIZE}")
        print(f"📈 Base learning rate: {BASE_LR}")
        print(f"🔄 LR scaling: {'ON' if SCALE_LR_WITH_GPUS else 'OFF'}")
        
        return True
    
    def create_model_config(self, exp_config: ExperimentConfig) -> ModelConfig:
        """Create ModelConfig from ExperimentConfig"""
        # Start with default ModelConfig
        config = ModelConfig()
        
        # Override with experiment-specific values
        for field_name, field_value in asdict(exp_config).items():
            if field_value is not None and hasattr(config, field_name):
                setattr(config, field_name, field_value)
        
        return config
    
    def run_experiment(self, exp_config: ExperimentConfig) -> Dict[str, Any]:
        """Run a single experiment"""
        print(f"\n🧪 RUNNING EXPERIMENT: {exp_config.name}")
        print("=" * 60)
        print(f"📝 Description: {exp_config.description}")
        print(f"💡 Hypothesis: {exp_config.hypothesis}")
        
        # Create experiment directory
        exp_hash = exp_config.get_hash()
        exp_dir = f"{self.results_dir}/{exp_config.name}_{exp_hash}"
        os.makedirs(exp_dir, exist_ok=True)
        
        # Save experiment config
        with open(f"{exp_dir}/experiment_config.json", 'w') as f:
            json.dump(asdict(exp_config), f, indent=2)
        
        # Create model config
        model_config = self.create_model_config(exp_config)
        
        # Save model config
        with open(f"{exp_dir}/model_config.json", 'w') as f:
            json.dump(asdict(model_config), f, indent=2)
        
        print(f"📊 Model: {model_config.d_model}d, {model_config.n_layers}L, {model_config.n_heads}H")
        print(f"🔧 Training: {model_config.max_steps} steps, LR {model_config.muon_lr}")
        print(f"📦 Batch: {model_config.batch_size} per GPU, {model_config.gradient_accumulation_steps} accum")
        
        # Temporarily modify the global ModelConfig in the training script
        # This is a bit hacky but works with the existing structure
        original_config_file = "temp_model_config.py"
        self.create_temp_config_file(model_config, original_config_file)
        
        # Run the training
        start_time = time.time()
        result = self.run_training(exp_config, exp_dir)
        training_time = (time.time() - start_time) / 60
        
        # Clean up temp file
        if os.path.exists(original_config_file):
            os.remove(original_config_file)
        
        # Save results
        result_data = {
            'experiment_config': asdict(exp_config),
            'model_config': asdict(model_config),
            'training_time_minutes': training_time,
            'timestamp': datetime.datetime.now().isoformat(),
            'gpu_count': torch.cuda.device_count(),
            'result': result
        }
        
        with open(f"{exp_dir}/results.json", 'w') as f:
            json.dump(result_data, f, indent=2)
        
        print(f"✅ Experiment completed in {training_time:.1f} minutes")
        print(f"📁 Results saved to: {exp_dir}")
        
        return result_data
    
    def create_temp_config_file(self, config: ModelConfig, filename: str):
        """Create a temporary config file to override the training script"""
        config_content = f"""
# Temporary model config for experiment
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    d_model: int = {config.d_model}
    n_heads: int = {config.n_heads}
    n_layers: int = {config.n_layers}
    d_ff: int = {config.d_ff}
    batch_size: int = {config.batch_size}
    max_steps: int = {config.max_steps}
    gradient_accumulation_steps: int = {config.gradient_accumulation_steps}
    muon_lr: float = {config.muon_lr}
    max_seq_len: int = {config.max_seq_len}
    num_documents: int = {config.num_documents}
    max_tokens: int = {config.max_tokens}
    eval_every: int = {config.eval_every}
    eval_steps: int = {config.eval_steps}
    weight_decay: float = {config.weight_decay}
    dropout: float = {config.dropout}
    grad_clip: float = {config.grad_clip}
    use_amp: bool = {config.use_amp}
    vocab_size: Optional[int] = {config.vocab_size}
    
    def __post_init__(self):
        self.d_k = self.d_model // self.n_heads
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
"""
        with open(filename, 'w') as f:
            f.write(config_content)
    
    def run_training(self, exp_config: ExperimentConfig, exp_dir: str) -> Dict[str, Any]:
        """Run the actual training based on the experiment configuration"""
        
        # Set environment variables for checkpointing
        os.environ["CHECKPOINT_DIR"] = f"{exp_dir}/checkpoints"
        
        try:
            if exp_config.training_method == "ddp":
                print("🔄 Using PyTorch DDP training...")
                success = self.run_ddp_training()
            elif exp_config.training_method == "custom":
                print("🔄 Using custom distributed training...")
                success = self.run_custom_training()
            elif exp_config.training_method == "single":
                print("🔄 Using single GPU training...")
                success = self.run_single_training()
            else:
                print("🔄 Using automatic training method selection...")
                success = self.run_auto_training()
            
            if success:
                return {"status": "success", "method": exp_config.training_method}
            else:
                return {"status": "failed", "method": exp_config.training_method, "error": "Training failed"}
                
        except Exception as e:
            return {"status": "error", "method": exp_config.training_method, "error": str(e)}
    
    def run_ddp_training(self) -> bool:
        """Run DDP training"""
        try:
            from train_distributed_llm import launch_distributed_ddp
            launch_distributed_ddp()
            return True
        except Exception as e:
            print(f"❌ DDP training failed: {e}")
            return False
    
    def run_custom_training(self) -> bool:
        """Run custom distributed training"""
        try:
            from train_distributed_llm import launch_distributed
            launch_distributed()
            return True
        except Exception as e:
            print(f"❌ Custom distributed training failed: {e}")
            return False
    
    def run_single_training(self) -> bool:
        """Run single GPU training"""
        try:
            from train_distributed_llm import run_training_direct
            run_training_direct()
            return True
        except Exception as e:
            print(f"❌ Single GPU training failed: {e}")
            return False
    
    def run_auto_training(self) -> bool:
        """Run automatic training method selection"""
        try:
            from train_distributed_llm import run_novita_4090_training
            run_novita_4090_training()
            return True
        except Exception as e:
            print(f"❌ Auto training failed: {e}")
            return False

def create_baseline_experiments() -> List[ExperimentConfig]:
    """Create baseline experiments to establish performance"""
    return [
        ExperimentConfig(
            name="baseline_small",
            description="Small model baseline with minimal training",
            hypothesis="Small model should train quickly and establish baseline performance",
            d_model=256,
            n_layers=4,
            n_heads=4,
            d_ff=1024,
            max_steps=1000,
            training_method="auto"
        ),
        ExperimentConfig(
            name="baseline_default",
            description="Default model configuration baseline",
            hypothesis="Default configuration should provide good balance of performance and efficiency",
            max_steps=2000,
            training_method="auto"
        ),
        ExperimentConfig(
            name="baseline_large",
            description="Larger model to test scaling",
            hypothesis="Larger model should achieve better performance with more training",
            d_model=512,
            n_layers=8,
            n_heads=8,
            d_ff=2048,
            max_steps=2000,
            training_method="auto"
        )
    ]

def create_architecture_experiments() -> List[ExperimentConfig]:
    """Create experiments to test different architectures"""
    return [
        ExperimentConfig(
            name="arch_wide",
            description="Wider model (more dimensions, same layers)",
            hypothesis="Wider model should capture more features per layer",
            d_model=512,
            n_layers=6,
            n_heads=8,
            d_ff=2048,
            max_steps=2000,
            training_method="auto"
        ),
        ExperimentConfig(
            name="arch_deep",
            description="Deeper model (more layers, same dimensions)",
            hypothesis="Deeper model should learn more complex patterns",
            d_model=384,
            n_layers=12,
            n_heads=8,
            d_ff=1536,
            max_steps=2000,
            training_method="auto"
        ),
        ExperimentConfig(
            name="arch_heads",
            description="More attention heads",
            hypothesis="More attention heads should capture diverse attention patterns",
            d_model=384,
            n_layers=6,
            n_heads=12,
            d_ff=1536,
            max_steps=2000,
            training_method="auto"
        )
    ]

def create_training_experiments() -> List[ExperimentConfig]:
    """Create experiments to test different training configurations"""
    return [
        ExperimentConfig(
            name="train_long",
            description="Extended training duration",
            hypothesis="More training steps should improve performance",
            max_steps=5000,
            training_method="auto"
        ),
        ExperimentConfig(
            name="train_high_lr",
            description="Higher learning rate",
            hypothesis="Higher learning rate might train faster initially",
            muon_lr=BASE_LR * NUM_GPUS * 2 if SCALE_LR_WITH_GPUS else BASE_LR * 2,
            max_steps=2000,
            training_method="auto"
        ),
        ExperimentConfig(
            name="train_low_lr",
            description="Lower learning rate",
            hypothesis="Lower learning rate might be more stable",
            muon_lr=BASE_LR * NUM_GPUS * 0.5 if SCALE_LR_WITH_GPUS else BASE_LR * 0.5,
            max_steps=2000,
            training_method="auto"
        ),
        ExperimentConfig(
            name="train_big_batch",
            description="Larger batch size",
            hypothesis="Larger batches might improve gradient estimates",
            batch_size=BASE_BATCH_SIZE * 2,
            gradient_accumulation_steps=2,
            max_steps=2000,
            training_method="auto"
        )
    ]

def create_data_experiments() -> List[ExperimentConfig]:
    """Create experiments to test different data configurations"""
    return [
        ExperimentConfig(
            name="data_long_seq",
            description="Longer sequence length",
            hypothesis="Longer sequences should improve context understanding",
            max_seq_len=1024,
            batch_size=BASE_BATCH_SIZE // 2,  # Reduce batch size for memory
            max_steps=2000,
            training_method="auto"
        ),
        ExperimentConfig(
            name="data_more_docs",
            description="More training documents",
            hypothesis="More diverse data should improve generalization",
            num_documents=6000,
            max_tokens=2000000,
            max_steps=2000,
            training_method="auto"
        ),
        ExperimentConfig(
            name="data_focused",
            description="Focused training on fewer documents",
            hypothesis="More repetition might improve learning efficiency",
            num_documents=1500,
            max_tokens=500000,
            max_steps=2000,
            training_method="auto"
        )
    ]

def main():
    """Main research experiment runner"""
    print("🔬 DISTRIBUTED LLM RESEARCH EXPERIMENTS")
    print("=" * 60)
    print("Based on train_distributed_llm.py structure")
    print()
    
    # Initialize runner
    runner = ResearchRunner()
    
    # Create all experiment sets
    all_experiments = []
    
    print("📋 Available experiment sets:")
    print("1. Baseline experiments (3 experiments)")
    print("2. Architecture experiments (3 experiments)")  
    print("3. Training experiments (4 experiments)")
    print("4. Data experiments (3 experiments)")
    print("5. All experiments (13 total)")
    print()
    
    choice = input("Select experiment set (1-5): ").strip()
    
    if choice == "1":
        all_experiments = create_baseline_experiments()
    elif choice == "2":
        all_experiments = create_architecture_experiments()
    elif choice == "3":
        all_experiments = create_training_experiments()
    elif choice == "4":
        all_experiments = create_data_experiments()
    elif choice == "5":
        all_experiments = (create_baseline_experiments() + 
                          create_architecture_experiments() + 
                          create_training_experiments() + 
                          create_data_experiments())
    else:
        print("Invalid choice, running baseline experiments")
        all_experiments = create_baseline_experiments()
    
    print(f"\n🚀 Running {len(all_experiments)} experiments...")
    
    results = []
    for i, exp_config in enumerate(all_experiments):
        print(f"\n{'='*60}")
        print(f"EXPERIMENT {i+1}/{len(all_experiments)}")
        print(f"{'='*60}")
        
        try:
            result = runner.run_experiment(exp_config)
            results.append(result)
        except Exception as e:
            print(f"❌ Experiment {exp_config.name} failed: {e}")
            continue
    
    # Summary
    print(f"\n📊 EXPERIMENT SUMMARY")
    print("=" * 40)
    print(f"Completed: {len(results)}/{len(all_experiments)} experiments")
    
    successful = [r for r in results if r['result']['status'] == 'success']
    failed = [r for r in results if r['result']['status'] != 'success']
    
    print(f"✅ Successful: {len(successful)}")
    print(f"❌ Failed: {len(failed)}")
    
    if successful:
        print(f"\n🎯 Successful experiments:")
        for result in successful:
            exp_name = result['experiment_config']['name']
            training_time = result['training_time_minutes']
            print(f"  {exp_name}: {training_time:.1f} minutes")
    
    if failed:
        print(f"\n💥 Failed experiments:")
        for result in failed:
            exp_name = result['experiment_config']['name']
            error = result['result'].get('error', 'Unknown error')
            print(f"  {exp_name}: {error}")
    
    print(f"\n📁 All results saved to: {runner.results_dir}/")

if __name__ == "__main__":
    main()