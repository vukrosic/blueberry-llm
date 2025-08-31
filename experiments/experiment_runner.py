"""
Experiment runner for systematic research studies
"""
import os
import json
import time
import torch
import torch.distributed as dist
from typing import Dict, Any, List, Optional
from dataclasses import asdict
import datetime
import hashlib

from configs.base_config import ExperimentConfig
from models.base_model import BaseTransformer
from training.trainer import Trainer
from training.data_utils import create_dataloaders
from utils.logging import setup_logging, log_experiment_start, log_experiment_end

class ExperimentRunner:
    """Runs systematic experiments with proper logging and checkpointing"""
    
    def __init__(self, results_dir: str = "results", use_distributed: bool = None):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
        # Auto-detect distributed training
        if use_distributed is None:
            use_distributed = torch.cuda.device_count() > 1
        
        self.use_distributed = use_distributed
        self.rank = 0
        self.world_size = 1
        
        if use_distributed:
            self._setup_distributed()
        
        # Setup logging
        self.logger = setup_logging(self.rank)
        
        if self.rank == 0:
            self.logger.info(f"🔬 ExperimentRunner initialized")
            self.logger.info(f"📁 Results directory: {results_dir}")
            self.logger.info(f"🌐 Distributed: {use_distributed}")
            if use_distributed:
                self.logger.info(f"🔧 World size: {self.world_size}")
    
    def _setup_distributed(self):
        """Setup distributed training if available"""
        try:
            if not dist.is_initialized():
                dist.init_process_group(backend='nccl')
            
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            
            # Set device for this process
            torch.cuda.set_device(self.rank)
            
        except Exception as e:
            print(f"⚠️ Distributed setup failed: {e}")
            self.use_distributed = False
            self.rank = 0
            self.world_size = 1
    
    def run_experiment(self, config: ExperimentConfig) -> Dict[str, Any]:
        """Run a single experiment"""
        if self.rank == 0:
            log_experiment_start(self.logger, config)
        
        # Create experiment directory
        exp_hash = self._get_config_hash(config)
        exp_dir = os.path.join(self.results_dir, f"{config.name}_{exp_hash}")
        
        if self.rank == 0:
            os.makedirs(exp_dir, exist_ok=True)
            
            # Save experiment config
            config_path = os.path.join(exp_dir, "experiment_config.json")
            with open(config_path, 'w') as f:
                json.dump(asdict(config), f, indent=2)
        
        # Synchronize all processes
        if self.use_distributed:
            dist.barrier()
        
        try:
            # Run the actual training
            start_time = time.time()
            result = self._run_training(config, exp_dir)
            training_time = time.time() - start_time
            
            # Collect results
            experiment_result = {
                'config': asdict(config),
                'result': result,
                'training_time_minutes': training_time / 60,
                'timestamp': datetime.datetime.now().isoformat(),
                'world_size': self.world_size,
                'success': result.get('success', False)
            }
            
            # Save results (only rank 0)
            if self.rank == 0:
                results_path = os.path.join(exp_dir, "results.json")
                with open(results_path, 'w') as f:
                    json.dump(experiment_result, f, indent=2)
                
                log_experiment_end(self.logger, config, experiment_result)
            
            return experiment_result
            
        except Exception as e:
            error_result = {
                'config': asdict(config),
                'error': str(e),
                'timestamp': datetime.datetime.now().isoformat(),
                'success': False
            }
            
            if self.rank == 0:
                self.logger.error(f"❌ Experiment {config.name} failed: {e}")
                
                # Save error result
                results_path = os.path.join(exp_dir, "results.json")
                with open(results_path, 'w') as f:
                    json.dump(error_result, f, indent=2)
            
            return error_result
    
    def _run_training(self, config: ExperimentConfig, exp_dir: str) -> Dict[str, Any]:
        """Run the actual training process"""
        # Set checkpoint directory
        checkpoint_dir = os.path.join(exp_dir, "checkpoints")
        
        # Create model
        model = BaseTransformer(config)
        
        if self.rank == 0:
            self.logger.info(f"📊 Model parameters: {model.get_num_params():,}")
        
        # Create data loaders
        train_loader, val_loader, tokenizer = create_dataloaders(
            config, rank=self.rank, world_size=self.world_size
        )
        
        # Update vocab size in config
        config.vocab_size = tokenizer.vocab_size
        
        # Create trainer
        trainer = Trainer(
            model=model,
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            tokenizer=tokenizer,
            checkpoint_dir=checkpoint_dir,
            use_distributed=self.use_distributed,
            rank=self.rank,
            world_size=self.world_size,
            logger=self.logger
        )
        
        # Train the model
        final_metrics = trainer.train()
        
        return {
            'success': True,
            'final_metrics': final_metrics,
            'model_params': model.get_num_params(),
            'checkpoint_dir': checkpoint_dir
        }
    
    def run_experiment_suite(self, experiments: List[ExperimentConfig]) -> List[Dict[str, Any]]:
        """Run a suite of experiments"""
        if self.rank == 0:
            self.logger.info(f"🚀 Starting experiment suite with {len(experiments)} experiments")
        
        results = []
        
        for i, config in enumerate(experiments):
            if self.rank == 0:
                self.logger.info(f"\n{'='*60}")
                self.logger.info(f"EXPERIMENT {i+1}/{len(experiments)}: {config.name}")
                self.logger.info(f"{'='*60}")
            
            try:
                result = self.run_experiment(config)
                results.append(result)
                
                # Brief summary
                if self.rank == 0:
                    status = "✅ SUCCESS" if result.get('success', False) else "❌ FAILED"
                    time_str = f"{result.get('training_time_minutes', 0):.1f}min"
                    self.logger.info(f"{status} - {config.name} ({time_str})")
                
            except Exception as e:
                if self.rank == 0:
                    self.logger.error(f"❌ Experiment {config.name} crashed: {e}")
                continue
        
        # Save combined results
        if self.rank == 0:
            combined_path = os.path.join(self.results_dir, "experiment_suite_results.json")
            with open(combined_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            self._print_suite_summary(results)
        
        return results
    
    def _print_suite_summary(self, results: List[Dict[str, Any]]):
        """Print summary of experiment suite"""
        self.logger.info(f"\n📊 EXPERIMENT SUITE SUMMARY")
        self.logger.info("=" * 60)
        
        successful = [r for r in results if r.get('success', False)]
        failed = [r for r in results if not r.get('success', False)]
        
        self.logger.info(f"Total experiments: {len(results)}")
        self.logger.info(f"✅ Successful: {len(successful)}")
        self.logger.info(f"❌ Failed: {len(failed)}")
        
        if successful:
            total_time = sum(r.get('training_time_minutes', 0) for r in successful)
            self.logger.info(f"⏱️ Total training time: {total_time:.1f} minutes")
            
            self.logger.info(f"\n🎯 Successful experiments:")
            for result in successful:
                config_name = result['config']['name']
                time_min = result.get('training_time_minutes', 0)
                
                # Extract key metrics if available
                metrics = result.get('result', {}).get('final_metrics', {})
                if metrics:
                    val_loss = metrics.get('val_loss', 0)
                    val_acc = metrics.get('val_accuracy', 0)
                    self.logger.info(f"  {config_name}: {time_min:.1f}min, loss={val_loss:.4f}, acc={val_acc:.3f}")
                else:
                    self.logger.info(f"  {config_name}: {time_min:.1f}min")
        
        if failed:
            self.logger.info(f"\n💥 Failed experiments:")
            for result in failed:
                config_name = result['config']['name']
                error = result.get('error', 'Unknown error')
                self.logger.info(f"  {config_name}: {error}")
    
    def _get_config_hash(self, config: ExperimentConfig) -> str:
        """Get unique hash for experiment configuration"""
        config_str = json.dumps(asdict(config), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]
    
    def cleanup(self):
        """Cleanup distributed resources"""
        if self.use_distributed and dist.is_initialized():
            dist.destroy_process_group()