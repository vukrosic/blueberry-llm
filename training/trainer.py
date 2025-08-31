"""
Main training loop implementation
"""
import os
import json
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
from typing import Dict, Any, List, Optional
import logging

from configs.base_config import ExperimentConfig
from .optimizers import create_optimizer, create_schedulers

class Trainer:
    """Main trainer class for LLM experiments"""
    
    def __init__(
        self,
        model: nn.Module,
        config: ExperimentConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
        tokenizer: AutoTokenizer,
        checkpoint_dir: str,
        use_distributed: bool = False,
        rank: int = 0,
        world_size: int = 1,
        logger: Optional[logging.Logger] = None
    ):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.tokenizer = tokenizer
        self.checkpoint_dir = checkpoint_dir
        self.use_distributed = use_distributed
        self.rank = rank
        self.world_size = world_size
        self.logger = logger or logging.getLogger(__name__)
        
        # Setup device
        if torch.cuda.is_available():
            self.device = torch.device(f'cuda:{rank}' if use_distributed else 'cuda')
        else:
            self.device = torch.device('cpu')
        
        self.model = self.model.to(self.device)
        
        # Setup distributed model
        if use_distributed and torch.cuda.is_available():
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, device_ids=[rank]
            )
        
        # Create optimizers and schedulers
        self.optimizers, self.optimizer_names = create_optimizer(self.model, config)
        self.schedulers = create_schedulers(self.optimizers, config)
        
        # Mixed precision scaler
        self.scaler = GradScaler() if config.use_amp else None
        
        # Training state
        self.step = 0
        self.best_val_loss = float('inf')
        
        # Create checkpoint directory
        if rank == 0:
            os.makedirs(checkpoint_dir, exist_ok=True)
        
        if rank == 0:
            self.logger.info(f"🔧 Trainer initialized")
            self.logger.info(f"📱 Device: {self.device}")
            self.logger.info(f"🔧 Optimizers: {', '.join(self.optimizer_names)}")
            self.logger.info(f"📊 Model parameters: {self._count_parameters():,}")
    
    def _count_parameters(self) -> int:
        """Count total model parameters"""
        if hasattr(self.model, 'module'):
            return sum(p.numel() for p in self.model.module.parameters())
        else:
            return sum(p.numel() for p in self.model.parameters())
    
    def train(self) -> Dict[str, Any]:
        """Main training loop"""
        if self.rank == 0:
            self.logger.info(f"🚀 Starting training for {self.config.max_steps} steps")
        
        self.model.train()
        start_time = time.time()
        
        # Progress bar (only on rank 0)
        pbar = None
        if self.rank == 0:
            pbar = tqdm(total=self.config.max_steps, desc="Training")
        
        # Training loop
        while self.step < self.config.max_steps:
            # Set epoch for distributed sampler
            if hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(self.step // len(self.train_loader))
            
            for batch_idx, (x, y) in enumerate(self.train_loader):
                if self.step >= self.config.max_steps:
                    break
                
                # Move data to device
                x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
                
                # Forward pass with gradient accumulation
                loss = self._forward_step(x, y)
                
                # Backward pass
                self._backward_step(loss)
                
                # Optimizer step
                if (self.step + 1) % self.config.gradient_accumulation_steps == 0:
                    self._optimizer_step()
                
                # Logging
                if self.step % 100 == 0:
                    self._log_step(x, y, loss, pbar)
                
                # Evaluation
                if self.step % self.config.eval_every == 0 and self.step > 0:
                    self._evaluate()
                
                # Checkpointing
                if self.step % self.config.save_every == 0 and self.step > 0:
                    self._save_checkpoint()
                
                self.step += 1
                
                # Update progress bar
                if pbar and self.step % 100 == 0:
                    pbar.update(100)
        
        if pbar:
            pbar.close()
        
        # Final evaluation
        final_metrics = self._evaluate()
        
        # Save final checkpoint
        if self.rank == 0:
            self._save_checkpoint(is_final=True)
        
        training_time = time.time() - start_time
        
        if self.rank == 0:
            self.logger.info(f"✅ Training completed in {training_time/60:.1f} minutes")
        
        return final_metrics
    
    def _forward_step(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional mixed precision"""
        if self.config.use_amp:
            with autocast():
                logits = self.model(x)
                loss = F.cross_entropy(logits.view(-1, self.config.vocab_size), y.view(-1))
                loss = loss / self.config.gradient_accumulation_steps
        else:
            logits = self.model(x)
            loss = F.cross_entropy(logits.view(-1, self.config.vocab_size), y.view(-1))
            loss = loss / self.config.gradient_accumulation_steps
        
        return loss
    
    def _backward_step(self, loss: torch.Tensor):
        """Backward pass with optional mixed precision"""
        if self.config.use_amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
    
    def _optimizer_step(self):
        """Optimizer step with gradient clipping"""
        if self.config.use_amp:
            # Unscale gradients
            for optimizer in self.optimizers:
                self.scaler.unscale_(optimizer)
            
            # Clip gradients
            if hasattr(self.model, 'module'):
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.module.parameters(), self.config.grad_clip)
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            
            # Optimizer step
            for optimizer in self.optimizers:
                self.scaler.step(optimizer)
                optimizer.zero_grad()
            
            # Update scaler
            self.scaler.update()
        else:
            # Clip gradients
            if hasattr(self.model, 'module'):
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.module.parameters(), self.config.grad_clip)
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            
            # Optimizer step
            for optimizer in self.optimizers:
                optimizer.step()
                optimizer.zero_grad()
        
        # Scheduler step
        for scheduler in self.schedulers:
            scheduler.step()
    
    def _log_step(self, x: torch.Tensor, y: torch.Tensor, loss: torch.Tensor, pbar: Optional[tqdm]):
        """Log training step metrics"""
        if self.rank != 0:
            return
        
        with torch.no_grad():
            # Calculate metrics
            logits = self.model(x)
            predictions = logits.argmax(dim=-1)
            accuracy = (predictions == y).float().mean().item()
            current_loss = loss.item() * self.config.gradient_accumulation_steps
            perplexity = math.exp(min(current_loss, 20))
            
            # Get learning rate
            lr = self.optimizers[0].param_groups[0]['lr']
            
            # Update progress bar
            if pbar:
                pbar.set_postfix({
                    'loss': f'{current_loss:.4f}',
                    'acc': f'{accuracy:.3f}',
                    'ppl': f'{perplexity:.1f}',
                    'lr': f'{lr:.2e}'
                })
    
    def _evaluate(self) -> Dict[str, Any]:
        """Evaluate model on validation set"""
        self.model.eval()
        total_loss = 0
        total_tokens = 0
        total_correct = 0
        
        with torch.no_grad():
            for i, (x, y) in enumerate(self.val_loader):
                if i >= self.config.eval_steps:
                    break
                
                x, y = x.to(self.device), y.to(self.device)
                
                if self.config.use_amp:
                    with autocast():
                        logits = self.model(x)
                        loss = F.cross_entropy(logits.view(-1, self.config.vocab_size), y.view(-1))
                else:
                    logits = self.model(x)
                    loss = F.cross_entropy(logits.view(-1, self.config.vocab_size), y.view(-1))
                
                total_loss += loss.item() * y.numel()
                total_tokens += y.numel()
                
                predictions = logits.argmax(dim=-1)
                total_correct += (predictions == y).sum().item()
        
        # Reduce across all GPUs if distributed
        if self.use_distributed:
            total_loss_tensor = torch.tensor(total_loss, device=self.device)
            total_tokens_tensor = torch.tensor(total_tokens, device=self.device)
            total_correct_tensor = torch.tensor(total_correct, device=self.device)
            
            dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_tokens_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_correct_tensor, op=dist.ReduceOp.SUM)
            
            total_loss = total_loss_tensor.item()
            total_tokens = total_tokens_tensor.item()
            total_correct = total_correct_tensor.item()
        
        # Calculate metrics
        avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
        accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0
        perplexity = math.exp(min(avg_loss, 20))
        
        metrics = {
            'val_loss': avg_loss,
            'val_accuracy': accuracy,
            'val_perplexity': perplexity,
            'step': self.step
        }
        
        # Log metrics
        if self.rank == 0:
            self.logger.info(f"Step {self.step}: Val Loss: {avg_loss:.4f}, "
                           f"Val Acc: {accuracy:.4f}, Val PPL: {perplexity:.2f}")
        
        # Update best loss
        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
        
        self.model.train()
        return metrics
    
    def _save_checkpoint(self, is_final: bool = False):
        """Save model checkpoint"""
        if self.rank != 0:
            return
        
        # Create checkpoint data
        model_state = self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict()
        
        checkpoint_data = {
            'step': self.step,
            'model_state_dict': model_state,
            'optimizer_states': [opt.state_dict() for opt in self.optimizers],
            'scheduler_states': [sched.state_dict() for sched in self.schedulers],
            'best_val_loss': self.best_val_loss,
            'config': self.config.__dict__
        }
        
        # Save checkpoint
        if is_final:
            checkpoint_path = os.path.join(self.checkpoint_dir, 'final_model.pt')
        else:
            checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_step_{self.step}.pt')
        
        torch.save(checkpoint_data, checkpoint_path)
        
        # Save config as JSON
        config_path = os.path.join(self.checkpoint_dir, 'config.json')
        config_dict = {
            'd_model': self.config.d_model,
            'n_heads': self.config.n_heads,
            'n_layers': self.config.n_layers,
            'd_ff': self.config.d_ff,
            'vocab_size': self.config.vocab_size,
            'max_seq_len': self.config.max_seq_len,
            'dropout': self.config.dropout,
            'attention_type': self.config.attention_config.attention_type
        }
        
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(self.checkpoint_dir)
        
        self.logger.info(f"💾 Checkpoint saved: {checkpoint_path}")