"""
Logging utilities for experiments
"""
import logging
import sys
from typing import Dict, Any
from configs.base_config import ExperimentConfig

def setup_logging(rank: int = 0) -> logging.Logger:
    """Setup logging for experiments"""
    logger = logging.getLogger(f'experiment_rank_{rank}')
    
    # Only setup logging for rank 0 in distributed training
    if rank == 0:
        logger.setLevel(logging.INFO)
        
        # Create console handler
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        
        # Add handler to logger
        if not logger.handlers:
            logger.addHandler(handler)
    else:
        # Disable logging for other ranks
        logger.setLevel(logging.CRITICAL)
    
    return logger

def log_experiment_start(logger: logging.Logger, config: ExperimentConfig):
    """Log experiment start information"""
    logger.info(f"🧪 STARTING EXPERIMENT: {config.name}")
    logger.info("=" * 60)
    logger.info(f"📝 Description: {config.description}")
    logger.info(f"💡 Hypothesis: {config.hypothesis}")
    logger.info(f"🏗️ Architecture: {config.d_model}d, {config.n_layers}L, {config.n_heads}H")
    logger.info(f"🔧 Attention: {config.attention_config.attention_type}")
    logger.info(f"📚 Training: {config.max_steps} steps, LR {config.learning_rate}")
    logger.info(f"📦 Batch: {config.batch_size}, Accum: {config.gradient_accumulation_steps}")
    logger.info(f"📏 Sequence: {config.max_seq_len} tokens")

def log_experiment_end(logger: logging.Logger, config: ExperimentConfig, result: Dict[str, Any]):
    """Log experiment completion information"""
    logger.info(f"✅ COMPLETED EXPERIMENT: {config.name}")
    logger.info("=" * 60)
    
    if result.get('success', False):
        training_time = result.get('training_time_minutes', 0)
        logger.info(f"⏱️ Training time: {training_time:.1f} minutes")
        
        # Log final metrics if available
        final_metrics = result.get('result', {}).get('final_metrics', {})
        if final_metrics:
            val_loss = final_metrics.get('val_loss', 0)
            val_acc = final_metrics.get('val_accuracy', 0)
            val_ppl = final_metrics.get('val_perplexity', 0)
            logger.info(f"📊 Final metrics: Loss={val_loss:.4f}, Acc={val_acc:.3f}, PPL={val_ppl:.2f}")
        
        model_params = result.get('result', {}).get('model_params', 0)
        if model_params:
            logger.info(f"📊 Model parameters: {model_params:,}")
    else:
        error = result.get('error', 'Unknown error')
        logger.info(f"❌ Experiment failed: {error}")
    
    logger.info("=" * 60)