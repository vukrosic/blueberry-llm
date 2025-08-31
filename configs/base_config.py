"""
Base configuration for LLM experiments
"""
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

@dataclass
class BaseConfig:
    """Base configuration class for all experiments"""
    # Model architecture
    d_model: int = 384
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 1536
    vocab_size: Optional[int] = None
    max_seq_len: int = 512
    dropout: float = 0.1
    
    # Training parameters
    batch_size: int = 16
    max_steps: int = 10000
    gradient_accumulation_steps: int = 4
    learning_rate: float = 0.01
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    
    # Data parameters
    num_documents: int = 3000
    max_tokens: int = 1000000
    
    # Evaluation
    eval_every: int = 500
    eval_steps: int = 100
    
    # Technical
    use_amp: bool = True
    seed: int = 42
    
    def __post_init__(self):
        self.d_k = self.d_model // self.n_heads
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"

@dataclass 
class AttentionConfig:
    """Configuration for attention mechanisms"""
    attention_type: str = "standard"  # standard, gla, retnet, mamba, etc.
    use_rotary: bool = True
    use_flash_attention: bool = True
    
    # GLA specific
    expand_k: float = 0.5
    expand_v: float = 1.0
    use_gk: bool = True
    use_gv: bool = False
    use_output_gate: bool = True
    
    # RetNet specific
    use_decay: bool = True
    
    # Mamba specific
    state_size: int = 16
    conv_kernel: int = 4
    expand: int = 2

@dataclass
class ExperimentConfig(BaseConfig):
    """Extended configuration for experiments"""
    # Experiment metadata
    name: str = "baseline"
    description: str = "Baseline experiment"
    hypothesis: str = "Standard transformer baseline"
    
    # Architecture selection
    attention_config: AttentionConfig = None
    
    # Training method
    training_method: str = "auto"  # auto, ddp, single
    
    # Checkpointing
    save_every: int = 5000
    checkpoint_dir: str = "checkpoints"
    
    def __post_init__(self):
        super().__post_init__()
        if self.attention_config is None:
            self.attention_config = AttentionConfig()