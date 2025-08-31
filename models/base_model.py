"""
Base transformer model with pluggable attention mechanisms
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, Any
from configs.base_config import ExperimentConfig
from .attention_layers import get_attention_layer

# Try to import FLA modules
try:
    from fla.modules import RMSNorm, FusedRMSNormGated
    FLA_AVAILABLE = True
except ImportError:
    FLA_AVAILABLE = False

class BaseTransformer(nn.Module):
    """Base transformer model with configurable attention mechanisms"""
    
    def __init__(self, config: ExperimentConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_dropout = nn.Dropout(config.dropout)
        
        # Transformer blocks with configurable attention
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        
        # Output layers - use FLA if available
        if FLA_AVAILABLE:
            self.norm = RMSNorm(config.d_model)
        else:
            self.norm = nn.RMSNorm(config.d_model)
        self.output_dropout = nn.Dropout(config.dropout)
        
        # Language modeling head (tied weights)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize model weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass"""
        # Token embeddings with scaling
        x = self.token_embedding(x) * math.sqrt(self.config.d_model)
        x = self.position_dropout(x)
        
        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(x, **kwargs)
        
        # Output processing
        x = self.norm(x)
        x = self.output_dropout(x)
        logits = self.lm_head(x)
        
        return logits
    
    def get_num_params(self) -> int:
        """Get total number of parameters"""
        return sum(p.numel() for p in self.parameters())

class TransformerBlock(nn.Module):
    """Transformer block with configurable attention mechanism"""
    
    def __init__(self, config: ExperimentConfig):
        super().__init__()
        self.config = config
        
        # Get attention layer based on config
        self.attention = get_attention_layer(config)
        
        # Feed-forward network
        self.feed_forward = FeedForward(config.d_model, config.d_ff, config.dropout)
        
        # Layer normalization - use FLA if available
        if FLA_AVAILABLE:
            self.norm1 = RMSNorm(config.d_model)
            self.norm2 = RMSNorm(config.d_model)
        else:
            self.norm1 = nn.RMSNorm(config.d_model)
            self.norm2 = nn.RMSNorm(config.d_model)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass with residual connections"""
        # Attention block
        attn_out = self.attention(self.norm1(x), **kwargs)
        x = x + self.dropout(attn_out)
        
        # Feed-forward block
        ff_out = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_out)
        
        return x

class FeedForward(nn.Module):
    """Feed-forward network with SiLU activation"""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff, bias=False)
        self.linear2 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.silu(self.linear1(x))))

class Rotary(nn.Module):
    """Rotary Position Embedding"""
    
    def __init__(self, dim: int, max_seq_len: int):
        super().__init__()
        angular_freq = (1 / 10000) ** torch.linspace(0, 1, steps=dim//4, dtype=torch.float32)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(dim//4)])
        t = torch.arange(max_seq_len, dtype=torch.float32)
        theta = torch.einsum("i,j -> ij", t, angular_freq)
        self.register_buffer('cos', theta.cos(), persistent=False)
        self.register_buffer('sin', theta.sin(), persistent=False)

    def forward(self, x_BTHD: torch.Tensor):
        assert self.cos.size(0) >= x_BTHD.size(-3)
        cos, sin = self.cos[None, :x_BTHD.size(-3), None, :], self.sin[None, :x_BTHD.size(-3), None, :]
        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x_BTHD)