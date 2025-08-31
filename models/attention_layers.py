"""
Attention layer implementations using Flash Linear Attention library
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from configs.base_config import ExperimentConfig

# Import from Flash Linear Attention library
try:
    from fla.layers import (
        GatedLinearAttention,
        MultiScaleRetention, 
        Mamba,
        Based,
        DeltaNet,
        HGRN,
        HGRN2,
        RWKV6,
        GSA
    )
    from fla.modules import (
        RMSNorm,
        FusedRMSNormGated,
        RotaryEmbedding
    )
    FLA_AVAILABLE = True
    print("✅ Flash Linear Attention library loaded successfully")
except ImportError as e:
    print(f"⚠️ Flash Linear Attention not available: {e}")
    print("📦 Install with: pip install flash-linear-attention")
    FLA_AVAILABLE = False

def get_attention_layer(config: ExperimentConfig):
    """Factory function to get attention layer based on config"""
    attention_type = config.attention_config.attention_type.lower()
    
    if not FLA_AVAILABLE and attention_type != "standard":
        print(f"⚠️ FLA not available, falling back to standard attention")
        attention_type = "standard"
    
    if attention_type == "standard":
        return StandardAttention(config)
    elif attention_type == "gla":
        return FLAGatedLinearAttention(config)
    elif attention_type == "retnet":
        return FLARetNet(config)
    elif attention_type == "mamba":
        return FLAMamba(config)
    elif attention_type == "based":
        return FLABased(config)
    elif attention_type == "deltanet":
        return FLADeltaNet(config)
    elif attention_type == "hgrn":
        return FLAHGRN(config)
    elif attention_type == "hgrn2":
        return FLAHGRN2(config)
    elif attention_type == "rwkv6":
        return FLRWKV6(config)
    elif attention_type == "gsa":
        return FLAGSA(config)
    else:
        raise ValueError(f"Unknown attention type: {attention_type}")

class StandardAttention(nn.Module):
    """Standard multi-head attention with rotary embeddings"""
    
    def __init__(self, config: ExperimentConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.d_k = config.d_model // config.n_heads
        
        # Linear projections
        self.qkv = nn.Linear(config.d_model, config.d_model * 3, bias=False)
        self.w_o = nn.Linear(config.d_model, config.d_model, bias=False)
        
        # Rotary embeddings
        if config.attention_config.use_rotary:
            if FLA_AVAILABLE:
                self.rotary = RotaryEmbedding(dim=self.d_k)
            else:
                self.rotary = SimpleRotary(self.d_k, config.max_seq_len)
        else:
            self.rotary = None
        
        self.dropout = config.dropout
    
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        batch_size, seq_len = x.size(0), x.size(1)
        
        # QKV projections
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.n_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, T, D]
        Q, K, V = qkv[0], qkv[1], qkv[2]
        
        # Apply rotary embeddings
        if self.rotary is not None:
            if FLA_AVAILABLE:
                Q = self.rotary(Q)
                K = self.rotary(K)
            else:
                Q = self.rotary(Q)
                K = self.rotary(K)
        
        # Scaled dot-product attention with Flash Attention if available
        if self.config.attention_config.use_flash_attention:
            attn_output = F.scaled_dot_product_attention(
                Q, K, V, is_causal=True, 
                dropout_p=self.dropout if self.training else 0.0
            )
        else:
            # Manual attention computation
            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
            
            # Causal mask
            mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1)
            scores = scores.masked_fill(mask.bool(), float('-inf'))
            
            attn_weights = F.softmax(scores, dim=-1)
            if self.training:
                attn_weights = F.dropout(attn_weights, p=self.dropout)
            
            attn_output = torch.matmul(attn_weights, V)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        return self.w_o(attn_output)

# FLA-based attention layers
class FLAGatedLinearAttention(nn.Module):
    """Gated Linear Attention using FLA implementation"""
    
    def __init__(self, config: ExperimentConfig):
        super().__init__()
        if not FLA_AVAILABLE:
            raise ImportError("Flash Linear Attention library required for GLA")
        
        self.gla = GatedLinearAttention(
            hidden_size=config.d_model,
            num_heads=config.n_heads,
            expand_k=config.attention_config.expand_k,
            expand_v=config.attention_config.expand_v,
            use_gk=config.attention_config.use_gk,
            use_gv=config.attention_config.use_gv,
            use_output_gate=config.attention_config.use_output_gate,
            mode='chunk',  # Use chunk mode for training efficiency
            fuse_norm=True,
            norm_eps=1e-6
        )
    
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        # FLA expects input in format [batch, seq_len, hidden_size]
        output, _ = self.gla(x)
        return output

class FLARetNet(nn.Module):
    """RetNet using FLA implementation"""
    
    def __init__(self, config: ExperimentConfig):
        super().__init__()
        if not FLA_AVAILABLE:
            raise ImportError("Flash Linear Attention library required for RetNet")
        
        self.retnet = MultiScaleRetention(
            hidden_size=config.d_model,
            num_heads=config.n_heads,
            mode='chunk',  # Use chunk mode for training
            fuse_norm=True,
            norm_eps=1e-5
        )
    
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        output, _ = self.retnet(x)
        return output

class FLAMamba(nn.Module):
    """Mamba using FLA implementation"""
    
    def __init__(self, config: ExperimentConfig):
        super().__init__()
        if not FLA_AVAILABLE:
            raise ImportError("Flash Linear Attention library required for Mamba")
        
        self.mamba = Mamba(
            d_model=config.d_model,
            d_state=config.attention_config.state_size,
            d_conv=config.attention_config.conv_kernel,
            expand=config.attention_config.expand
        )
    
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.mamba(x)

class FLABased(nn.Module):
    """Based attention using FLA implementation"""
    
    def __init__(self, config: ExperimentConfig):
        super().__init__()
        if not FLA_AVAILABLE:
            raise ImportError("Flash Linear Attention library required for Based")
        
        self.based = Based(
            hidden_size=config.d_model,
            num_heads=config.n_heads,
            mode='chunk',
            fuse_norm=True
        )
    
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        output, _ = self.based(x)
        return output

class FLADeltaNet(nn.Module):
    """DeltaNet using FLA implementation"""
    
    def __init__(self, config: ExperimentConfig):
        super().__init__()
        if not FLA_AVAILABLE:
            raise ImportError("Flash Linear Attention library required for DeltaNet")
        
        self.deltanet = DeltaNet(
            hidden_size=config.d_model,
            num_heads=config.n_heads,
            mode='chunk',
            fuse_norm=True
        )
    
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        output, _ = self.deltanet(x)
        return output

class FLAHGRN(nn.Module):
    """HGRN using FLA implementation"""
    
    def __init__(self, config: ExperimentConfig):
        super().__init__()
        if not FLA_AVAILABLE:
            raise ImportError("Flash Linear Attention library required for HGRN")
        
        self.hgrn = HGRN(
            hidden_size=config.d_model,
            num_heads=config.n_heads,
            mode='chunk',
            fuse_norm=True
        )
    
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        output, _ = self.hgrn(x)
        return output

class FLAHGRN2(nn.Module):
    """HGRN2 using FLA implementation"""
    
    def __init__(self, config: ExperimentConfig):
        super().__init__()
        if not FLA_AVAILABLE:
            raise ImportError("Flash Linear Attention library required for HGRN2")
        
        self.hgrn2 = HGRN2(
            hidden_size=config.d_model,
            num_heads=config.n_heads,
            mode='chunk',
            fuse_norm=True
        )
    
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        output, _ = self.hgrn2(x)
        return output

class FLRWKV6(nn.Module):
    """RWKV6 using FLA implementation"""
    
    def __init__(self, config: ExperimentConfig):
        super().__init__()
        if not FLA_AVAILABLE:
            raise ImportError("Flash Linear Attention library required for RWKV6")
        
        self.rwkv6 = RWKV6(
            hidden_size=config.d_model,
            num_heads=config.n_heads,
            mode='chunk',
            fuse_norm=True
        )
    
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        output, _ = self.rwkv6(x)
        return output

class FLAGSA(nn.Module):
    """GSA using FLA implementation"""
    
    def __init__(self, config: ExperimentConfig):
        super().__init__()
        if not FLA_AVAILABLE:
            raise ImportError("Flash Linear Attention library required for GSA")
        
        self.gsa = GSA(
            hidden_size=config.d_model,
            num_heads=config.n_heads,
            mode='chunk',
            fuse_norm=True
        )
    
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        output, _ = self.gsa(x)
        return output

# Fallback implementations when FLA is not available
class SimpleRotary(nn.Module):
    """Simple rotary position embedding fallback"""
    
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