"""
Attention layer implementations using Flash Linear Attention library
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from configs.base_config import ExperimentConfig
# Removed safe_fla_forward import - handling FLA returns directly in each layer

# Import from Flash Linear Attention library - REQUIRED
try:
    from fla.layers import (
        GatedLinearAttention,
        MultiScaleRetention, 
        Mamba,
        BasedLinearAttention,
        DeltaNet,
        HGRNAttention,
        HGRN2Attention,
        RWKV6Attention,
        GatedSlotAttention,
        PaTHAttention
    )
    from fla.modules import (
        RMSNorm,
        FusedRMSNormGated,
        RotaryEmbedding
    )
    print("✅ Flash Linear Attention library loaded successfully")
except ImportError as e:
    raise ImportError(
        f"Flash Linear Attention library is required but not available: {e}\n"
        f"Install with: pip install flash-linear-attention\n"
        f"Or run: python setup_fla.py"
    )

def get_attention_layer(config: ExperimentConfig):
    """Factory function to get attention layer based on config"""
    attention_type = config.attention_config.attention_type.lower()
    
    if attention_type == "standard":
        return StandardAttention(config)
    elif attention_type == "simple_linear":
        return SimpleLinearAttention(config)
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
    elif attention_type == "path":
        return FLAPaTH(config)
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
            self.rotary = RotaryEmbedding(dim=self.d_k)
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

class SimpleLinearAttention(nn.Module):
    """Simple linear attention implementation without FLA dependencies"""
    
    def __init__(self, config: ExperimentConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.d_k = config.d_model // config.n_heads
        
        # Linear projections
        self.qkv = nn.Linear(config.d_model, config.d_model * 3, bias=False)
        self.w_o = nn.Linear(config.d_model, config.d_model, bias=False)
        
        self.dropout = config.dropout
    
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        batch_size, seq_len = x.size(0), x.size(1)
        
        # QKV projections
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.n_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, T, D]
        Q, K, V = qkv[0], qkv[1], qkv[2]
        
        # Apply feature map (ELU + 1 for positivity)
        Q = F.elu(Q) + 1
        K = F.elu(K) + 1
        
        # Linear attention computation
        # Compute K^T V first (more efficient)
        KV = torch.einsum('bhnd,bhnf->bhdf', K, V)  # [B, H, D, F]
        
        # Compute normalizer
        K_sum = K.sum(dim=2, keepdim=True)  # [B, H, 1, D]
        
        # Compute output
        numerator = torch.einsum('bhnd,bhdf->bhnf', Q, KV)  # [B, H, N, F]
        denominator = torch.einsum('bhnd,bhnd->bhn', Q, K_sum.expand_as(Q))  # [B, H, N]
        
        # Avoid division by zero
        denominator = denominator.unsqueeze(-1) + 1e-6
        attn_output = numerator / denominator
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        return self.w_o(attn_output)

# FLA-based attention layers
class FLAGatedLinearAttention(nn.Module):
    """Gated Linear Attention using FLA implementation"""
    
    def __init__(self, config: ExperimentConfig):
        super().__init__()
        self.gla = GatedLinearAttention(
            mode='chunk',
            hidden_size=config.d_model,
            num_heads=config.n_heads,
            expand_k=0.5,
            expand_v=1.0,
            use_output_gate=True,
            fuse_norm=True,
            layer_idx=0  # Add layer_idx parameter
        )
    
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        # FLA expects input in format [batch, seq_len, hidden_size]
        # Remove any kwargs that might cause issues
        clean_kwargs = {}
        result = self.gla(x, **clean_kwargs)
        # FLA layers return (output, attention_weights, past_key_values)
        if isinstance(result, tuple):
            return result[0]
        return result

class FLARetNet(nn.Module):
    """RetNet using FLA implementation"""
    
    def __init__(self, config: ExperimentConfig):
        super().__init__()
        # Ensure head_k_dim <= 256 for RetNet
        head_k_dim = config.d_model // config.n_heads
        if head_k_dim > 256:
            # Adjust num_heads to keep head_k_dim <= 256
            num_heads = (config.d_model + 255) // 256  # Round up division
        else:
            num_heads = config.n_heads
            
        self.retnet = MultiScaleRetention(
            mode='chunk',
            hidden_size=config.d_model,
            num_heads=num_heads,
            expand_k=1.0,
            expand_v=2.0,
            use_output_gate=True,
            fuse_norm=True,
            layer_idx=0  # Add layer_idx parameter
        )
    
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        # Remove any kwargs that might cause issues
        clean_kwargs = {}
        result = self.retnet(x, **clean_kwargs)
        # FLA layers return (output, attention_weights, past_key_values)
        if isinstance(result, tuple):
            return result[0]
        return result

class FLAMamba(nn.Module):
    """Mamba using FLA implementation"""
    
    def __init__(self, config: ExperimentConfig):
        super().__init__()
        self.mamba = Mamba(
            d_model=config.d_model
        )
    
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        result = self.mamba(x, **kwargs)
        # FLA layers return (output, cache, extra_info) - we only need the output
        if isinstance(result, tuple):
            return result[0]
        return result

class FLABased(nn.Module):
    """Based attention using FLA implementation"""
    
    def __init__(self, config: ExperimentConfig):
        super().__init__()
        # For BasedLinearAttention, we need to be careful with dimensions
        # The feature_dim should be small (typically 16) and num_heads should match
        # Let's use standard values that work well
        self.based = BasedLinearAttention(
            hidden_size=config.d_model,
            num_heads=config.n_heads,
            num_key_value_heads=config.n_heads,
            feature_dim=16,  # Use standard feature_dim
            mode='chunk'
        )
    
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        result = self.based(x, **kwargs)
        # FLA layers return (output, cache, extra_info) - we only need the output
        if isinstance(result, tuple):
            return result[0]
        return result

class FLADeltaNet(nn.Module):
    """DeltaNet using FLA implementation"""
    
    def __init__(self, config: ExperimentConfig):
        super().__init__()
        self.deltanet = DeltaNet(
            mode='chunk',
            hidden_size=config.d_model,
            num_heads=config.n_heads,
            expand_k=1.0,
            expand_v=1.0,
            use_beta=True,
            use_gate=False,
            use_short_conv=True
        )
    
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        result = self.deltanet(x, **kwargs)
        # FLA layers return (output, cache, extra_info) - we only need the output
        if isinstance(result, tuple):
            return result[0]
        return result

class FLAHGRN(nn.Module):
    """HGRN using FLA implementation"""
    
    def __init__(self, config: ExperimentConfig):
        super().__init__()
        self.hgrn = HGRNAttention(
            hidden_size=config.d_model,
            num_heads=config.n_heads,
            mode='chunk'
        )
    
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        result = self.hgrn(x, **kwargs)
        # FLA layers return (output, cache, extra_info) - we only need the output
        if isinstance(result, tuple):
            return result[0]
        return result

class FLAHGRN2(nn.Module):
    """HGRN2 using FLA implementation"""
    
    def __init__(self, config: ExperimentConfig):
        super().__init__()
        self.hgrn2 = HGRN2Attention(
            hidden_size=config.d_model,
            num_heads=config.n_heads,
            mode='chunk'
        )
    
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        result = self.hgrn2(x, **kwargs)
        # FLA layers return (output, cache, extra_info) - we only need the output
        if isinstance(result, tuple):
            return result[0]
        return result

class FLRWKV6(nn.Module):
    """RWKV6 using FLA implementation"""
    
    def __init__(self, config: ExperimentConfig):
        super().__init__()
        self.rwkv6 = RWKV6Attention(
            hidden_size=config.d_model,
            num_heads=config.n_heads,
            mode='chunk'
        )
    
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        result = self.rwkv6(x, **kwargs)
        # FLA layers return (output, cache, extra_info) - we only need the output
        if isinstance(result, tuple):
            return result[0]
        return result

class FLAGSA(nn.Module):
    """GSA using FLA implementation"""
    
    def __init__(self, config: ExperimentConfig):
        super().__init__()
        self.gsa = GatedSlotAttention(
            hidden_size=config.d_model,
            num_heads=config.n_heads,
            mode='chunk'
        )
    
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        result = self.gsa(x, **kwargs)
        # FLA layers return (output, cache, extra_info) - we only need the output
        if isinstance(result, tuple):
            return result[0]
        return result


class FLAPaTH(nn.Module):
    """PaTH Attention using FLA implementation"""
    
    def __init__(self, config: ExperimentConfig):
        super().__init__()
        # PaTH attention only supports head_dim in [16, 32, 64, 128]
        # Current config: d_model=384, n_heads=8 -> head_dim=48 (not supported)
        # Let's use num_heads=6 to get head_dim=64 (supported)
        supported_head_dims = [16, 32, 64, 128]
        original_head_dim = config.d_model // config.n_heads
        
        # Find the closest supported head_dim
        if original_head_dim in supported_head_dims:
            num_heads = config.n_heads
        else:
            # Try to find a divisor of d_model that gives a supported head_dim
            num_heads = None
            for head_dim in supported_head_dims:
                if config.d_model % head_dim == 0:
                    num_heads = config.d_model // head_dim
                    break
            
            if num_heads is None:
                # Fallback: use head_dim=64 and adjust hidden_size
                head_dim = 64
                num_heads = config.d_model // head_dim
                if config.d_model % head_dim != 0:
                    num_heads = 6  # 384 // 64 = 6
        
        self.path_attn = PaTHAttention(
            hidden_size=config.d_model,
            num_heads=num_heads,
            num_kv_heads=num_heads,
            use_forget_gate=False,
            use_qk_norm=False,
            use_low_rank_w=True,
            use_w_shortconv=True,
            conv_size=3,
            conv_bias=False
        )
    
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        result = self.path_attn(x, **kwargs)
        # PaTH attention returns (output, attn_weights, past_key_values) - we only need the output
        if isinstance(result, tuple):
            return result[0]
        return result

