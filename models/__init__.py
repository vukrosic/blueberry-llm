"""
Model implementations for LLM research experiments
"""

from .base_model import BaseTransformer
from .attention_layers import (
    StandardAttention,
    FLAGatedLinearAttention,
    FLARetNet,
    FLAMamba,
    get_attention_layer
)

__all__ = [
    'BaseTransformer',
    'StandardAttention',
    'FLAGatedLinearAttention',
    'FLARetNet',
    'FLAMamba',
    'get_attention_layer'
]