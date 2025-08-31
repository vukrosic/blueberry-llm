"""
Utilities for handling FLA layer outputs
"""
import torch
from typing import Union, Tuple

def handle_fla_output(result: Union[torch.Tensor, Tuple]) -> torch.Tensor:
    """
    Handle different FLA layer output formats
    
    Some FLA layers return (output, state), others return just output
    This function normalizes the return to just the output tensor
    """
    if isinstance(result, tuple):
        # Most FLA layers return (output, state) or (output, cache)
        return result[0]
    else:
        # Some layers return just the output tensor
        return result

def safe_fla_forward(layer, x: torch.Tensor, **kwargs) -> torch.Tensor:
    """
    Safely call an FLA layer and handle its output format
    """
    result = layer(x, **kwargs)
    return handle_fla_output(result)