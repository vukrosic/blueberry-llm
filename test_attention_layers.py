#!/usr/bin/env python3
"""
Test script to verify attention layer implementations work correctly
"""
import torch
from configs.base_config import ExperimentConfig
from models.attention_layers import get_attention_layer

def test_attention_layers():
    """Test different attention layer implementations"""
    print("🧪 TESTING ATTENTION LAYER IMPLEMENTATIONS")
    print("=" * 50)
    
    if not torch.cuda.is_available():
        print("❌ CUDA required for FLA testing")
        return
    
    device = torch.device('cuda')
    # Create base config first to get the correct dimensions
    config = ExperimentConfig()
    x = torch.randn(2, 32, config.d_model, device=device)
    
    # Config already created above
    
    # Test different attention types
    attention_types = ['gla', 'retnet', 'deltanet', 'based']
    
    for attention_type in attention_types:
        print(f"\n🔍 Testing {attention_type.upper()} attention layer...")
        
        try:
            # Set attention type
            config.attention_config.attention_type = attention_type
            
            # Create attention layer
            attention_layer = get_attention_layer(config).to(device)
            
            # Forward pass
            output = attention_layer(x)
            
            # Check output
            print(f"✅ {attention_type.upper()} successful!")
            print(f"   Input shape:  {x.shape}")
            print(f"   Output shape: {output.shape}")
            print(f"   Output type:  {type(output)}")
            
            # Verify output is a tensor (not tuple)
            if isinstance(output, torch.Tensor):
                print(f"   ✅ Returns single tensor (correct)")
            else:
                print(f"   ❌ Returns {type(output)} (should be tensor)")
                
        except Exception as e:
            print(f"❌ {attention_type.upper()} failed: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n🎉 Attention layer testing complete!")

if __name__ == "__main__":
    test_attention_layers()