#!/usr/bin/env python3
"""
Debug script to check what FLA layers actually return
"""
import torch

def test_fla_returns():
    """Test what different FLA layers return"""
    print("🔍 DEBUGGING FLA LAYER RETURNS")
    print("=" * 40)
    
    if not torch.cuda.is_available():
        print("❌ CUDA required for FLA testing")
        return
    
    device = torch.device('cuda')
    x = torch.randn(2, 32, 256, device=device)
    
    try:
        from fla.layers import GatedLinearAttention, MultiScaleRetention, BasedLinearAttention, DeltaNet
        
        # Test GLA
        print("\n🧪 Testing GLA...")
        gla = GatedLinearAttention(hidden_size=256, num_heads=4, mode='chunk').to(device)
        result = gla(x)
        print(f"GLA returns: {type(result)}")
        if isinstance(result, tuple):
            print(f"  Tuple length: {len(result)}")
            for i, elem in enumerate(result):
                print(f"  Element {i}: {type(elem)}")
                if hasattr(elem, 'shape'):
                    print(f"    Shape: {elem.shape}")
                elif elem is not None:
                    print(f"    Value: {elem}")
                else:
                    print(f"    Value: None")
        else:
            print(f"  Single tensor shape: {result.shape}")
        
        # Test RetNet
        print("\n🧪 Testing RetNet...")
        retnet = MultiScaleRetention(hidden_size=256, num_heads=4, mode='chunk').to(device)
        result = retnet(x)
        print(f"RetNet returns: {type(result)}")
        if isinstance(result, tuple):
            print(f"  Tuple length: {len(result)}")
            for i, elem in enumerate(result):
                print(f"  Element {i}: {type(elem)}")
                if hasattr(elem, 'shape'):
                    print(f"    Shape: {elem.shape}")
                elif elem is not None:
                    print(f"    Value: {elem}")
                else:
                    print(f"    Value: None")
        else:
            print(f"  Single tensor shape: {result.shape}")
        
        # Test BasedLinearAttention
        print("\n🧪 Testing BasedLinearAttention...")
        try:
            based = BasedLinearAttention(hidden_size=256, num_heads=8, mode='chunk').to(device)
            result = based(x)
            print(f"BasedLinearAttention returns: {type(result)}")
            if isinstance(result, tuple):
                print(f"  Tuple length: {len(result)}")
                for i, elem in enumerate(result):
                    print(f"  Element {i}: {type(elem)}")
                    if hasattr(elem, 'shape'):
                        print(f"    Shape: {elem.shape}")
                    elif elem is not None:
                        print(f"    Value: {elem}")
                    else:
                        print(f"    Value: None")
            else:
                print(f"  Single tensor shape: {result.shape}")
        except Exception as e:
            print(f"  BasedLinearAttention failed: {e}")
            print("  Skipping BasedLinearAttention for now...")
        
        # Test DeltaNet
        print("\n🧪 Testing DeltaNet...")
        deltanet = DeltaNet(hidden_size=256, num_heads=4, mode='chunk').to(device)
        result = deltanet(x)
        print(f"DeltaNet returns: {type(result)}")
        if isinstance(result, tuple):
            print(f"  Tuple length: {len(result)}")
            for i, elem in enumerate(result):
                print(f"  Element {i}: {type(elem)}")
                if hasattr(elem, 'shape'):
                    print(f"    Shape: {elem.shape}")
                elif elem is not None:
                    print(f"    Value: {elem}")
                else:
                    print(f"    Value: None")
        else:
            print(f"  Single tensor shape: {result.shape}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_fla_returns()