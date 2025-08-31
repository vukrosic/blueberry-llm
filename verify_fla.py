#!/usr/bin/env python3
"""
Quick verification script for Flash Linear Attention
"""
import torch

def main():
    print("🔍 VERIFYING FLASH LINEAR ATTENTION")
    print("=" * 40)
    
    # Check if FLA is available
    try:
        import fla
        print(f"✅ FLA imported successfully")
        print(f"📦 Version: {getattr(fla, '__version__', 'unknown')}")
        
        # Test GLA
        from fla.layers import GatedLinearAttention
        gla = GatedLinearAttention(
            hidden_size=256,
            num_heads=4,
            mode='chunk'
        )
        
        x = torch.randn(2, 32, 256)
        with torch.no_grad():
            output, _ = gla(x)
        
        print(f"✅ GLA test: {x.shape} -> {output.shape}")
        
        # Test RetNet
        from fla.layers import MultiScaleRetention
        retnet = MultiScaleRetention(
            hidden_size=256,
            num_heads=4,
            mode='chunk'
        )
        
        with torch.no_grad():
            output, _ = retnet(x)
        
        print(f"✅ RetNet test: {x.shape} -> {output.shape}")
        
        print("\n🎉 Flash Linear Attention is working!")
        print("🚀 Ready to run experiments with real FLA implementations")
        
        return True
        
    except ImportError as e:
        print(f"❌ FLA not available: {e}")
        print("\n💡 To install Flash Linear Attention:")
        print("   python setup_fla.py")
        print("   or")
        print("   pip install flash-linear-attention")
        
        return False
    
    except Exception as e:
        print(f"❌ FLA test failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n⚠️ Framework will fall back to standard attention")
        print("   You can still run experiments, but won't get FLA optimizations")
    
    exit(0 if success else 1)