#!/usr/bin/env python3
"""
Setup script to install Flash Linear Attention and verify installation
"""
import subprocess
import sys
import torch

def install_fla():
    """Install Flash Linear Attention library"""
    print("🔧 Installing Flash Linear Attention...")
    
    # Check PyTorch version
    torch_version = torch.__version__
    print(f"📦 PyTorch version: {torch_version}")
    
    if torch.version.cuda:
        print(f"🔥 CUDA version: {torch.version.cuda}")
    else:
        print("⚠️ No CUDA detected")
    
    # Install FLA
    try:
        # First try pip install
        print("📦 Installing flash-linear-attention...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "flash-linear-attention"
        ])
        print("✅ flash-linear-attention installed successfully")
        
    except subprocess.CalledProcessError:
        print("❌ pip install failed, trying from source...")
        try:
            # Try installing from source
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "git+https://github.com/fla-org/flash-linear-attention.git"
            ])
            print("✅ flash-linear-attention installed from source")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install FLA: {e}")
            return False
    
    return True

def verify_fla():
    """Verify FLA installation"""
    print("\n🧪 Verifying Flash Linear Attention installation...")
    
    try:
        # Test basic imports
        import fla
        print(f"✅ FLA imported successfully, version: {getattr(fla, '__version__', 'unknown')}")
        
        # Test layer imports
        from fla.layers import GatedLinearAttention, MultiScaleRetention
        print("✅ Core layers imported successfully")
        
        # Test modules
        from fla.modules import RMSNorm, RotaryEmbedding
        print("✅ Modules imported successfully")
        
        # Test a simple forward pass (requires CUDA)
        if not torch.cuda.is_available():
            print("⚠️ CUDA not available - FLA requires CUDA for Triton kernels")
            print("✅ FLA imports work, but forward pass requires GPU")
            return True
        
        print("\n🧪 Testing GLA layer on CUDA...")
        device = torch.device('cuda')
        
        gla = GatedLinearAttention(
            hidden_size=256,
            num_heads=4,
            mode='chunk'
        ).to(device)
        
        # Create test input on CUDA
        x = torch.randn(2, 32, 256, device=device)  # [batch, seq_len, hidden_size]
        
        with torch.no_grad():
            from utils.fla_utils import safe_fla_forward
            output = safe_fla_forward(gla, x)
        
        print(f"✅ GLA forward pass successful: {x.shape} -> {output.shape}")
        
        # Test RetNet
        print("\n🧪 Testing RetNet layer on CUDA...")
        retnet = MultiScaleRetention(
            hidden_size=256,
            num_heads=4,
            mode='chunk'
        ).to(device)
        
        with torch.no_grad():
            output = safe_fla_forward(retnet, x)
        
        print(f"✅ RetNet forward pass successful: {x.shape} -> {output.shape}")
        
        print("\n🎉 Flash Linear Attention is working correctly!")
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def install_dependencies():
    """Install other required dependencies"""
    print("\n📦 Installing other dependencies...")
    
    dependencies = [
        "torch>=2.5.0",
        "transformers>=4.45.0", 
        "datasets>=3.3.0",
        "einops",
        "tqdm",
        "numpy",
        "matplotlib",
        "seaborn"
    ]
    
    for dep in dependencies:
        try:
            print(f"Installing {dep}...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", dep
            ])
        except subprocess.CalledProcessError:
            print(f"⚠️ Failed to install {dep}, continuing...")
    
    print("✅ Dependencies installation completed")

def main():
    print("🚀 FLASH LINEAR ATTENTION SETUP")
    print("=" * 50)
    
    # Check Python version
    python_version = sys.version_info
    print(f"🐍 Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version < (3, 8):
        print("❌ Python 3.8+ required")
        return False
    
    # Install dependencies first
    install_dependencies()
    
    # Install FLA
    if not install_fla():
        print("\n❌ FLA installation failed!")
        print("💡 You can try manual installation:")
        print("   pip install flash-linear-attention")
        print("   or")
        print("   pip install git+https://github.com/fla-org/flash-linear-attention.git")
        return False
    
    # Verify installation
    if not verify_fla():
        print("\n❌ FLA verification failed!")
        return False
    
    print("\n✅ SETUP COMPLETED SUCCESSFULLY!")
    print("🚀 You can now run experiments with Flash Linear Attention")
    print("\nNext steps:")
    print("  python test_framework.py          # Test the framework")
    print("  python run_experiments.py --experiment-set attention  # Run attention experiments")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)