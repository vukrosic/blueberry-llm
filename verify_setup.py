#!/usr/bin/env python3
"""
Verify 8x RTX 4090 setup for distributed research
Run this first to check everything is working
"""

import torch
import os
import subprocess
import sys

def check_cuda_setup():
    """Check CUDA and GPU setup"""
    print("🔍 CHECKING CUDA SETUP")
    print("-" * 30)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return False
    
    gpu_count = torch.cuda.device_count()
    print(f"✅ CUDA available")
    print(f"📊 Detected {gpu_count} GPUs")
    
    if gpu_count < 8:
        print(f"⚠️ Warning: Expected 8 GPUs, found {gpu_count}")
    
    # Check each GPU
    for i in range(min(gpu_count, 8)):
        try:
            gpu_name = torch.cuda.get_device_name(i)
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / 1e9
            
            print(f"  GPU {i}: {gpu_name}")
            print(f"    Memory: {memory_gb:.1f} GB")
            print(f"    Compute: {props.major}.{props.minor}")
            
            # Test GPU
            torch.cuda.set_device(i)
            test_tensor = torch.randn(1000, 1000).cuda()
            result = torch.mm(test_tensor, test_tensor.t())
            print(f"    Test: ✅ Working")
            
        except Exception as e:
            print(f"    Test: ❌ Failed - {e}")
            return False
    
    return True

def check_distributed_setup():
    """Check if distributed training tools are available"""
    print("\n🔍 CHECKING DISTRIBUTED SETUP")
    print("-" * 30)
    
    # Check torchrun
    try:
        result = subprocess.run(['torchrun', '--help'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ torchrun available")
        else:
            print("❌ torchrun not working")
            return False
    except FileNotFoundError:
        print("❌ torchrun not found")
        print("Install with: pip install torch")
        return False
    
    # Check NCCL
    try:
        import torch.distributed as dist
        print("✅ torch.distributed available")
    except ImportError:
        print("❌ torch.distributed not available")
        return False
    
    return True

def check_dependencies():
    """Check required Python packages"""
    print("\n🔍 CHECKING DEPENDENCIES")
    print("-" * 30)
    
    required_packages = [
        'torch',
        'transformers', 
        'datasets',
        'tqdm',
        'numpy'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing.append(package)
    
    if missing:
        print(f"\nInstall missing packages:")
        print(f"pip install {' '.join(missing)}")
        return False
    
    return True

def check_files():
    """Check required files exist"""
    print("\n🔍 CHECKING FILES")
    print("-" * 30)
    
    required_files = [
        'train_distributed_llm.py',
        'tiny_llm_benchmarks.py',
        'research_framework.py', 
        'run_distributed_research.py',
        'start_research.py'
    ]
    
    missing = []
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file}")
            missing.append(file)
    
    if missing:
        print(f"\nMissing files: {missing}")
        return False
    
    return True

def run_quick_test():
    """Run a very quick distributed test"""
    print("\n🔍 RUNNING QUICK DISTRIBUTED TEST")
    print("-" * 30)
    
    try:
        # Simple multi-GPU test
        if torch.cuda.device_count() >= 2:
            print("Testing multi-GPU tensor operations...")
            
            # Test on GPU 0
            torch.cuda.set_device(0)
            tensor_0 = torch.randn(100, 100).cuda()
            
            # Test on GPU 1  
            torch.cuda.set_device(1)
            tensor_1 = torch.randn(100, 100).cuda()
            
            print("✅ Multi-GPU tensor operations working")
        else:
            print("⚠️ Less than 2 GPUs, skipping multi-GPU test")
        
        return True
        
    except Exception as e:
        print(f"❌ Multi-GPU test failed: {e}")
        return False

def main():
    print("🔬 DISTRIBUTED RESEARCH SETUP VERIFICATION")
    print("=" * 50)
    
    checks = [
        ("CUDA Setup", check_cuda_setup),
        ("Distributed Setup", check_distributed_setup), 
        ("Dependencies", check_dependencies),
        ("Files", check_files),
        ("Quick Test", run_quick_test)
    ]
    
    all_passed = True
    
    for name, check_func in checks:
        try:
            if not check_func():
                all_passed = False
        except Exception as e:
            print(f"❌ {name} check failed with error: {e}")
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 ALL CHECKS PASSED!")
        print("✅ System ready for distributed research")
        print("\nNext step: Run 'python start_research.py'")
    else:
        print("❌ SOME CHECKS FAILED")
        print("🔧 Fix the issues above before running research")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)