#!/usr/bin/env python3
"""
Simple launcher for distributed research experiments
Run this file to start the research framework
"""

import os
import sys

def main():
    print("🚀 TINY LLM DISTRIBUTED RESEARCH LAUNCHER")
    print("=" * 50)
    print("This will run research experiments on 8x RTX 4090 GPUs")
    print()
    
    # Check if we're in the right directory
    required_files = [
        'train_distributed_llm.py',
        'tiny_llm_benchmarks.py', 
        'research_framework.py',
        'run_distributed_research.py'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"  - {file}")
        print("\nMake sure you're in the correct directory with all research files.")
        return
    
    print("✅ All required files found")
    print()
    
    # Import and run the distributed research
    try:
        from run_distributed_research import main as research_main
        research_main()
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all dependencies are installed:")
        print("pip install torch transformers datasets tqdm numpy")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()