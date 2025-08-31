#!/usr/bin/env python3
"""
Example script showing how to run a single experiment
"""
import torch
from configs.base_config import ExperimentConfig, AttentionConfig
from experiments.experiment_runner import ExperimentRunner

def main():
    print("🧪 SINGLE EXPERIMENT EXAMPLE")
    print("=" * 40)
    
    # Create a simple experiment configuration
    config = ExperimentConfig(
        name="example_gla_small",
        description="Small GLA model for testing",
        hypothesis="GLA should work better than standard attention",
        
        # Small model for quick testing
        d_model=256,
        n_layers=4,
        n_heads=4,
        d_ff=1024,
        
        # Short training for demo
        max_steps=1000,
        batch_size=8,
        learning_rate=0.01,
        
        # Use GLA attention
        attention_config=AttentionConfig(
            attention_type="gla",
            expand_k=0.5,
            expand_v=1.0,
            use_gk=True,
            use_output_gate=True
        ),
        
        # Small dataset for quick testing
        num_documents=1000,
        max_tokens=500000,
        max_seq_len=256
    )
    
    print(f"📋 Experiment: {config.name}")
    print(f"🏗️ Architecture: {config.d_model}d, {config.n_layers}L, {config.n_heads}H")
    print(f"🔧 Attention: {config.attention_config.attention_type}")
    print(f"📚 Training: {config.max_steps} steps")
    
    # Check if we should use distributed training
    use_distributed = torch.cuda.device_count() > 1
    if use_distributed:
        print(f"⚡ Using {torch.cuda.device_count()} GPUs")
    else:
        print("🔧 Using single GPU/CPU")
    
    # Create experiment runner
    runner = ExperimentRunner(
        results_dir="example_results",
        use_distributed=use_distributed
    )
    
    # Ask for confirmation
    confirm = input("\nRun experiment? (y/N): ").strip().lower()
    if confirm != 'y':
        print("Aborted.")
        return
    
    try:
        # Run the experiment
        print("\n🚀 Starting experiment...")
        result = runner.run_experiment(config)
        
        # Print results
        if result.get('success', False):
            print(f"\n✅ Experiment completed successfully!")
            training_time = result.get('training_time_minutes', 0)
            print(f"⏱️ Training time: {training_time:.1f} minutes")
            
            # Show final metrics if available
            final_metrics = result.get('result', {}).get('final_metrics', {})
            if final_metrics:
                val_loss = final_metrics.get('val_loss', 0)
                val_acc = final_metrics.get('val_accuracy', 0)
                print(f"📊 Final validation loss: {val_loss:.4f}")
                print(f"📊 Final validation accuracy: {val_acc:.3f}")
            
            print(f"📁 Results saved in: example_results/")
            print(f"\n💡 To benchmark this model, run:")
            print(f"   python run_benchmarks.py --results-dir example_results --experiment {config.name}")
            
        else:
            print(f"\n❌ Experiment failed!")
            error = result.get('error', 'Unknown error')
            print(f"Error: {error}")
    
    except KeyboardInterrupt:
        print(f"\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"\n❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        runner.cleanup()

if __name__ == "__main__":
    main()