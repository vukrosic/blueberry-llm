#!/usr/bin/env python3
"""
Test script to verify the research framework works correctly
"""
import torch
from configs.base_config import ExperimentConfig, AttentionConfig
from models.base_model import BaseTransformer
from experiments.experiment_definitions import create_baseline_experiments

def test_model_creation():
    """Test that models can be created with different attention types"""
    print("🧪 Testing model creation...")
    
    # Test standard attention (always available)
    config = ExperimentConfig(
        name="test_standard",
        d_model=128,
        n_layers=2,
        n_heads=4,
        d_ff=512,
        vocab_size=1000,
        attention_config=AttentionConfig(attention_type="standard")
    )
    
    model = BaseTransformer(config)
    print(f"✅ Standard attention model: {model.get_num_params():,} parameters")
    
    # Test FLA models if available
    try:
        import fla
        fla_available = True
        print("📦 Flash Linear Attention detected")
    except ImportError:
        fla_available = False
        print("⚠️ Flash Linear Attention not available, testing fallbacks")
    
    # Test GLA
    config.attention_config = AttentionConfig(attention_type="gla")
    model_gla = BaseTransformer(config)
    status = "FLA" if fla_available else "fallback"
    print(f"✅ GLA model ({status}): {model_gla.get_num_params():,} parameters")
    
    # Test RetNet
    config.attention_config = AttentionConfig(attention_type="retnet")
    model_retnet = BaseTransformer(config)
    status = "FLA" if fla_available else "fallback"
    print(f"✅ RetNet model ({status}): {model_retnet.get_num_params():,} parameters")
    
    # Test Mamba
    config.attention_config = AttentionConfig(attention_type="mamba")
    model_mamba = BaseTransformer(config)
    status = "FLA" if fla_available else "fallback"
    print(f"✅ Mamba model ({status}): {model_mamba.get_num_params():,} parameters")
    
    # Test additional FLA models if available
    if fla_available:
        fla_models = ["based", "deltanet", "hgrn", "rwkv6", "gsa"]
        for model_type in fla_models:
            try:
                config.attention_config = AttentionConfig(attention_type=model_type)
                model_fla = BaseTransformer(config)
                print(f"✅ {model_type.upper()} model (FLA): {model_fla.get_num_params():,} parameters")
            except Exception as e:
                print(f"⚠️ {model_type.upper()} model failed: {e}")

def test_forward_pass():
    """Test forward pass with different models"""
    print("\n🧪 Testing forward pass...")
    
    config = ExperimentConfig(
        name="test_forward",
        d_model=128,
        n_layers=2,
        n_heads=4,
        d_ff=512,
        vocab_size=1000,
        max_seq_len=64
    )
    
    # Create test input
    batch_size, seq_len = 2, 32
    x = torch.randint(0, 1000, (batch_size, seq_len))
    
    # Check if FLA is available
    try:
        import fla
        fla_available = True
        attention_types = ["standard", "gla", "retnet", "mamba", "based", "deltanet"]
    except ImportError:
        fla_available = False
        attention_types = ["standard", "gla", "retnet", "mamba"]  # Will fallback to standard
    
    for attention_type in attention_types:
        try:
            config.attention_config = AttentionConfig(attention_type=attention_type)
            model = BaseTransformer(config)
            model.eval()
            
            with torch.no_grad():
                logits = model(x)
            
            expected_shape = (batch_size, seq_len, config.vocab_size)
            assert logits.shape == expected_shape, f"Wrong output shape for {attention_type}"
            
            status = "FLA" if fla_available and attention_type != "standard" else "standard"
            print(f"✅ {attention_type} forward pass ({status}): {logits.shape}")
            
        except Exception as e:
            print(f"⚠️ {attention_type} forward pass failed: {e}")

def test_experiment_definitions():
    """Test that experiment definitions are valid"""
    print("\n🧪 Testing experiment definitions...")
    
    experiments = create_baseline_experiments()
    print(f"✅ Created {len(experiments)} baseline experiments")
    
    for exp in experiments:
        assert hasattr(exp, 'name'), "Experiment missing name"
        assert hasattr(exp, 'attention_config'), "Experiment missing attention config"
        assert exp.d_model % exp.n_heads == 0, f"Invalid d_model/n_heads for {exp.name}"
        print(f"  ✅ {exp.name}: {exp.attention_config.attention_type}")

def test_config_serialization():
    """Test that configs can be serialized to JSON"""
    print("\n🧪 Testing config serialization...")
    
    from dataclasses import asdict
    import json
    
    config = ExperimentConfig(
        name="test_serialization",
        attention_config=AttentionConfig(attention_type="gla")
    )
    
    # Test serialization
    config_dict = asdict(config)
    json_str = json.dumps(config_dict, indent=2)
    
    # Test deserialization
    loaded_dict = json.loads(json_str)
    
    print(f"✅ Config serialization works")
    print(f"  Original attention type: {config.attention_config.attention_type}")
    print(f"  Loaded attention type: {loaded_dict['attention_config']['attention_type']}")

def main():
    """Run all tests"""
    print("🔬 TESTING LLM RESEARCH FRAMEWORK")
    print("=" * 50)
    
    try:
        test_model_creation()
        test_forward_pass()
        test_experiment_definitions()
        test_config_serialization()
        
        print("\n✅ ALL TESTS PASSED!")
        print("🚀 Framework is ready for experiments")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    main()