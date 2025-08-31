"""
Experiment definitions for systematic ablation studies
"""
from typing import List
from configs.base_config import ExperimentConfig, AttentionConfig

def create_baseline_experiments() -> List[ExperimentConfig]:
    """Create baseline experiments with standard attention"""
    return [
        ExperimentConfig(
            name="baseline_small",
            description="Small baseline model for quick testing",
            hypothesis="Small model should train quickly and establish baseline",
            d_model=256,
            n_layers=4,
            n_heads=4,
            d_ff=1024,
            max_steps=2000,
            attention_config=AttentionConfig(attention_type="standard")
        ),
        ExperimentConfig(
            name="baseline_medium",
            description="Medium baseline model",
            hypothesis="Medium model should provide good performance baseline",
            d_model=384,
            n_layers=6,
            n_heads=8,
            d_ff=1536,
            max_steps=5000,
            attention_config=AttentionConfig(attention_type="standard")
        ),
        ExperimentConfig(
            name="baseline_large",
            description="Large baseline model",
            hypothesis="Large model should achieve best standard attention performance",
            d_model=512,
            n_layers=8,
            n_heads=8,
            d_ff=2048,
            max_steps=5000,
            attention_config=AttentionConfig(attention_type="standard")
        )
    ]

def create_attention_experiments() -> List[ExperimentConfig]:
    """Create experiments comparing different attention mechanisms"""
    base_config = {
        "d_model": 384,
        "n_layers": 6,
        "n_heads": 8,
        "d_ff": 1536,
        "max_steps": 5000
    }
    
    return [
        ExperimentConfig(
            name="attention_gla",
            description="Gated Linear Attention (FLA)",
            hypothesis="GLA should be more efficient than standard attention with similar performance",
            attention_config=AttentionConfig(
                attention_type="gla",
                expand_k=0.5,
                expand_v=1.0,
                use_gk=True,
                use_output_gate=True
            ),
            **base_config
        ),
        ExperimentConfig(
            name="attention_gla_no_gating",
            description="GLA without key gating",
            hypothesis="Key gating in GLA provides performance benefits",
            attention_config=AttentionConfig(
                attention_type="gla",
                expand_k=0.5,
                expand_v=1.0,
                use_gk=False,
                use_output_gate=True
            ),
            **base_config
        ),
        ExperimentConfig(
            name="attention_retnet",
            description="Retentive Network (FLA)",
            hypothesis="RetNet should provide good efficiency with retention mechanism",
            attention_config=AttentionConfig(
                attention_type="retnet",
                use_decay=True
            ),
            **base_config
        ),
        ExperimentConfig(
            name="attention_mamba",
            description="Mamba state space model (FLA)",
            hypothesis="Mamba should excel at long sequences with linear complexity",
            attention_config=AttentionConfig(
                attention_type="mamba",
                state_size=16,
                conv_kernel=4,
                expand=2
            ),
            max_seq_len=1024,  # Test longer sequences
            batch_size=8,      # Reduce batch size for memory
            **{k: v for k, v in base_config.items() if k != "max_steps"}
        ),
        ExperimentConfig(
            name="attention_based",
            description="Based Linear Attention (FLA)",
            hypothesis="Based should provide good balance of efficiency and performance",
            attention_config=AttentionConfig(attention_type="based"),
            **base_config
        ),
        ExperimentConfig(
            name="attention_deltanet",
            description="DeltaNet (FLA)",
            hypothesis="DeltaNet should provide efficient parallel training",
            attention_config=AttentionConfig(attention_type="deltanet"),
            **base_config
        ),
        ExperimentConfig(
            name="attention_hgrn",
            description="Hierarchically Gated RNN (FLA)",
            hypothesis="HGRN should provide good sequence modeling capabilities",
            attention_config=AttentionConfig(attention_type="hgrn"),
            **base_config
        ),
        ExperimentConfig(
            name="attention_rwkv6",
            description="RWKV6 (FLA)",
            hypothesis="RWKV6 should provide good efficiency for long sequences",
            attention_config=AttentionConfig(attention_type="rwkv6"),
            max_seq_len=1024,
            batch_size=8,
            **{k: v for k, v in base_config.items() if k != "max_steps"}
        ),
        ExperimentConfig(
            name="attention_gsa",
            description="Gated Slot Attention (FLA)",
            hypothesis="GSA should provide efficient linear-time sequence modeling",
            attention_config=AttentionConfig(attention_type="gsa"),
            **base_config
        )
    ]

def create_architecture_experiments() -> List[ExperimentConfig]:
    """Create experiments testing different architectural choices"""
    return [
        ExperimentConfig(
            name="arch_wide_gla",
            description="Wide GLA model (more dimensions)",
            hypothesis="Wider GLA model should capture more features per layer",
            d_model=512,
            n_layers=6,
            n_heads=8,
            d_ff=2048,
            max_steps=5000,
            attention_config=AttentionConfig(attention_type="gla")
        ),
        ExperimentConfig(
            name="arch_deep_gla",
            description="Deep GLA model (more layers)",
            hypothesis="Deeper GLA model should learn more complex patterns",
            d_model=384,
            n_layers=12,
            n_heads=8,
            d_ff=1536,
            max_steps=5000,
            attention_config=AttentionConfig(attention_type="gla")
        ),
        ExperimentConfig(
            name="arch_many_heads_gla",
            description="GLA with many attention heads",
            hypothesis="More heads should capture diverse attention patterns",
            d_model=384,
            n_layers=6,
            n_heads=16,
            d_ff=1536,
            max_steps=5000,
            attention_config=AttentionConfig(attention_type="gla")
        ),
        ExperimentConfig(
            name="arch_expansion_test",
            description="GLA with different expansion factors",
            hypothesis="Optimal expansion factors improve GLA performance",
            d_model=384,
            n_layers=6,
            n_heads=8,
            d_ff=1536,
            max_steps=5000,
            attention_config=AttentionConfig(
                attention_type="gla",
                expand_k=1.0,  # Different expansion
                expand_v=2.0
            )
        )
    ]

def create_training_experiments() -> List[ExperimentConfig]:
    """Create experiments testing different training configurations"""
    base_attention = AttentionConfig(attention_type="gla")
    
    return [
        ExperimentConfig(
            name="train_long_context",
            description="Training with longer context",
            hypothesis="Longer context should improve language understanding",
            max_seq_len=1024,
            batch_size=8,  # Reduce for memory
            max_steps=5000,
            attention_config=base_attention
        ),
        ExperimentConfig(
            name="train_high_lr",
            description="Higher learning rate training",
            hypothesis="Higher LR might converge faster for linear attention",
            learning_rate=0.02,
            max_steps=5000,
            attention_config=base_attention
        ),
        ExperimentConfig(
            name="train_low_dropout",
            description="Lower dropout rate",
            hypothesis="Linear attention might need less regularization",
            dropout=0.05,
            max_steps=5000,
            attention_config=base_attention
        ),
        ExperimentConfig(
            name="train_more_data",
            description="Training with more data",
            hypothesis="Linear attention should scale better with more data",
            num_documents=6000,
            max_tokens=2000000,
            max_steps=8000,
            attention_config=base_attention
        )
    ]

def create_efficiency_experiments() -> List[ExperimentConfig]:
    """Create experiments focused on efficiency comparisons"""
    return [
        ExperimentConfig(
            name="efficiency_standard_512",
            description="Standard attention with 512 context",
            hypothesis="Baseline efficiency for standard attention",
            max_seq_len=512,
            max_steps=3000,
            attention_config=AttentionConfig(attention_type="standard")
        ),
        ExperimentConfig(
            name="efficiency_gla_512",
            description="GLA with 512 context",
            hypothesis="GLA should be more efficient than standard attention",
            max_seq_len=512,
            max_steps=3000,
            attention_config=AttentionConfig(attention_type="gla")
        ),
        ExperimentConfig(
            name="efficiency_standard_1024",
            description="Standard attention with 1024 context",
            hypothesis="Standard attention should struggle with longer context",
            max_seq_len=1024,
            batch_size=8,
            max_steps=3000,
            attention_config=AttentionConfig(attention_type="standard")
        ),
        ExperimentConfig(
            name="efficiency_gla_1024",
            description="GLA with 1024 context",
            hypothesis="GLA should handle longer context more efficiently",
            max_seq_len=1024,
            batch_size=8,
            max_steps=3000,
            attention_config=AttentionConfig(attention_type="gla")
        ),
        ExperimentConfig(
            name="efficiency_mamba_2048",
            description="Mamba with very long context",
            hypothesis="Mamba should excel at very long sequences",
            max_seq_len=2048,
            batch_size=4,
            max_steps=3000,
            attention_config=AttentionConfig(attention_type="mamba")
        )
    ]