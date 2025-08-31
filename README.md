# LLM Research Framework

A structured research framework for experimenting with different attention mechanisms and architectures, inspired by Flash Linear Attention. This framework enables systematic ablation studies comparing standard attention, Gated Linear Attention (GLA), RetNet, Mamba, and other architectures.

## 🏗️ Project Structure

```
├── configs/                 # Configuration files
│   └── base_config.py      # Base configuration classes
├── models/                 # Model implementations
│   ├── base_model.py       # Base transformer with pluggable attention
│   └── attention_layers.py # Different attention mechanisms
├── experiments/            # Experiment definitions and runners
│   ├── experiment_runner.py      # Main experiment runner
│   └── experiment_definitions.py # Predefined experiment sets
├── training/               # Training utilities
│   ├── trainer.py          # Main training loop
│   ├── data_utils.py       # Data loading and preprocessing
│   └── optimizers.py       # Optimizers including Muon
├── utils/                  # Utility functions
│   ├── logging.py          # Logging utilities
│   └── benchmarking.py     # Benchmarking integration
├── run_experiments.py      # Main script to run experiments
├── run_benchmarks.py       # Script to benchmark trained models
└── requirements_new.txt    # Updated dependencies
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Automated setup (recommended)
python setup_fla.py

# Or manual installation
pip install -r requirements_new.txt
```

### 2. Run Baseline Experiments

```bash
python run_experiments.py --experiment-set baseline
```

### 3. Run Attention Mechanism Comparisons

```bash
python run_experiments.py --experiment-set attention
```

### 4. Benchmark Trained Models

```bash
python run_benchmarks.py
```

## 🧪 Available Experiments

### Baseline Experiments
- **baseline_small**: Quick baseline with small model
- **baseline_medium**: Standard baseline configuration  
- **baseline_large**: Large model baseline

### Attention Mechanism Experiments
- **attention_gla**: Gated Linear Attention (FLA)
- **attention_gla_no_gating**: GLA without key gating
- **attention_retnet**: Retentive Network (FLA)
- **attention_mamba**: Mamba state space model (FLA)
- **attention_based**: Based Linear Attention (FLA)
- **attention_deltanet**: DeltaNet (FLA)
- **attention_hgrn**: Hierarchically Gated RNN (FLA)
- **attention_rwkv6**: RWKV6 (FLA)
- **attention_gsa**: Gated Slot Attention (FLA)

### Architecture Experiments
- **arch_wide_gla**: Wider GLA model
- **arch_deep_gla**: Deeper GLA model
- **arch_many_heads_gla**: GLA with more attention heads
- **arch_expansion_test**: GLA with different expansion factors

### Training Experiments
- **train_long_context**: Training with longer sequences
- **train_high_lr**: Higher learning rate experiments
- **train_low_dropout**: Lower dropout experiments
- **train_more_data**: Training with more data

### Efficiency Experiments
- **efficiency_standard_512**: Standard attention baseline
- **efficiency_gla_512**: GLA efficiency comparison
- **efficiency_standard_1024**: Standard attention with long context
- **efficiency_gla_1024**: GLA with long context
- **efficiency_mamba_2048**: Mamba with very long context

## 🔧 Configuration

### Model Configuration

```python
from configs.base_config import ExperimentConfig, AttentionConfig

config = ExperimentConfig(
    name="my_experiment",
    description="Custom experiment",
    hypothesis="Testing custom configuration",
    
    # Model architecture
    d_model=384,
    n_layers=6,
    n_heads=8,
    d_ff=1536,
    
    # Training parameters
    max_steps=5000,
    learning_rate=0.01,
    batch_size=16,
    
    # Attention configuration
    attention_config=AttentionConfig(
        attention_type="gla",  # standard, gla, retnet, mamba
        expand_k=0.5,
        expand_v=1.0,
        use_gk=True
    )
)
```

### Running Custom Experiments

```python
from experiments import ExperimentRunner

runner = ExperimentRunner(results_dir="my_results")
result = runner.run_experiment(config)
```

## 📊 Attention Mechanisms (Using Flash Linear Attention)

### Standard Attention
- Multi-head self-attention with rotary embeddings
- Optional Flash Attention for efficiency
- Causal masking for autoregressive generation

### Gated Linear Attention (GLA)
- **Real FLA implementation** with efficient Triton kernels
- Linear attention with gating mechanisms
- Configurable key/value expansion factors
- Hardware-efficient training with chunk mode

### Retentive Network (RetNet)
- **Real FLA implementation** with multi-scale retention
- Retention mechanism with exponential decay
- Parallel and recurrent formulations
- Efficient chunk-based training

### Mamba
- **Real FLA implementation** of state space models
- Selective state space mechanisms
- Linear complexity in sequence length
- Excellent for very long sequences

### Based Linear Attention
- **FLA implementation** of Based architecture
- Simple linear attention with good recall-throughput tradeoff
- Efficient for long sequences

### DeltaNet
- **FLA implementation** with delta rule parallelization
- Efficient parallel training over sequence length
- Good for transformer-like architectures

### HGRN/HGRN2
- **FLA implementations** of Hierarchically Gated RNNs
- State expansion and gating mechanisms
- Good sequence modeling capabilities

### RWKV6
- **FLA implementation** of RWKV with matrix-valued states
- Dynamic recurrence mechanisms
- Efficient for very long contexts

### GSA (Gated Slot Attention)
- **FLA implementation** for efficient linear-time modeling
- Slot-based attention mechanisms
- Good for structured sequence modeling

## 🏃‍♂️ Running Experiments

### Single Experiment Set
```bash
# Run only baseline experiments
python run_experiments.py --experiment-set baseline

# Run attention mechanism comparisons
python run_experiments.py --experiment-set attention

# Run architecture ablations
python run_experiments.py --experiment-set architecture
```

### All Experiments
```bash
python run_experiments.py --experiment-set all
```

### Distributed Training
```bash
# Auto-detect GPUs
python run_experiments.py --experiment-set attention

# Force distributed training
python run_experiments.py --experiment-set attention --distributed

# Force single GPU
python run_experiments.py --experiment-set attention --single-gpu
```

## 📈 Benchmarking

The framework integrates with existing benchmarking tools:

### Comprehensive Benchmarks
- LAMBADA (last word prediction)
- Simple arithmetic
- Sentence completion
- Word association
- Simple QA
- PIQA (physical reasoning)
- SIQA (social reasoning)

### HellaSwag Benchmark
- Common sense reasoning
- Multiple choice completion

### Running Benchmarks
```bash
# Benchmark all trained models
python run_benchmarks.py

# Benchmark specific experiment
python run_benchmarks.py --experiment baseline_medium

# Custom results directory
python run_benchmarks.py --results-dir my_results
```

## 📁 Results Structure

```
results/
├── experiment_name_hash/
│   ├── experiment_config.json    # Experiment configuration
│   ├── results.json              # Training results and metrics
│   └── checkpoints/              # Model checkpoints
│       ├── config.json           # Model configuration
│       ├── final_model.pt        # Final trained model
│       └── checkpoint_step_*.pt  # Intermediate checkpoints
├── experiment_suite_results.json # Combined results
└── all_benchmark_results.json   # Benchmark results
```

## 🔬 Research Focus

This framework is designed for systematic research into:

1. **Attention Mechanism Efficiency**: Comparing quadratic vs linear attention
2. **Architecture Scaling**: How different mechanisms scale with model size
3. **Training Dynamics**: How different architectures learn
4. **Long Context Performance**: Efficiency with longer sequences
5. **Task-Specific Performance**: Which mechanisms work best for different tasks

## 🛠️ Extending the Framework

### Adding New Attention Mechanisms

1. Implement the attention layer in `models/attention_layers.py`
2. Add it to the factory function `get_attention_layer()`
3. Create experiments in `experiments/experiment_definitions.py`

### Adding New Benchmarks

1. Implement benchmark in `utils/benchmarking.py`
2. Integrate with existing benchmark tools
3. Add to the benchmark runner

### Custom Experiments

Create custom experiment configurations and run them with the experiment runner.

## 📚 References

- [Flash Linear Attention](https://github.com/fla-org/flash-linear-attention)
- [Gated Linear Attention](https://arxiv.org/abs/2312.06635)
- [Retentive Network](https://arxiv.org/abs/2307.08621)
- [Mamba](https://arxiv.org/abs/2312.00752)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add your experiments or improvements
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License.