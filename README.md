# 🫐 Train Your Own Small Language Model

A minimal toolkit for training and using small language models with the Muon optimizer.

YouTube video: https://youtu.be/Dw0b0Kc9Kpk

## 🚀 Quick Start

### Option 1: Google Colab (No Setup Required)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1m9wXIkMlSVW3whSHZiOywMdUClj0amnZ?usp=sharing)

Click the badge above to run everything in your browser with free GPU access!

### Option 2: Local Setup
```bash
# Clone and setup
git clone https://github.com/vukrosic/build-and-release-your-own-llm
cd build-and-release-your-own-llm
python setup.py  # Installs requirements and creates .env file
```

## 🎯 Four Ways to Use This Project

### 1. 🚀 Quick Start - Use My Pre-trained Model

Want to try text generation immediately?

```bash
# Install dependencies
pip install -r requirements.txt

# Run inference with my pre-trained model
python inference.py
```

The script will:
- Show available checkpoints from `vukrosic/blueberry-1`
- Download the model automatically
- Let you generate text interactively

**No setup required!** The model downloads automatically.

### 2. 🏗️ Train Your Own Model (Single GPU)

Want to train from scratch on one GPU?

```bash
# Install dependencies
pip install -r requirements.txt

# Start training (takes ~20 minutes on GPU)
python train_llm.py

# Use your trained model
python inference.py
```

Your model will be saved in `checkpoints/` and you can resume training anytime.

### 3. ⚡ Distributed Training (Multi-GPU)

Want to train faster with multiple GPUs?

```bash
# Install dependencies
pip install -r requirements.txt

# For 2x RTX 4090 or similar high-end GPUs:
python run_4090.py

# Monitor GPU utilization (in another terminal):
python gpu_monitor.py

# If you experience GPU imbalance, run diagnostics:
python fix_gpu_imbalance.py
```

**Distributed Training Features:**
- **Automatic GPU detection** and memory optimization
- **Load balancing** across multiple GPUs
- **Real-time monitoring** of GPU utilization
- **Fallback options** if distributed training fails
- **Memory-safe settings** for RTX 4090 class GPUs

**GPU Utilization Monitoring:**
The `gpu_monitor.py` script shows real-time GPU stats:
```
[14:23:45] GPU Status:
  GPU 0: 🟢  87% util | 18234/24564MB (74.2%) | 67°C | NVIDIA GeForce RTX 4090
  GPU 1: 🟢  89% util | 18891/24564MB (76.9%) | 69°C | NVIDIA GeForce RTX 4090
  📊 Average: 88.0% | Range: 87%-89% | Imbalance: 2%
```

### 4. 📤 Train and Share Your Model

Want to share your model on Hugging Face?

```bash
# 1. Copy environment template
cp .env.example .env

# 2. Edit .env file:
# HF_REPO_NAME=your-username/your-model-name
# HF_TOKEN=hf_your_token_here
# PUSH_TO_HUB=true

# 3. Train (single or multi-GPU)
python train_llm.py          # Single GPU
# OR
python run_4090.py           # Multi-GPU
```

Get your HF token from: https://huggingface.co/settings/tokens

## 📁 Project Structure

```
├── train_llm.py              # Single GPU training with Muon optimizer
├── train_distributed_llm.py  # Multi-GPU distributed training
├── run_4090.py               # Optimized launcher for RTX 4090 class GPUs
├── gpu_monitor.py            # Real-time GPU utilization monitoring
├── fix_gpu_imbalance.py      # Diagnostic tool for GPU load balancing
├── inference.py              # Text generation and model loading
├── upload_to_hf.py           # Upload checkpoints to Hugging Face
├── example_usage.py          # Example workflow script
├── setup.py                  # Easy setup script
├── requirements.txt          # Python dependencies
├── .env.example             # Environment variables template
└── README.md                # This file
```

## 🎯 What You Get

- **21M parameter transformer model** (384d, 6 layers, 8 heads)
- **Muon optimizer** for efficient training
- **Automatic checkpointing** every 5000 steps
- **Resume training** from any checkpoint
- **Interactive text generation**
- **Hugging Face integration** (optional)

## 📊 Expected Results

- **Training time**: ~16-20 minutes on modern GPU
- **Final perplexity**: ~1.06
- **Model size**: ~21M parameters
- **Memory usage**: ~4-6GB GPU

## 🔧 Customization

### Change Model Size
Edit `train_llm.py`:
```python
@dataclass
class ModelConfig:
    d_model: int = 512      # Bigger model (was 384)
    n_layers: int = 8       # More layers (was 6)
    max_steps: int = 5000  # Train longer for better results (20000)
```

### Use Your Own Data
Edit the dataset loading in `train_llm.py`:
```python
# Replace this line:
dataset = load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", split="train", streaming=True)

# With your dataset:
dataset = load_dataset("your-dataset-name", split="train", streaming=True)
```

### Adjust Training Speed
```python
batch_size: int = 16        # Smaller = less memory
gradient_accumulation_steps: int = 8  # Larger = same effective batch size
```

## 📊 Understanding the Output

### During Training
```
Training: 67%|██████▋   | 20000/30000 [12:34<06:15, 26.6it/s, loss=1.234, acc=0.876, ppl=3.4, lr=8.5e-03]
```

- **loss**: Lower is better (target: ~1.0)
- **acc**: Accuracy (target: ~98%)
- **ppl**: Perplexity (target: ~1.1)
- **lr**: Learning rate (automatically scheduled)

### During Inference
```
Prompt: The future of AI is
Generated text: The future of AI is bright and full of possibilities. Machine learning algorithms continue to evolve...
```

## 🚨 Common Issues

### "CUDA out of memory"
```python
# In train_llm.py, reduce batch size:
batch_size: int = 12  # or even 8
```

### "No checkpoints found"
Make sure you've run training first:
```bash
python train_llm.py  # Wait for it to complete
python inference.py  # Now this will work
```

### "HF upload failed"
Check your token permissions:
1. Go to https://huggingface.co/settings/tokens
2. Make sure token has "Write" permission
3. Update your `.env` file

## ⚡ Distributed Training Guide

### Understanding the Scripts

**`run_4090.py`** - The main launcher that:
- Sets optimal memory settings for RTX 4090 class GPUs
- Clears GPU memory to prevent fragmentation
- Calls `run_novita_4090_training()` from `train_distributed_llm.py`
- Provides fallback options if distributed training fails

**`train_distributed_llm.py`** - Contains the distributed training logic:
- `run_novita_4090_training()` - Main entry point for multi-GPU training
- `launch_distributed_ddp()` - PyTorch native DDP implementation (recommended)
- `launch_distributed()` - Custom distributed training with Muon optimizer
- `main_ddp()` - DDP training loop with standard optimizers
- `main()` - Custom distributed training loop

### Training Flow

1. **`run_4090.py`** → calls → **`run_novita_4090_training()`**
2. **`run_novita_4090_training()`** tries in order:
   - PyTorch DDP (most reliable)
   - Custom distributed training
   - Single GPU fallback

### GPU Utilization Troubleshooting

If you see uneven GPU utilization (e.g., 90% vs 19%):

```bash
# 1. Run diagnostics
python fix_gpu_imbalance.py

# 2. Monitor in real-time
python gpu_monitor.py

# 3. Try the balanced launcher
python run_balanced.py  # Created by fix_gpu_imbalance.py
```

**Common Causes of GPU Imbalance:**
- Data loading bottlenecks
- Memory fragmentation
- Poor batch distribution
- CPU affinity issues

**Solutions Applied:**
- Rank 0 loads data, broadcasts to other GPUs
- Improved distributed samplers with proper padding
- Memory optimization and periodic cache clearing
- CPU affinity optimization for data loading

**Note on GPU Monitoring:**
Different monitoring tools may show different utilization percentages:
- **Novita UI**: May show instantaneous or averaged values
- **gpu_monitor.py**: Shows real-time utilization every 2 seconds
- **nvidia-smi**: Shows current utilization at query time

If you see discrepancies (e.g., Novita shows 90%/19% but gpu_monitor.py shows 90%/90%), this usually means:
- The imbalance is intermittent (happening between measurements)
- Different measurement intervals are being used
- One tool is showing peak values while another shows averages

**Recommendation**: Run `gpu_monitor.py` for several minutes to see the true utilization pattern over time.

### Advanced Configuration

Edit `train_distributed_llm.py` to customize:

```python
# At the top of the file:
NUM_GPUS = 2                    # Number of GPUs to use
BASE_BATCH_SIZE = 6             # Batch size per GPU
BASE_LR = 0.01                  # Base learning rate
GPU_IDS = [0, 1]                # Specific GPU IDs to use
```

### Performance Expectations

**Single GPU (RTX 4090):**
- Training time: ~16-20 minutes
- Memory usage: ~18-20GB
- Utilization: ~85-95%

**Dual GPU (2x RTX 4090):**
- Training time: ~8-12 minutes
- Memory usage: ~18-20GB per GPU
- Utilization: ~80-90% each (balanced)

## 🎉 What's Next?

1. **Experiment with prompts** - Try different starting texts
2. **Adjust generation parameters** - Change temperature and top_k in inference.py
3. **Train on your data** - Replace the dataset with your own text
4. **Scale up** - Increase model size for better performance
5. **Multi-GPU training** - Use distributed training for faster results
6. **Monitor performance** - Use gpu_monitor.py to optimize utilization
7. **Share your model** - Upload to Hugging Face for others to use

## 📦 Checkpoint Management

### Automatic Checkpointing
The training script now saves checkpoints every 5000 steps in the `checkpoints/` directory:
```
checkpoints/
├── checkpoint_step_5000/
│   ├── model.pt          # Model weights and optimizer state
│   ├── config.json       # Model configuration
│   └── tokenizer files   # Tokenizer configuration
├── checkpoint_step_10000/
└── checkpoint_step_15000/
```

### Upload to Hugging Face
Share your trained models with the community:

```bash
# Set your Hugging Face token
export HF_TOKEN="hf_your_token_here"

# List available checkpoints
python upload_to_hf.py --list

# Upload latest checkpoint
python upload_to_hf.py --repo-name username/my-awesome-model

# Upload specific checkpoint
python upload_to_hf.py --repo-name username/my-model --checkpoint checkpoints/checkpoint_step_10000

# Create private repository
python upload_to_hf.py --repo-name username/my-model --private
```

Get your token from: https://huggingface.co/settings/tokens

### Example Workflow
```bash
# Run the complete example
python example_usage.py

# Or step by step:
python train_llm.py                    # Train model (saves checkpoints)
python upload_to_hf.py --list          # See available checkpoints  
python upload_to_hf.py --repo-name username/model  # Upload to HF
```

## 💡 Pro Tips

- **Resume training**: The script automatically detects checkpoints
- **Monitor GPU usage**: Use `nvidia-smi` to check memory usage
- **Save compute**: Use smaller models for experimentation
- **Better results**: More training steps = better model (usually)
- **Checkpoint frequency**: Adjust `save_every` in ModelConfig for different intervals
- **Share early**: Upload intermediate checkpoints to track training progress

## 🎯 HellaSwag Benchmark

Evaluate your trained model's common sense reasoning capabilities with the HellaSwag benchmark!

### What is HellaSwag?

HellaSwag is a benchmark for evaluating common sense reasoning in language models. It presents a context and four possible endings, where the model must choose the most plausible continuation.

**Example:**
```
Context: "A woman is outside with a bucket and a dog. The dog is running around trying to avoid a bath. She..."
Endings:
A) "rinses the bucket out with soap and water."
B) "uses a hose to fill the bucket with water."  ← Correct
C) "gets into the bathtub with the dog."
D) "starts doing jumping jacks."
```

### 🚀 Quick Benchmark

For a fast evaluation on 100 random examples:

```bash
python quick_benchmark.py
```

This will:
- Show available checkpoints
- Let you choose which model to evaluate
- Run a quick test on 100 HellaSwag examples
- Show accuracy vs random baseline (25%)

### 📊 Full Benchmark

For comprehensive evaluation on the full validation set:

```bash
# Full evaluation (1000+ examples)
python hellaswag_benchmark.py

# Evaluate specific checkpoint
python hellaswag_benchmark.py --checkpoint checkpoints/checkpoint-15000

# Limit number of examples
python hellaswag_benchmark.py --max-examples 500

# Use CPU instead of GPU
python hellaswag_benchmark.py --device cpu
```

### 📈 Understanding Results

**Sample Output:**
```
🎯 HELLASWAG BENCHMARK RESULTS
============================================================
📊 Overall Accuracy: 0.347 (347/1000)
🎲 Random Baseline: 0.250 (25%)
📈 Improvement over random: +38.8%

📍 Position Statistics:
  Position 0: 23.1% correct, 28.4% predicted
  Position 1: 26.7% correct, 24.1% predicted  
  Position 2: 25.8% correct, 23.9% predicted
  Position 3: 24.4% correct, 23.6% predicted

🔍 Sample Examples:
Example 1:
Context: A person is seen performing on stage with a guitar...
Correct ending (1): continues to play the guitar and sing.
Predicted ending (1): continues to play the guitar and sing.
Result: ✅ Correct
```

**What the metrics mean:**
- **Overall Accuracy**: Percentage of correct predictions (higher is better)
- **Random Baseline**: 25% (chance level with 4 choices)
- **Improvement**: How much better than random guessing
- **Position Statistics**: Shows if model has bias toward certain answer positions

### 🎯 Expected Performance

**Typical results for small models:**
- **Untrained model**: ~25% (random)
- **Early training (5K steps)**: ~28-32%
- **Well-trained model (15K+ steps)**: ~35-45%
- **Large models (GPT-3 scale)**: ~78-85%

**Performance factors:**
- **Model size**: Larger models generally perform better
- **Training data**: More diverse data helps reasoning
- **Training steps**: More training usually improves performance
- **Architecture**: Some designs are better for reasoning tasks

### 🔧 Benchmark Features

**hellaswag_benchmark.py** includes:
- ✅ Full HellaSwag validation set evaluation
- ✅ Automatic checkpoint detection
- ✅ GPU/CPU support
- ✅ Progress tracking with tqdm
- ✅ Detailed results with examples
- ✅ JSON results export
- ✅ Position bias analysis

**quick_benchmark.py** includes:
- ✅ Fast evaluation on sample data
- ✅ Works with both training scripts
- ✅ Simple accuracy reporting
- ✅ Interactive checkpoint selection

### 📊 Tracking Progress

Monitor your model's reasoning ability throughout training:

```bash
# Evaluate at different training stages
python quick_benchmark.py  # Choose checkpoint-5000
python quick_benchmark.py  # Choose checkpoint-10000  
python quick_benchmark.py  # Choose checkpoint-15000

# Plot improvement over time
# Accuracy: 0.287 → 0.324 → 0.347
```

### 🚨 Troubleshooting

**"Failed to load HellaSwag dataset"**
```bash
# Install datasets library
pip install datasets

# Check internet connection
# The dataset downloads automatically on first use
```

**"CUDA out of memory during evaluation"**
```bash
# Use CPU for evaluation
python hellaswag_benchmark.py --device cpu

# Or reduce batch processing (edit the script)
```

**"No checkpoints found"**
```bash
# Make sure you've trained a model first
python train_llm.py

# Or specify checkpoint path directly
python hellaswag_benchmark.py --checkpoint path/to/your/checkpoint
```

### 💡 Improving HellaSwag Performance

**Training strategies:**
- **More training steps**: 20K+ steps often help
- **Diverse data**: Include reasoning-heavy text
- **Larger context**: Increase max_seq_len for better context understanding
- **Model scaling**: Larger models (more layers/dimensions) perform better

**Data considerations:**
- **Common sense text**: Stories, how-to guides, explanations
- **Causal reasoning**: Text with cause-and-effect relationships
- **Narrative structure**: Stories with logical progressions

Happy benchmarking! 🚀
