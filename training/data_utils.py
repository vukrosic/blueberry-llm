"""
Data loading and preprocessing utilities
"""
import os
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from typing import List, Tuple, Optional
from tqdm import tqdm

from configs.base_config import ExperimentConfig

class TextTokenDataset(Dataset):
    """Dataset for token sequences"""
    
    def __init__(self, tokens: List[int], seq_len: int = 512):
        self.tokens = tokens
        self.seq_len = seq_len
    
    def __len__(self):
        return max(0, len(self.tokens) - self.seq_len)
    
    def __getitem__(self, idx):
        x = torch.tensor(self.tokens[idx:idx + self.seq_len], dtype=torch.long)
        y = torch.tensor(self.tokens[idx + 1:idx + self.seq_len + 1], dtype=torch.long)
        return x, y

class DistributedSampler:
    """Simple distributed sampler"""
    
    def __init__(self, dataset, rank: int, world_size: int, shuffle: bool = True):
        self.dataset = dataset
        self.rank = rank
        self.world_size = world_size
        self.shuffle = shuffle
        self.epoch = 0
        
        # Calculate samples per rank
        self.total_size = len(dataset)
        self.num_samples = (self.total_size + world_size - 1) // world_size
        self.padded_size = self.num_samples * world_size
    
    def __iter__(self):
        # Generate indices
        indices = list(range(self.total_size))
        
        if self.shuffle:
            # Use epoch as seed for consistent shuffling
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(indices), generator=g).tolist()
        
        # Pad indices
        padding_size = self.padded_size - len(indices)
        if padding_size > 0:
            indices += indices[:padding_size]
        
        # Subsample for this rank
        indices = indices[self.rank:self.padded_size:self.world_size]
        
        return iter(indices)
    
    def __len__(self):
        return self.num_samples
    
    def set_epoch(self, epoch: int):
        self.epoch = epoch

def load_and_cache_data(config: ExperimentConfig, cache_dir: str = "data_cache", 
                       rank: int = 0, world_size: int = 1) -> Tuple[List[str], AutoTokenizer, List[int]]:
    """Load and cache tokenized data with distributed support"""
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = f"{cache_dir}/tokenized_data_{config.num_documents}_{config.max_tokens}.pkl"
    
    # Only rank 0 processes data
    if rank == 0:
        if os.path.exists(cache_file):
            print(f"📦 Loading cached data from {cache_file}")
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
        else:
            print(f"🔄 Processing new data...")
            cached_data = _process_new_data(config, cache_file)
    else:
        cached_data = None
    
    # Broadcast data to all ranks if distributed
    if world_size > 1:
        import torch.distributed as dist
        
        # Synchronize
        dist.barrier()
        
        # Broadcast from rank 0
        if rank == 0:
            broadcast_data = {
                'vocab_size': cached_data['tokenizer'].vocab_size,
                'tokens': cached_data['tokens']
            }
        else:
            broadcast_data = None
        
        broadcast_list = [broadcast_data]
        dist.broadcast_object_list(broadcast_list, src=0)
        broadcast_data = broadcast_list[0]
        
        if rank != 0:
            # Reconstruct tokenizer on other ranks
            tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M", token=False)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            cached_data = {
                'texts': [],  # Not needed for training
                'tokenizer': tokenizer,
                'tokens': broadcast_data['tokens']
            }
    
    return cached_data['texts'], cached_data['tokenizer'], cached_data['tokens']

def _process_new_data(config: ExperimentConfig, cache_file: str) -> dict:
    """Process new data and cache it"""
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M", token=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset
    dataset = load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", 
                          split="train", streaming=True, token=False)
    
    # Collect texts
    texts = []
    for i, item in enumerate(dataset):
        if i >= config.num_documents:
            break
        texts.append(item["text"][:3000])  # Truncate long texts
    
    print(f"Loaded {len(texts)} documents")
    
    # Tokenize
    print("Tokenizing texts...")
    all_tokens = []
    for text in tqdm(texts, desc="Tokenizing"):
        tokens = tokenizer.encode(text, add_special_tokens=False)
        all_tokens.extend(tokens)
    
    tokens = all_tokens[:config.max_tokens]
    print(f"Using {len(tokens):,} tokens")
    
    # Cache data
    cached_data = {
        'texts': texts,
        'tokenizer': tokenizer,
        'tokens': tokens
    }
    
    with open(cache_file, 'wb') as f:
        pickle.dump(cached_data, f)
    
    print(f"💾 Cached data to {cache_file}")
    return cached_data

def create_dataloaders(config: ExperimentConfig, rank: int = 0, world_size: int = 1) -> Tuple[DataLoader, DataLoader, AutoTokenizer]:
    """Create train and validation data loaders"""
    # Load data
    texts, tokenizer, tokens = load_and_cache_data(config, rank=rank, world_size=world_size)
    
    # Update config with vocab size
    config.vocab_size = tokenizer.vocab_size
    
    # Create dataset
    dataset = TextTokenDataset(tokens, config.max_seq_len)
    
    # Train/val split
    val_size = len(dataset) // 10
    train_size = len(dataset) - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(config.seed)
    )
    
    # Create samplers for distributed training
    if world_size > 1:
        train_sampler = DistributedSampler(train_dataset, rank, world_size, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, rank, world_size, shuffle=False)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=config.batch_size,
            sampler=train_sampler,
            num_workers=2,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            sampler=val_sampler,
            num_workers=2,
            pin_memory=True
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
    
    if rank == 0:
        print(f"📊 Dataset: {len(train_dataset)} train, {len(val_dataset)} val samples")
    
    return train_loader, val_loader, tokenizer