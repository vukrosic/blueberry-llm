#!/usr/bin/env python3
"""
Comprehensive Benchmark Suite for Tiny LLMs
Includes multiple evaluation tasks suitable for small language models
"""

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer
import json
import os
import glob
import re
from tqdm import tqdm
import argparse
from typing import Dict, List, Tuple
import numpy as np
import random
import string

# Import your model classes
try:
    from train_distributed_llm import MinimalLLM, ModelConfig
    print("📦 Using distributed training model")
except ImportError:
    try:
        from train_llm import MinimalLLM, ModelConfig
        print("📦 Using regular training model")
    except ImportError:
        print("❌ Could not import model classes")
        exit(1)

class TinyLLMBenchmark:
    def __init__(self, model_path: str, device: str = 'auto'):
        """Initialize the benchmark suite"""
        self.device = self._setup_device(device)
        self.model, self.config = self._load_model(model_path)
        self.tokenizer = self._load_tokenizer()
        
        print(f"✅ Model loaded on {self.device}")
        print(f"📊 Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def _setup_device(self, device: str) -> torch.device:
        """Setup compute device"""
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return torch.device(device)
    
    def _load_model(self, checkpoint_path: str) -> Tuple[MinimalLLM, ModelConfig]:
        """Load model from checkpoint"""
        config_path = os.path.join(checkpoint_path, "config.json")
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        config = ModelConfig()
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        model = MinimalLLM(config)
        model_path = os.path.join(checkpoint_path, "model.pt")
        if not os.path.exists(model_path):
            model_path = os.path.join(checkpoint_path, "pytorch_model.bin")
        
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
            
        model = model.to(self.device)
        model.eval()
        return model, config
    
    def _load_tokenizer(self):
        """Load tokenizer"""
        tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    
    def _get_logprobs(self, text: str) -> float:
        """Get log probability of a text sequence"""
        tokens = self.tokenizer.encode(text, return_tensors='pt').to(self.device)
        if tokens.size(1) <= 1:
            return float('-inf')
        
        with torch.no_grad():
            logits = self.model(tokens)
            log_probs = F.log_softmax(logits, dim=-1)
            target_tokens = tokens[:, 1:]
            gathered_log_probs = log_probs[0, :-1].gather(1, target_tokens.T).T
            return gathered_log_probs.mean().item()
    
    def _get_completion_logprob(self, context: str, completion: str) -> float:
        """Get log probability of completion given context"""
        full_text = context + " " + completion
        tokens = self.tokenizer.encode(full_text, return_tensors='pt').to(self.device)
        context_tokens = self.tokenizer.encode(context, return_tensors='pt').to(self.device)
        
        if tokens.size(1) <= context_tokens.size(1):
            return float('-inf')
        
        with torch.no_grad():
            logits = self.model(tokens)
            log_probs = F.log_softmax(logits, dim=-1)
            completion_start = context_tokens.size(1) - 1
            completion_tokens = tokens[:, completion_start + 1:]
            completion_logits = log_probs[0, completion_start:-1]
            
            if completion_tokens.size(1) == 0:
                return float('-inf')
            
            token_log_probs = completion_logits.gather(1, completion_tokens.T).T
            return token_log_probs.mean().item()

    # ==================== BENCHMARK 1: LAMBADA ====================
    def evaluate_lambada(self, max_examples: int = 500) -> Dict:
        """
        LAMBADA: Last word prediction task
        Perfect for tiny models - tests basic language understanding
        """
        print("🔍 Loading LAMBADA dataset...")
        try:
            dataset = load_dataset("lambada", split="test")
        except:
            print("❌ Failed to load LAMBADA dataset")
            return {}
        
        if max_examples:
            indices = random.sample(range(len(dataset)), min(max_examples, len(dataset)))
            dataset = [dataset[i] for i in indices]
        
        print(f"📊 Evaluating LAMBADA on {len(dataset)} examples...")
        
        correct = 0
        total = 0
        examples = []
        
        for i, example in enumerate(tqdm(dataset, desc="LAMBADA")):
            text = example['text']
            words = text.split()
            if len(words) < 2:
                continue
                
            context = ' '.join(words[:-1])
            target_word = words[-1].lower()
            
            # Get model's top prediction
            context_tokens = self.tokenizer.encode(context, return_tensors='pt').to(self.device)
            
            with torch.no_grad():
                logits = self.model(context_tokens)
                next_token_logits = logits[0, -1, :]
                top_token_id = torch.argmax(next_token_logits).item()
                predicted_word = self.tokenizer.decode([top_token_id]).strip().lower()
            
            is_correct = predicted_word == target_word
            if is_correct:
                correct += 1
            total += 1
            
            if i < 5:  # Save examples
                examples.append({
                    'context': context,
                    'target': target_word,
                    'predicted': predicted_word,
                    'correct': is_correct
                })
        
        accuracy = correct / total if total > 0 else 0
        return {
            'task': 'LAMBADA',
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'examples': examples
        }

    # ==================== BENCHMARK 2: SIMPLE ARITHMETIC ====================
    def evaluate_arithmetic(self, max_examples: int = 200) -> Dict:
        """
        Simple arithmetic: Addition of small numbers
        Tests basic reasoning capabilities
        """
        print("🔍 Generating arithmetic problems...")
        
        problems = []
        for _ in range(max_examples):
            a = random.randint(1, 20)
            b = random.randint(1, 20)
            problems.append((a, b, a + b))
        
        print(f"📊 Evaluating arithmetic on {len(problems)} examples...")
        
        correct = 0
        total = 0
        examples = []
        
        for i, (a, b, answer) in enumerate(tqdm(problems, desc="Arithmetic")):
            prompt = f"{a} + {b} ="
            
            # Get model prediction
            tokens = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
            
            with torch.no_grad():
                logits = self.model(tokens)
                next_token_logits = logits[0, -1, :]
                
                # Try to get a number
                predicted_token_id = torch.argmax(next_token_logits).item()
                predicted_text = self.tokenizer.decode([predicted_token_id]).strip()
                
                # Extract number from prediction
                try:
                    predicted_num = int(''.join(filter(str.isdigit, predicted_text)))
                    is_correct = predicted_num == answer
                except:
                    is_correct = False
                    predicted_num = None
            
            if is_correct:
                correct += 1
            total += 1
            
            if i < 5:  # Save examples
                examples.append({
                    'problem': prompt,
                    'answer': answer,
                    'predicted': predicted_num,
                    'correct': is_correct
                })
        
        accuracy = correct / total if total > 0 else 0
        return {
            'task': 'Simple Arithmetic',
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'examples': examples
        }

    # ==================== BENCHMARK 3: SENTENCE COMPLETION ====================
    def evaluate_sentence_completion(self, max_examples: int = 100) -> Dict:
        """
        Common sense sentence completion
        Custom dataset of simple completions
        """
        print("🔍 Creating sentence completion tasks...")
        
        # Simple completion tasks
        completions = [
            ("The sun rises in the", ["east", "morning"]),
            ("Ice is made of frozen", ["water", "H2O"]),
            ("Birds can", ["fly", "sing"]),
            ("Fish live in", ["water", "ocean", "sea"]),
            ("At night, we can see the", ["moon", "stars"]),
            ("Cats like to", ["sleep", "play"]),
            ("In winter, it is", ["cold", "snowy"]),
            ("Books are for", ["reading", "learning"]),
            ("Cars need", ["gas", "fuel", "gasoline"]),
            ("People eat food with", ["mouth", "teeth"]),
            ("Rain falls from", ["clouds", "sky"]),
            ("Trees have", ["leaves", "branches"]),
            ("Dogs are", ["animals", "pets"]),
            ("Fire is", ["hot", "warm"]),
            ("Snow is", ["white", "cold"]),
        ]
        
        # Extend with more examples
        while len(completions) < max_examples:
            completions.extend(completions[:max_examples - len(completions)])
        
        completions = completions[:max_examples]
        random.shuffle(completions)
        
        print(f"📊 Evaluating sentence completion on {len(completions)} examples...")
        
        correct = 0
        total = 0
        examples = []
        
        for i, (prompt, valid_answers) in enumerate(tqdm(completions, desc="Sentence Completion")):
            tokens = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
            
            with torch.no_grad():
                logits = self.model(tokens)
                next_token_logits = logits[0, -1, :]
                top_token_id = torch.argmax(next_token_logits).item()
                predicted_word = self.tokenizer.decode([top_token_id]).strip().lower()
            
            is_correct = any(predicted_word in answer.lower() or answer.lower() in predicted_word 
                           for answer in valid_answers)
            
            if is_correct:
                correct += 1
            total += 1
            
            if i < 5:  # Save examples
                examples.append({
                    'prompt': prompt,
                    'valid_answers': valid_answers,
                    'predicted': predicted_word,
                    'correct': is_correct
                })
        
        accuracy = correct / total if total > 0 else 0
        return {
            'task': 'Sentence Completion',
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'examples': examples
        }

    # ==================== BENCHMARK 4: WORD ASSOCIATION ====================
    def evaluate_word_association(self, max_examples: int = 100) -> Dict:
        """
        Simple word association task
        Tests semantic understanding
        """
        print("🔍 Creating word association tasks...")
        
        associations = [
            ("cat", ["dog", "animal", "pet", "meow"]),
            ("hot", ["cold", "warm", "fire", "sun"]),
            ("big", ["small", "large", "huge", "giant"]),
            ("happy", ["sad", "joy", "smile", "glad"]),
            ("red", ["blue", "color", "green", "yellow"]),
            ("fast", ["slow", "quick", "speed", "rapid"]),
            ("day", ["night", "sun", "morning", "light"]),
            ("up", ["down", "high", "above", "top"]),
            ("good", ["bad", "nice", "great", "well"]),
            ("old", ["new", "young", "age", "ancient"]),
        ]
        
        # Extend dataset
        while len(associations) < max_examples:
            associations.extend(associations[:max_examples - len(associations)])
        
        associations = associations[:max_examples]
        random.shuffle(associations)
        
        print(f"📊 Evaluating word association on {len(associations)} examples...")
        
        correct = 0
        total = 0
        examples = []
        
        for i, (word, valid_associations) in enumerate(tqdm(associations, desc="Word Association")):
            prompt = f"{word} is related to"
            tokens = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
            
            with torch.no_grad():
                logits = self.model(tokens)
                next_token_logits = logits[0, -1, :]
                top_token_id = torch.argmax(next_token_logits).item()
                predicted_word = self.tokenizer.decode([top_token_id]).strip().lower()
            
            is_correct = any(predicted_word in assoc.lower() or assoc.lower() in predicted_word 
                           for assoc in valid_associations)
            
            if is_correct:
                correct += 1
            total += 1
            
            if i < 5:  # Save examples
                examples.append({
                    'word': word,
                    'valid_associations': valid_associations,
                    'predicted': predicted_word,
                    'correct': is_correct
                })
        
        accuracy = correct / total if total > 0 else 0
        return {
            'task': 'Word Association',
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'examples': examples
        }

    # ==================== BENCHMARK 5: SIMPLE QA ====================
    def evaluate_simple_qa(self, max_examples: int = 50) -> Dict:
        """
        Very simple question answering
        Basic factual questions
        """
        print("🔍 Creating simple QA tasks...")
        
        qa_pairs = [
            ("What color is the sky?", ["blue", "light blue"]),
            ("How many legs does a cat have?", ["four", "4"]),
            ("What do we use to see?", ["eyes", "eye"]),
            ("What season comes after winter?", ["spring"]),
            ("What do bees make?", ["honey"]),
            ("What do we breathe?", ["air", "oxygen"]),
            ("What is 2 + 2?", ["4", "four"]),
            ("What do fish live in?", ["water", "ocean", "sea"]),
            ("What do we use to hear?", ["ears", "ear"]),
            ("What color is grass?", ["green"]),
        ]
        
        # Extend dataset
        while len(qa_pairs) < max_examples:
            qa_pairs.extend(qa_pairs[:max_examples - len(qa_pairs)])
        
        qa_pairs = qa_pairs[:max_examples]
        random.shuffle(qa_pairs)
        
        print(f"📊 Evaluating simple QA on {len(qa_pairs)} examples...")
        
        correct = 0
        total = 0
        examples = []
        
        for i, (question, valid_answers) in enumerate(tqdm(qa_pairs, desc="Simple QA")):
            tokens = self.tokenizer.encode(question, return_tensors='pt').to(self.device)
            
            with torch.no_grad():
                logits = self.model(tokens)
                next_token_logits = logits[0, -1, :]
                top_token_id = torch.argmax(next_token_logits).item()
                predicted_word = self.tokenizer.decode([top_token_id]).strip().lower()
            
            is_correct = any(predicted_word in answer.lower() or answer.lower() in predicted_word 
                           for answer in valid_answers)
            
            if is_correct:
                correct += 1
            total += 1
            
            if i < 3:  # Save examples
                examples.append({
                    'question': question,
                    'valid_answers': valid_answers,
                    'predicted': predicted_word,
                    'correct': is_correct
                })
        
        accuracy = correct / total if total > 0 else 0
        return {
            'task': 'Simple QA',
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'examples': examples
        }

    # ==================== BENCHMARK 6: PIQA ====================
    def evaluate_piqa(self, max_examples: int = 200) -> Dict:
        """
        PIQA: Physical Interaction QA
        Tests understanding of how objects work in the real world
        2-choice QA, baseline is 50%
        """
        print("🔍 Loading PIQA dataset...")
        try:
            dataset = load_dataset("piqa", split="validation", trust_remote_code=True)
        except Exception as e:
            try:
                # Try alternative loading method
                dataset = load_dataset("ybisk/piqa", split="validation")
            except Exception as e2:
                print(f"❌ Failed to load PIQA dataset: {e}")
                print(f"❌ Alternative method also failed: {e2}")
                print("🔄 Using fallback PIQA examples...")
                return self._evaluate_piqa_fallback(max_examples)
        
        if max_examples:
            indices = random.sample(range(len(dataset)), min(max_examples, len(dataset)))
            sample_data = [dataset[i] for i in indices]
        else:
            sample_data = list(dataset)
        
        print(f"📊 Evaluating PIQA on {len(sample_data)} examples...")
        
        correct = 0
        total = 0
        examples = []
        
        for i, example in enumerate(tqdm(sample_data, desc="PIQA")):
            goal = example['goal']
            sol1 = example['sol1']
            sol2 = example['sol2']
            correct_idx = int(example['label'])
            
            # Calculate log probabilities for both solutions
            context1 = f"{goal} {sol1}"
            context2 = f"{goal} {sol2}"
            
            logprob1 = self._get_logprobs(context1)
            logprob2 = self._get_logprobs(context2)
            
            # Predict the solution with higher log probability
            predicted_idx = 0 if logprob1 > logprob2 else 1
            is_correct = predicted_idx == correct_idx
            
            if is_correct:
                correct += 1
            total += 1
            
            if i < 5:  # Save examples
                examples.append({
                    'goal': goal,
                    'sol1': sol1,
                    'sol2': sol2,
                    'correct_idx': correct_idx,
                    'predicted_idx': predicted_idx,
                    'logprob1': logprob1,
                    'logprob2': logprob2,
                    'correct': is_correct
                })
        
        accuracy = correct / total if total > 0 else 0
        return {
            'task': 'PIQA',
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'baseline': 0.5,
            'examples': examples
        }

    def _evaluate_piqa_fallback(self, max_examples: int = 50) -> Dict:
        """Fallback PIQA evaluation with manually created examples"""
        piqa_examples = [
            {
                'goal': 'How do you separate egg yolks from whites?',
                'sol1': 'Use a strainer to separate them.',
                'sol2': 'Use a slotted spoon to separate them.',
                'label': 0  # strainer is better
            },
            {
                'goal': 'How do you remove a splinter from your finger?',
                'sol1': 'Use tweezers to carefully pull it out.',
                'sol2': 'Use a hammer to knock it out.',
                'label': 0  # tweezers is correct
            },
            {
                'goal': 'How do you water plants when you are away?',
                'sol1': 'Set up a drip irrigation system.',
                'sol2': 'Leave them in direct sunlight.',
                'label': 0  # irrigation system is correct
            },
            {
                'goal': 'How do you keep food cold during a picnic?',
                'sol1': 'Use a cooler with ice packs.',
                'sol2': 'Wrap it in aluminum foil.',
                'label': 0  # cooler is correct
            },
            {
                'goal': 'How do you remove gum from hair?',
                'sol1': 'Apply peanut butter and work it out.',
                'sol2': 'Pull it out with your hands.',
                'label': 0  # peanut butter method works
            },
        ]
        
        # Extend examples if needed
        while len(piqa_examples) < max_examples:
            piqa_examples.extend(piqa_examples[:max_examples - len(piqa_examples)])
        
        piqa_examples = piqa_examples[:max_examples]
        
        print(f"📊 Evaluating PIQA (fallback) on {len(piqa_examples)} examples...")
        
        correct = 0
        total = 0
        examples = []
        
        for i, example in enumerate(tqdm(piqa_examples, desc="PIQA Fallback")):
            goal = example['goal']
            sol1 = example['sol1']
            sol2 = example['sol2']
            correct_idx = example['label']
            
            # Calculate log probabilities for both solutions
            context1 = f"{goal} {sol1}"
            context2 = f"{goal} {sol2}"
            
            logprob1 = self._get_logprobs(context1)
            logprob2 = self._get_logprobs(context2)
            
            # Predict the solution with higher log probability
            predicted_idx = 0 if logprob1 > logprob2 else 1
            is_correct = predicted_idx == correct_idx
            
            if is_correct:
                correct += 1
            total += 1
            
            if i < 3:  # Save examples
                examples.append({
                    'goal': goal,
                    'sol1': sol1,
                    'sol2': sol2,
                    'correct_idx': correct_idx,
                    'predicted_idx': predicted_idx,
                    'logprob1': logprob1,
                    'logprob2': logprob2,
                    'correct': is_correct
                })
        
        accuracy = correct / total if total > 0 else 0
        return {
            'task': 'PIQA (Fallback)',
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'baseline': 0.5,
            'examples': examples
        }

    def _evaluate_siqa_fallback(self, max_examples: int = 30) -> Dict:
        """Fallback SIQA evaluation with manually created examples"""
        siqa_examples = [
            {
                'context': 'Taylor felt embarrassed after the meeting.',
                'question': 'What was a likely reason?',
                'answerA': "Taylor's phone went off during the presentation.",
                'answerB': "Taylor's boss praised their work.",
                'answerC': "Taylor's proposal was accepted.",
                'label': 1  # phone going off is embarrassing
            },
            {
                'context': 'Alex avoided eye contact with their friend.',
                'question': 'Why might Alex be acting this way?',
                'answerA': "Alex is feeling guilty about something.",
                'answerB': "Alex is very happy today.",
                'answerC': "Alex got a promotion at work.",
                'label': 1  # guilt causes avoidance
            },
            {
                'context': 'Sam brought flowers to their neighbor.',
                'question': 'What was Sam likely trying to do?',
                'answerA': "Apologize for being too loud last night.",
                'answerB': "Show off their gardening skills.",
                'answerC': "Ask for money.",
                'label': 1  # flowers are often apologies
            },
        ]
        
        # Extend examples if needed
        while len(siqa_examples) < max_examples:
            siqa_examples.extend(siqa_examples[:max_examples - len(siqa_examples)])
        
        siqa_examples = siqa_examples[:max_examples]
        
        print(f"📊 Evaluating SIQA (fallback) on {len(siqa_examples)} examples...")
        
        correct = 0
        total = 0
        examples = []
        
        for i, example in enumerate(tqdm(siqa_examples, desc="SIQA Fallback")):
            context = example['context']
            question = example['question']
            answerA = example['answerA']
            answerB = example['answerB']
            answerC = example['answerC']
            correct_idx = example['label'] - 1  # Convert 1,2,3 to 0,1,2
            
            answers = [answerA, answerB, answerC]
            
            # Calculate log probabilities for each answer
            log_probs = []
            for answer in answers:
                full_context = f"{context} {question} {answer}"
                logprob = self._get_logprobs(full_context)
                log_probs.append(logprob)
            
            # Predict the answer with highest log probability
            predicted_idx = max(range(len(log_probs)), key=lambda i: log_probs[i])
            is_correct = predicted_idx == correct_idx
            
            if is_correct:
                correct += 1
            total += 1
            
            if i < 3:  # Save examples
                examples.append({
                    'context': context,
                    'question': question,
                    'answers': answers,
                    'correct_idx': correct_idx,
                    'predicted_idx': predicted_idx,
                    'log_probs': log_probs,
                    'correct': is_correct
                })
        
        accuracy = correct / total if total > 0 else 0
        return {
            'task': 'SIQA (Fallback)',
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'baseline': 0.333,
            'examples': examples
        }

    # ==================== BENCHMARK 7: SIQA ====================
    def evaluate_siqa(self, max_examples: int = 200) -> Dict:
        """
        SIQA: Social Interaction QA
        Tests social commonsense and reasoning about people's motivations
        3-choice QA, baseline is 33%
        """
        print("🔍 Loading SIQA dataset...")
        try:
            dataset = load_dataset("social_i_qa", split="validation", trust_remote_code=True)
        except Exception as e:
            try:
                # Try alternative loading method
                dataset = load_dataset("allenai/social_i_qa", split="validation")
            except Exception as e2:
                print(f"❌ Failed to load SIQA dataset: {e}")
                print(f"❌ Alternative method also failed: {e2}")
                print("🔄 Using fallback SIQA examples...")
                return self._evaluate_siqa_fallback(max_examples)
        
        if max_examples:
            indices = random.sample(range(len(dataset)), min(max_examples, len(dataset)))
            sample_data = [dataset[i] for i in indices]
        else:
            sample_data = list(dataset)
        
        print(f"📊 Evaluating SIQA on {len(sample_data)} examples...")
        
        correct = 0
        total = 0
        examples = []
        
        for i, example in enumerate(tqdm(sample_data, desc="SIQA")):
            context = example['context']
            question = example['question']
            answerA = example['answerA']
            answerB = example['answerB']
            answerC = example['answerC']
            correct_idx = int(example['label']) - 1  # Convert 1,2,3 to 0,1,2
            
            answers = [answerA, answerB, answerC]
            
            # Calculate log probabilities for each answer
            log_probs = []
            for answer in answers:
                full_context = f"{context} {question} {answer}"
                logprob = self._get_logprobs(full_context)
                log_probs.append(logprob)
            
            # Predict the answer with highest log probability
            predicted_idx = max(range(len(log_probs)), key=lambda i: log_probs[i])
            is_correct = predicted_idx == correct_idx
            
            if is_correct:
                correct += 1
            total += 1
            
            if i < 5:  # Save examples
                examples.append({
                    'context': context,
                    'question': question,
                    'answers': answers,
                    'correct_idx': correct_idx,
                    'predicted_idx': predicted_idx,
                    'log_probs': log_probs,
                    'correct': is_correct
                })
        
        accuracy = correct / total if total > 0 else 0
        return {
            'task': 'SIQA',
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'baseline': 0.333,
            'examples': examples
        }

    # ==================== BENCHMARK 8: SQUAD 1.1 (SIMPLIFIED) ====================
    def evaluate_squad_simple(self, max_examples: int = 100) -> Dict:
        """
        Simplified SQuAD 1.1: Reading Comprehension
        Tests ability to extract answers from context
        We'll use a simplified version that checks if key words appear in the prediction
        """
        print("🔍 Loading SQuAD 1.1 dataset...")
        try:
            dataset = load_dataset("squad", split="validation")
        except Exception as e:
            print(f"❌ Failed to load SQuAD dataset: {e}")
            return {'error': str(e)}
        
        if max_examples:
            indices = random.sample(range(len(dataset)), min(max_examples, len(dataset)))
            sample_data = [dataset[i] for i in indices]
        else:
            sample_data = list(dataset)
        
        print(f"📊 Evaluating SQuAD (simplified) on {len(sample_data)} examples...")
        
        exact_matches = 0
        partial_matches = 0
        total = 0
        examples = []
        
        for i, example in enumerate(tqdm(sample_data, desc="SQuAD")):
            context = example['context']
            question = example['question']
            answers = example['answers']['text']
            
            if not answers:  # Skip if no answers
                continue
            
            # Use the first answer as ground truth
            ground_truth = answers[0].lower().strip()
            
            # Create prompt for the model
            prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
            
            # Get model's prediction (next few tokens)
            tokens = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
            
            with torch.no_grad():
                # Generate a few tokens as the answer
                generated_tokens = []
                current_tokens = tokens
                
                for _ in range(10):  # Generate up to 10 tokens
                    logits = self.model(current_tokens)
                    next_token_logits = logits[0, -1, :]
                    next_token_id = torch.argmax(next_token_logits).item()
                    
                    # Stop if we hit end of sequence or punctuation
                    next_token = self.tokenizer.decode([next_token_id]).strip()
                    if next_token_id == self.tokenizer.eos_token_id or next_token in ['.', '!', '?', '\n']:
                        break
                    
                    generated_tokens.append(next_token_id)
                    current_tokens = torch.cat([current_tokens, torch.tensor([[next_token_id]]).to(self.device)], dim=1)
                
                if generated_tokens:
                    predicted_answer = self.tokenizer.decode(generated_tokens).strip().lower()
                else:
                    predicted_answer = ""
            
            # Check for exact match
            exact_match = predicted_answer == ground_truth
            
            # Check for partial match (key words present)
            ground_truth_words = set(ground_truth.split())
            predicted_words = set(predicted_answer.split())
            partial_match = len(ground_truth_words.intersection(predicted_words)) > 0
            
            if exact_match:
                exact_matches += 1
                partial_matches += 1
            elif partial_match:
                partial_matches += 1
            
            total += 1
            
            if i < 3:  # Save examples
                examples.append({
                    'context': context[:200] + "..." if len(context) > 200 else context,
                    'question': question,
                    'ground_truth': ground_truth,
                    'predicted': predicted_answer,
                    'exact_match': exact_match,
                    'partial_match': partial_match
                })
        
        exact_match_score = exact_matches / total if total > 0 else 0
        partial_match_score = partial_matches / total if total > 0 else 0
        
        return {
            'task': 'SQuAD 1.1 (Simplified)',
            'exact_match_score': exact_match_score,
            'partial_match_score': partial_match_score,
            'exact_matches': exact_matches,
            'partial_matches': partial_matches,
            'total': total,
            'baseline': 0.0,  # Very low baseline for exact match
            'examples': examples
        }

    # ==================== RUN ALL BENCHMARKS ====================
    def run_all_benchmarks(self, save_results: bool = True) -> Dict:
        """Run all benchmark tasks"""
        print("🚀 Running Tiny LLM Benchmark Suite")
        print("=" * 60)
        
        all_results = {}
        
        # Run each benchmark
        benchmarks = [
            ('lambada', lambda: self.evaluate_lambada(200)),
            ('arithmetic', lambda: self.evaluate_arithmetic(100)),
            ('sentence_completion', lambda: self.evaluate_sentence_completion(50)),
            ('word_association', lambda: self.evaluate_word_association(50)),
            ('simple_qa', lambda: self.evaluate_simple_qa(30)),
            ('piqa', lambda: self.evaluate_piqa(100)),
            ('siqa', lambda: self.evaluate_siqa(100)),
            ('squad', lambda: self.evaluate_squad_simple(50)),
        ]
        
        for name, benchmark_func in benchmarks:
            print(f"\n🔄 Running {name.replace('_', ' ').title()}...")
            try:
                result = benchmark_func()
                all_results[name] = result
                print(f"✅ {result['task']}: {result['accuracy']:.3f} ({result['correct']}/{result['total']})")
            except Exception as e:
                print(f"❌ Failed {name}: {e}")
                all_results[name] = {'error': str(e)}
        
        # Calculate overall score
        valid_results = []
        accuracy_scores = []
        
        for result in all_results.values():
            if 'error' in result:
                continue
            elif 'accuracy' in result:
                valid_results.append(result)
                accuracy_scores.append(result['accuracy'])
            elif 'exact_match_score' in result:
                # For SQuAD, use exact match score as the primary metric
                valid_results.append(result)
                accuracy_scores.append(result['exact_match_score'])
        
        if accuracy_scores:
            overall_accuracy = sum(accuracy_scores) / len(accuracy_scores)
            all_results['overall'] = {
                'average_accuracy': overall_accuracy,
                'num_tasks': len(valid_results),
                'successful_tasks': len(accuracy_scores)
            }
        
        # Save results
        if save_results:
            results_file = "tiny_llm_benchmark_results.json"
            with open(results_file, 'w') as f:
                # Convert numpy types for JSON compatibility
                def make_json_safe(obj):
                    if isinstance(obj, dict):
                        return {k: make_json_safe(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [make_json_safe(item) for item in obj]
                    elif isinstance(obj, (bool, np.bool_)):
                        return bool(obj)
                    elif isinstance(obj, (np.int64, np.int32)):
                        return int(obj)
                    elif isinstance(obj, (np.float64, np.float32)):
                        return float(obj)
                    else:
                        return obj
                
                json_safe_results = make_json_safe(all_results)
                json.dump(json_safe_results, f, indent=2)
            print(f"\n💾 Results saved to {results_file}")
        
        return all_results
    
    def print_detailed_results(self, results: Dict):
        """Print detailed benchmark results"""
        print("\n" + "=" * 80)
        print("🎯 TINY LLM BENCHMARK RESULTS")
        print("=" * 80)
        
        if 'overall' in results:
            overall = results['overall']
            print(f"📊 Overall Average Accuracy: {overall['average_accuracy']:.3f}")
            print(f"📈 Tasks Completed: {overall['num_tasks']}")
        
        print("\n📋 Individual Task Results:")
        for task_name, result in results.items():
            if task_name == 'overall' or 'error' in result:
                continue
            
            print(f"\n🔹 {result['task']}:")
            
            # Handle different result formats
            if 'accuracy' in result:
                accuracy = result['accuracy']
                print(f"   Accuracy: {accuracy:.3f} ({result['correct']}/{result['total']})")
                if 'baseline' in result:
                    baseline = result['baseline']
                    improvement = (accuracy - baseline) / baseline * 100 if baseline > 0 else 0
                    print(f"   Baseline: {baseline:.3f}, Improvement: {improvement:+.1f}%")
            elif 'exact_match_score' in result:
                # SQuAD results
                print(f"   Exact Match: {result['exact_match_score']:.3f} ({result['exact_matches']}/{result['total']})")
                print(f"   Partial Match: {result['partial_match_score']:.3f} ({result['partial_matches']}/{result['total']})")
            
            if 'examples' in result and result['examples']:
                print("   Sample Examples:")
                for i, example in enumerate(result['examples'][:2]):
                    if 'correct' in example:
                        status = "✅" if example['correct'] else "❌"
                    elif 'exact_match' in example:
                        status = "✅" if example['exact_match'] else ("🔶" if example.get('partial_match') else "❌")
                    else:
                        status = "❓"
                    
                    if 'question' in example and 'valid_answers' in example:
                        print(f"     {status} Q: {example['question']}")
                        print(f"        A: {example['predicted']} (expected: {example['valid_answers']})")
                    elif 'problem' in example:
                        print(f"     {status} {example['problem']} {example['predicted']} (expected: {example['answer']})")
                    elif 'prompt' in example:
                        print(f"     {status} {example['prompt']} → {example['predicted']}")
                    elif 'goal' in example:  # PIQA
                        print(f"     {status} Goal: {example['goal']}")
                        print(f"        Predicted: {example['predicted_idx']+1}, Correct: {example['correct_idx']+1}")
                    elif 'context' in example and 'answers' in example:  # SIQA
                        print(f"     {status} Context: {example['context'][:50]}...")
                        print(f"        Question: {example['question']}")
                        print(f"        Predicted: {example['answers'][example['predicted_idx']]}")
                    elif 'ground_truth' in example:  # SQuAD
                        print(f"     {status} Q: {example['question']}")
                        print(f"        Predicted: '{example['predicted']}' | Expected: '{example['ground_truth']}'")
                    elif 'target' in example:  # LAMBADA
                        print(f"     {status} ...{example['context'][-30:]} → {example['predicted']} (expected: {example['target']})")
        
        print("\n" + "=" * 80)

def find_checkpoints(checkpoint_dir: str = "checkpoints") -> List[str]:
    """Find available model checkpoints"""
    if not os.path.exists(checkpoint_dir):
        return []
    
    checkpoints = (glob.glob(os.path.join(checkpoint_dir, "checkpoint-*")) + 
                  glob.glob(os.path.join(checkpoint_dir, "checkpoint_step_*")))
    
    def extract_step(path):
        basename = os.path.basename(path)
        if 'checkpoint-' in basename:
            step_part = basename.split('-')[-1]
            return int(step_part) if step_part.isdigit() else 0
        elif 'checkpoint_step_' in basename:
            return int(basename.split('_')[-1])
        return 0
    
    checkpoints.sort(key=extract_step)
    return checkpoints

def main():
    parser = argparse.ArgumentParser(description="Tiny LLM Benchmark Suite")
    parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument("--device", type=str, default="auto", help="Device to use (auto, cuda, cpu)")
    parser.add_argument("--task", type=str, choices=['all', 'lambada', 'arithmetic', 'sentence', 'word', 'qa', 'piqa', 'siqa', 'squad'], 
                       default='all', help="Specific task to run")
    
    args = parser.parse_args()
    
    # Find checkpoint if not specified
    if not args.checkpoint:
        checkpoints = find_checkpoints()
        if not checkpoints:
            print("❌ No checkpoints found in 'checkpoints/' directory")
            return
        
        print("📁 Available checkpoints:")
        for i, checkpoint in enumerate(checkpoints):
            basename = os.path.basename(checkpoint)
            if 'checkpoint-' in basename:
                step = basename.split('-')[-1]
            elif 'checkpoint_step_' in basename:
                step = basename.split('_')[-1]
            else:
                step = "unknown"
            print(f"  {i+1}. Step {step} ({checkpoint})")
        
        while True:
            try:
                choice = int(input(f"\nChoose checkpoint (1-{len(checkpoints)}): ")) - 1
                if 0 <= choice < len(checkpoints):
                    args.checkpoint = checkpoints[choice]
                    break
                else:
                    print("Invalid choice, try again.")
            except (ValueError, KeyboardInterrupt):
                print("\nExiting...")
                return
    
    print(f"🚀 Starting Tiny LLM Benchmark Suite...")
    print(f"📂 Checkpoint: {args.checkpoint}")
    
    try:
        # Initialize benchmark
        benchmark = TinyLLMBenchmark(args.checkpoint, device=args.device)
        
        # Run benchmarks
        if args.task == 'all':
            results = benchmark.run_all_benchmarks()
            benchmark.print_detailed_results(results)
        else:
            # Run specific task
            task_map = {
                'lambada': benchmark.evaluate_lambada,
                'arithmetic': benchmark.evaluate_arithmetic,
                'sentence': benchmark.evaluate_sentence_completion,
                'word': benchmark.evaluate_word_association,
                'qa': benchmark.evaluate_simple_qa,
                'piqa': benchmark.evaluate_piqa,
                'siqa': benchmark.evaluate_siqa,
                'squad': benchmark.evaluate_squad_simple
            }
            
            if args.task in task_map:
                result = task_map[args.task]()
                print(f"\n🎯 {result['task']} Results:")
                print(f"📊 Accuracy: {result['accuracy']:.3f} ({result['correct']}/{result['total']})")
                
                if 'examples' in result:
                    print("\n🔍 Sample Examples:")
                    for example in result['examples'][:3]:
                        status = "✅" if example['correct'] else "❌"
                        print(f"  {status} {example}")
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()