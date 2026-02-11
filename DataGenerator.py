import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTNeoXTokenizerFast
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from scipy.stats import mannwhitneyu
from typing import Dict, List, Tuple, Optional
import json
from dataclasses import dataclass, asdict
from tqdm import tqdm
import warnings
import gc
from datasets import load_dataset
from collections import defaultdict
warnings.filterwarnings('ignore')



@dataclass
class ReasoningExample:
    """"
    Represents a single reasoning or control example used for evaluation or benchmarking.
    
    Attributes:
        task_type (str): Category of reasoning task (e.g., arithmetic, comparison, pattern).
        prompt (str): The text prompt or question given to the model.
        expected_steps (List[str]): List of reasoning steps expected in the correct solution.
        is_control (bool): Whether the task is a control (non-reasoning) task.
        ground_truth (Optional[str]): The correct final answer for evaluation.
        benchmark (str): Name of the benchmark or dataset source (e.g., 'custom', 'gsm8k').
        """
    
    
    task_type: str
    prompt: str
    expected_steps: List[str]
    is_control: bool = False
    ground_truth: Optional[str] = None
    benchmark: str = "custom"  # Track which benchmark this is from

class DatasetGenerator:
    """
    Utility class for generating and loading reasoning and control task datasets.
    
    Methods:
        generate_reasoning_tasks(n_per_type: int = 100):
            Creates multi-step reasoning problems across arithmetic, comparison, and sequential categories.
        
        generate_control_tasks(n_per_type: int = 100):
            Generates simple control tasks involving patterns, associations, and factual recall.
        
        load_gsm8k_tasks(n_samples: int = 100):
            Loads reasoning problems from the GSM8K benchmark dataset with structured step annotations.
    
    """
    
    @staticmethod
    def generate_reasoning_tasks(n_per_type: int = 100) -> List[ReasoningExample]:
        """Generate medium-difficulty multi-step reasoning tasks"""
        tasks = []
        np.random.seed(42)
        
        # Category 1: Multi-step arithmetic
        for i in range(n_per_type):
            a = np.random.randint(5, 25)
            b = np.random.randint(5, 25)
            c = np.random.randint(2, 6)
            
            total = a + b
            
            prompt = f"John has {a} apples. Mary gives him {b} more. How many apples total do they have? Show your reasoning step by step."
            
            steps = [
                f"Start: {a} apples",
                f"Add Mary's: {a} + {b} = {total}",
                f"Total: {total} apples"
            ]
            
            tasks.append(ReasoningExample(
                task_type="arithmetic",
                prompt=prompt,
                expected_steps=steps,
                ground_truth=str(total),
                benchmark="custom"
            ))
        
        for i in range(n_per_type):
            age_alice = np.random.randint(15, 35)
            diff = np.random.randint(3, 12)
            age_bob = age_alice + diff
            total_age = age_alice + age_bob
            
            prompt = f"Alice is {age_alice} years old. Bob is {diff} years older than Alice. What is the sum of their ages? Think step by step."
            
            steps = [
                f"Alice: {age_alice} years",
                f"Bob is {diff} years older",
                f"Bob: {age_alice} + {diff} = {age_bob}",
                f"Sum: {age_alice} + {age_bob} = {total_age}"
            ]
            
            tasks.append(ReasoningExample(
                task_type="comparison",
                prompt=prompt,
                expected_steps=steps,
                ground_truth=str(total_age),
                benchmark="custom"
            ))
        
        # Category 3: Sequential operations
        for i in range(n_per_type):
            start = np.random.randint(15, 40)
            add_val = np.random.randint(10, 25)
            multiply_val = np.random.randint(2, 5)
            
            step1 = start + add_val
            result = step1 * multiply_val
            
            prompt = f"Start with {start}. First add {add_val}. Then multiply the result by {multiply_val}. What is the final answer? Show each step."
            
            steps = [
                f"Start: {start}",
                f"Add {add_val}: {start} + {add_val} = {step1}",
                f"Multiply by {multiply_val}: {step1} × {multiply_val} = {result}"
            ]
            
            tasks.append(ReasoningExample(
                task_type="sequential",
                prompt=prompt,
                expected_steps=steps,
                ground_truth=str(result),
                benchmark="custom"
            ))
        
        print(f"✓ Generated {len(tasks)} custom reasoning problems")
        return tasks
    
    @staticmethod
    def generate_control_tasks(n_per_type: int = 100) -> List[ReasoningExample]:
        tasks = []
        
        patterns = [
            ("Red, blue, green, red, blue, what comes next?", "green"),
            ("Monday, Tuesday, Wednesday, Monday, Tuesday, what follows?", "wednesday"),
            ("Apple, banana, cherry, apple, banana, what's next?", "cherry"),
            ("One, two, three, one, two, what number follows?", "three"),
        ]
        
        for i in range(n_per_type):
            prompt, gt = patterns[i % len(patterns)]
            tasks.append(ReasoningExample("pattern", prompt, [], is_control=True, ground_truth=gt, benchmark="custom"))
        
        associations = [
            ("Hot is to cold as day is to what?", "night"),
            ("Cat is to meow as dog is to what?", "bark"),
            ("Pen is to write as knife is to what?", "cut"),
            ("Happy is to sad as light is to what?", "dark"),
        ]
        
        for i in range(n_per_type):
            prompt, gt = associations[i % len(associations)]
            tasks.append(ReasoningExample("association", prompt, [], is_control=True, ground_truth=gt, benchmark="custom"))
        
        facts = [
            ("What is the capital of France?", "paris"),
            ("How many days in a week?", "seven"),
            ("What color is the sky?", "blue"),
            ("How many legs does a spider have?", "eight"),
        ]
        
        for i in range(n_per_type):
            prompt, gt = facts[i % len(facts)]
            tasks.append(ReasoningExample("factual", prompt, [], is_control=True, ground_truth=gt, benchmark="custom"))
        
        return tasks
    
    @staticmethod
    def load_gsm8k_tasks(n_samples: int = 100) -> List[ReasoningExample]:
        """Load GSM8K benchmark tasks"""
        print(f"Loading GSM8K benchmark...")
        try:
            dataset = load_dataset("openai/gsm8k", "main", split="test")
        except:
            dataset = load_dataset("gsm8k", "main", split="test")
        
        tasks = []
        import random
        random.seed(42)
        sampled = random.sample(list(dataset), min(n_samples, len(dataset)))
        
        for problem in sampled:
            question = problem['question']
            answer = problem['answer']
            
            if '####' in answer:
                ground_truth = answer.split('####')[-1].strip()
            else:
                import re
                numbers = re.findall(r'-?\d+\.?\d*', answer)
                ground_truth = numbers[-1] if numbers else ""
            
            steps = [s.strip() for s in answer.split('\n') if s.strip() and '####' not in s]
            
            tasks.append(ReasoningExample(
                task_type="math_word_problem",
                prompt=question,
                expected_steps=steps,
                ground_truth=ground_truth,
                benchmark="gsm8k"
            ))
        
        print(f"✓ Loaded {len(tasks)} GSM8K problems")
        return tasks