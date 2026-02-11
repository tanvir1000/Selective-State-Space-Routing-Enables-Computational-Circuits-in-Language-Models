
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
from Circuit_Dectection_Framework.core.UniversalActivationExtractor import UniversalActivationExtractor
from Circuit_Dectection_Framework.core.DataGenerator import ReasoningExample

class MechanisticInterpreter:
    """
    Provides tools for mechanistic interpretability analysis by examining how 
    individual neurons respond to different reasoning and control tasks.

    This class quantifies neuron selectivity, identifies task-dependent activation 
    patterns, and retrieves examples that maximally activate specific neurons — 
    enabling analysis of circuit-level functional organization within neural models.

    Attributes:
        model: The neural network model under investigation.
        tokenizer: The tokenizer used to process text inputs for the model.
        device: The computation device ('cpu' or 'cuda') for activation extraction.

    Methods:
        analyze_neuron_selectivity(
            neuron_indices: List[int],
            layer_idx: int,
            tasks: List[ReasoningExample],
            extractor: UniversalActivationExtractor
        ) -> Dict:
            Computes neuron-level selectivity by analyzing mean activation responses 
            across different reasoning task categories. Returns per-neuron summaries 
            of selectivity strength and task preference.

            Returns:
                Dict:
                    {
                        neuron_idx: {
                            'most_selective_for': str,        # Task type with highest mean activation
                            'selectivity_strength': float,    # Mean activation for the preferred task
                            'all_task_responses': Dict[str, Dict[str, float]]  # Mean/std/max per task
                        }
                    }

        find_example_activations(
            neuron_idx: int,
            layer_idx: int,
            tasks: List[ReasoningExample],
            extractor: UniversalActivationExtractor,
            n_examples: int = 5
        ) -> List[Dict]:
            Identifies the top examples (tasks) that maximally activate a given neuron 
            within a specified layer, providing insights into neuron interpretability 
            and semantic alignment.

            Returns:
                List[Dict]:
                    A list of dictionaries, each containing:
                        {
                            'activation': float,     # Activation value
                            'task_type': str,        # Task type
                            'prompt': str,           # Task prompt (truncated)
                            'is_control': bool       # Whether the task is control or reasoning
                        }
    """
    
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
    
    def analyze_neuron_selectivity(
        self,
        neuron_indices: List[int],
        layer_idx: int,
        tasks: List[ReasoningExample],
        extractor: UniversalActivationExtractor
    ) -> Dict:
        """
        Determine what each neuron is selective for
        """
        print(f"\nAnalyzing neuron selectivity in layer {layer_idx}...")
        
        neuron_task_activations = defaultdict(lambda: defaultdict(list))
        
        # Extract activations for each task type
        for task in tqdm(tasks, desc=f"Layer {layer_idx} selectivity"):
            acts = extractor.extract(task.prompt)
            if layer_idx in acts:
                activation_vector = acts[layer_idx]
                for neuron_idx in neuron_indices:
                    if neuron_idx < len(activation_vector):
                        neuron_task_activations[neuron_idx][task.task_type].append(
                            float(activation_vector[neuron_idx])
                        )
        
        # Compute selectivity for each neuron
        neuron_selectivity = {}
        for neuron_idx in neuron_indices:
            task_means = {}
            for task_type, activations in neuron_task_activations[neuron_idx].items():
                if activations:
                    task_means[task_type] = {
                        'mean': float(np.mean(activations)),
                        'std': float(np.std(activations)),
                        'max': float(np.max(activations)),
                        'n_samples': len(activations)
                    }
            
            # Find most selective task
            if task_means:
                max_task = max(task_means.items(), key=lambda x: x[1]['mean'])
                neuron_selectivity[neuron_idx] = {
                    'most_selective_for': max_task[0],
                    'selectivity_strength': max_task[1]['mean'],
                    'all_task_responses': task_means
                }
        
        return neuron_selectivity
    
    def find_example_activations(
        self,
        neuron_idx: int,
        layer_idx: int,
        tasks: List[ReasoningExample],
        extractor: UniversalActivationExtractor,
        n_examples: int = 5
    ) -> List[Dict]:
        """Find examples that maximally activate a specific neuron"""
        
        activations_with_tasks = []
        
        for task in tasks:
            acts = extractor.extract(task.prompt)
            if layer_idx in acts:
                activation_vector = acts[layer_idx]
                if neuron_idx < len(activation_vector):
                    activations_with_tasks.append({
                        'activation': float(activation_vector[neuron_idx]),
                        'task_type': task.task_type,
                        'prompt': task.prompt[:200],  # Truncate for storage
                        'is_control': task.is_control
                    })
        
        # Sort by activation strength
        activations_with_tasks.sort(key=lambda x: x['activation'], reverse=True)
        
        return activations_with_tasks[:n_examples]
