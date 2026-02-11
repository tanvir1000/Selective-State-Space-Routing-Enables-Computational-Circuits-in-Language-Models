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




class UniversalActivationExtractor:
    """
    A unified utility for extracting layer activations and gradients across different 
    neural network architectures (e.g., Transformers, Mamba, hybrid models).

    This class dynamically detects model layer structures, registers hooks to capture 
    activations (and optionally gradients), and provides clean extraction of activation 
    patterns for analysis, interpretability, or circuit detection studies.

    Attributes:
        model: The neural network model (supports various architectures).
        tokenizer: The corresponding tokenizer for text input.
        device: The device ('cpu' or 'cuda') used for computation.
        model_id (str): Identifier for the model, used to infer architecture type.
        activations (Dict[int, List[Tensor]]): Stores layer-wise activations.
        gradients (Dict[int, List[Tensor]]): Stores layer-wise gradients if enabled.
        hooks (List): Registered PyTorch hook handles.
        is_mamba (bool): Flag indicating whether the model is a Mamba-type architecture.

    Methods:
        get_layer_modules():
            Automatically detects and returns the model's main layer modules across
            different architectures (Transformer, Mamba, etc.).

        register_hooks(layer_indices: List[int], capture_gradients: bool = False):
            Registers forward hooks on selected layers to capture activations and,
            optionally, backward hooks to record gradients.

        remove_hooks():
            Removes all registered hooks and clears stored activations, gradients,
            and GPU cache to prevent memory leaks.

        extract(text: str, max_length: int = 256) -> Dict[int, np.ndarray]:
            Performs a forward pass on the input text, collects activations from
            registered layers, pools and validates them, and returns a dictionary
            mapping layer indices to NumPy arrays of activations.
    """
    
    def __init__(self, model, tokenizer, device, model_id):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model_id = model_id
        self.activations = {}
        self.gradients = {}
        self.hooks = []
        self.is_mamba = 'mamba' in model_id.lower()
        
    def get_layer_modules(self):
        """Get layers for ANY architecture"""
        patterns = [
            ('model.layers', lambda: self.model.model.layers),
            ('layers', lambda: self.model.layers),
            ('transformer.h', lambda: self.model.transformer.h),
            ('backbone.layers', lambda: self.model.backbone.layers),
            ('backbone.mixer.layers', lambda: self.model.backbone.mixer.layers),
        ]
        
        for pattern_name, getter in patterns:
            try:
                layers = getter()
                if layers and len(layers) > 0:
                    print(f"Found layers via: {pattern_name}")
                    return layers
            except:
                continue
        
        raise ValueError(f"Could not find layers in {type(self.model)}")
    
    def register_hooks(self, layer_indices: List[int], capture_gradients: bool = False):
        """Register hooks for activations and optionally gradients"""
        self.hooks = []
        self.activations = {i: [] for i in layer_indices}
        if capture_gradients:
            self.gradients = {i: [] for i in layer_indices}
        
        def get_activation(layer_idx):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    hidden = output[0]
                else:
                    hidden = output
                
                # Store activation
                self.activations[layer_idx].append(hidden.detach().cpu().clone())
                
                # Register gradient hook if needed
                if capture_gradients and hidden.requires_grad:
                    def grad_hook(grad):
                        self.gradients[layer_idx].append(grad.detach().cpu().clone())
                    hidden.register_hook(grad_hook)
            
            return hook
        
        layers = self.get_layer_modules()
        n_layers = len(layers)
        print(f"Model has {n_layers} layers, registering hooks on: {layer_indices}")
        
        for idx in layer_indices:
            if idx < n_layers:
                hook = layers[idx].register_forward_hook(get_activation(idx))
                self.hooks.append(hook)
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.activations = {}
        self.gradients = {}
        torch.cuda.empty_cache()
        gc.collect()
    
    def extract(self, text: str, max_length: int = 256) -> Dict[int, np.ndarray]:
        """Extract activations"""
        try:
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                max_length=max_length,
                truncation=True,
                padding=False
            ).to(self.device)
        except Exception as e:
            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        for key in self.activations:
            self.activations[key] = []
        
        with torch.no_grad():
            try:
                outputs = self.model(**inputs)
            except Exception as e:
                print(f"Forward pass warning: {e}")
                del inputs
                torch.cuda.empty_cache()
                return {}
        
        result = {}
        for layer_idx, acts in self.activations.items():
            if acts and len(acts) > 0:
                act_tensor = acts[0]
                if act_tensor.dim() == 3:
                    pooled = act_tensor.squeeze(0).mean(dim=0)
                elif act_tensor.dim() == 2:
                    pooled = act_tensor.mean(dim=0)
                else:
                    pooled = act_tensor
                
                # Convert to numpy and validate
                pooled_np = pooled.numpy()
                
                # Clip extreme values to prevent infinity
                pooled_np = np.clip(pooled_np, -1e6, 1e6)
                
                # Check for validity
                if (not np.any(np.isnan(pooled_np)) and 
                    not np.any(np.isinf(pooled_np)) and
                    np.std(pooled_np) > 1e-8):  # Has some variance
                    result[layer_idx] = pooled_np
                # Skip this layer if invalid - don't add to results
        
        del inputs, outputs
        torch.cuda.empty_cache()
        return result