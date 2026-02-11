
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
from Circuit_Dectection_Framework.core.DataGenerator import ReasoningExample
warnings.filterwarnings('ignore')
class SparseProbe:
    """
    Trains a sparse linear probe to identify neurons encoding specific reasoning features.

    The SparseProbe uses an L1-regularized logistic regression model (sparse probe)
    to detect which neurons within a layer contribute most to task-specific distinctions.
    This supports mechanistic interpretability by isolating neurons carrying high
    discriminative information about reasoning categories or benchmark tasks.

    Attributes:
        scaler: A sklearn StandardScaler used to normalize neuron activations before training.

    Methods:
        train_probe(
            activations: Dict[str, Dict[int, np.ndarray]],
            labels: Dict[str, str],
            layer: int
        ) -> Dict:
            Trains a sparse logistic regression probe on neuron activations to determine
            which neurons encode task-relevant signals for a given layer.

            Args:
                activations (Dict[str, Dict[int, np.ndarray]]):
                    Dictionary mapping example IDs to a dictionary of layer activations.
                    Each layer’s activation is represented as a 1D numpy array of neuron outputs.
                labels (Dict[str, str]):
                    Mapping of example IDs to class labels (e.g., reasoning vs. control).
                layer (int):
                    The layer index to analyze using the probe.

            Returns:
                Dict:
                    {
                        'accuracy': float,                # Probe classification accuracy
                        'n_important': int,               # Number of neurons above importance threshold
                        'important_neurons': List[int],   # Indices of top neurons by absolute weight
                        'total_neurons': int,             # Total number of neurons in the layer
                        'sparsity': float                 # Proportion of neurons deemed non-important
                    }

            Notes:
                - Uses L1 regularization to encourage sparsity, isolating the smallest set
                  of neurons that explain task variance.
                - Importance is determined by the top 10% of absolute weight magnitudes.
                - This method assumes neuron activations are precomputed and layer-aligned.
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
    
    def train_probe(
        self,
        activations: Dict[str, Dict[int, np.ndarray]],
        labels: Dict[str, str],
        layer: int
    ) -> Dict:
        X = []
        y = []
        for example_id, layer_acts in activations.items():
            X.append(layer_acts[layer])
            y.append(labels[example_id])
        
        X = np.array(X)
        y = np.array(y)
        X_scaled = self.scaler.fit_transform(X)
        
        probe = LogisticRegressionCV(
            penalty='l1',
            solver='saga',
            cv=3,
            max_iter=500,
            n_jobs=1,
            random_state=42
        )
        
        probe.fit(X_scaled, y)
        
        if len(probe.classes_) == 2:
            weights = probe.coef_[0]
        else:
            weights = np.mean(np.abs(probe.coef_), axis=0)
        
        threshold = np.percentile(np.abs(weights), 90)
        important_neurons = np.where(np.abs(weights) > threshold)[0]
        
        return {
            'accuracy': float(probe.score(X_scaled, y)),
            'n_important': int(len(important_neurons)),
            'important_neurons': important_neurons.tolist(),
            'total_neurons': int(len(weights)),
            'sparsity': float(1 - (len(important_neurons) / len(weights)))
        }