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



class CircuitDetector:
    
    """
    Detects the presence of functional reasoning circuits in neural models by 
    comparing activation similarity patterns between reasoning and control tasks.

    This class provides robust similarity computation and statistical testing to 
    determine whether specific layers exhibit selective clustering of activations 
    during reasoning, indicating specialized computational circuits.

    Attributes:
        significance_level (float): 
            The significance threshold (default = 0.01) for statistical testing 
            during circuit detection.

    Methods:
        compute_similarity(act1: np.ndarray, act2: np.ndarray) -> float:
            Computes cosine similarity between two activation vectors with 
            comprehensive handling for NaN, infinity, zero, or degenerate values.

        detect_with_baseline(
            activations: Dict[str, Dict[int, np.ndarray]],
            reasoning_ids: List[str],
            control_ids: List[str]
        ) -> Dict[int, Dict]:
            Performs circuit detection by comparing intra-group similarity 
            distributions between reasoning and control activations for each 
            model layer.

            The detection process includes:
                - Validation of activation quality (variance, NaN/Inf check)
                - Intra-group and cross-group cosine similarity computation
                - Mann-Whitney U testing for statistical significance
                - Cohen’s d computation for effect size
                - Multi-criteria decision rule based on significance, gap size,
                  relative difference, and bounded similarity thresholds.

            Returns:
                Dict[int, Dict]: 
                    A dictionary mapping each layer index to its analysis results:
                        {
                            'is_circuit': bool,
                            'reasoning_similarity_mean': float,
                            'control_similarity_mean': float,
                            'cross_similarity_mean': float,
                            'p_value': float,
                            'cohens_d': float,
                            'gap': float,
                            'relative_diff': float,
                            'note': Optional[str]
                        }
    """
    
    def __init__(self, significance_level: float = 0.01):
        self.significance_level = significance_level
    
    def compute_similarity(self, act1: np.ndarray, act2: np.ndarray) -> float:
        """Compute cosine similarity with robust NaN, zero, and infinity handling"""
        try:
            # Check for NaN, infinite, or invalid arrays
            if (np.any(np.isnan(act1)) or np.any(np.isnan(act2)) or 
                np.any(np.isinf(act1)) or np.any(np.isinf(act2)) or
                np.all(act1 == 0) or np.all(act2 == 0)):
                return 0.0
            
            # Check for constant vectors (which cause division by zero)
            if np.all(act1 == act1[0]) and np.all(act2 == act2[0]):
                return 1.0 if act1[0] * act2[0] > 0 else -1.0
            
            norm1 = np.linalg.norm(act1)
            norm2 = np.linalg.norm(act2)
            
            # Check for infinite or zero norms
            if (norm1 < 1e-8 or norm2 < 1e-8 or 
                np.isinf(norm1) or np.isinf(norm2)):
                return 0.0
            
            similarity = np.dot(act1, act2) / (norm1 * norm2)
            
            # Handle floating point errors that might push beyond [-1, 1]
            if similarity > 1.0:
                return 1.0
            elif similarity < -1.0:
                return -1.0
            else:
                return similarity
                
        except Exception as e:
            # If anything goes wrong, return neutral similarity
            return 0.0
    
    def detect_with_baseline(
        self, 
        activations: Dict[str, Dict[int, np.ndarray]],
        reasoning_ids: List[str],
        control_ids: List[str]
    ) -> Dict[int, Dict]:
        results = {}
        layer_indices = list(activations[reasoning_ids[0]].keys())
        
        for layer in layer_indices:
            reasoning_sims = []
            control_sims = []
            cross_sims = []
            
            # ============================================================
            # COMPREHENSIVE ACTIVATION VALIDATION
            # ============================================================
            
            def is_valid_activation(act):
                """Check if activation is valid for analysis"""
                return (not np.any(np.isnan(act)) and 
                        not np.any(np.isinf(act)) and
                        np.std(act) > 1e-6 and  # Has meaningful variance
                        not np.all(act == 0))    # Not all zeros
            
            # Filter for valid activations
            valid_reasoning = [rid for rid in reasoning_ids 
                              if layer in activations[rid] and 
                              is_valid_activation(activations[rid][layer])]
            valid_control = [cid for cid in control_ids 
                            if layer in activations[cid] and 
                            is_valid_activation(activations[cid][layer])]
            
            # Check if we have enough valid samples
            if len(valid_reasoning) < 10 or len(valid_control) < 10:
                print(f"⚠️ Layer {layer}: Insufficient valid activations (reasoning={len(valid_reasoning)}, control={len(valid_control)})")
                results[layer] = {
                    'is_circuit': False,
                    'reasoning_similarity_mean': 0.0,
                    'control_similarity_mean': 0.0,
                    'cross_similarity_mean': 0.0,
                    'p_value': 1.0,
                    'cohens_d': 0.0,
                    'note': 'Insufficient valid activations (inf/NaN detected)'
                }
                continue
            
            # ============================================================
            # COMPUTE SIMILARITIES
            # ============================================================
            
            # Within reasoning
            for i in range(len(valid_reasoning)):
                for j in range(i + 1, len(valid_reasoning)):
                    sim = self.compute_similarity(
                        activations[valid_reasoning[i]][layer],
                        activations[valid_reasoning[j]][layer]
                    )
                    reasoning_sims.append(sim)
            
            # Within control
            for i in range(len(valid_control)):
                for j in range(i + 1, len(valid_control)):
                    sim = self.compute_similarity(
                        activations[valid_control[i]][layer],
                        activations[valid_control[j]][layer]
                    )
                    control_sims.append(sim)
            
            # Cross (sample subset for efficiency)
            for r_id in valid_reasoning[:min(20, len(valid_reasoning))]:
                for c_id in valid_control[:min(20, len(valid_control))]:
                    sim = self.compute_similarity(
                        activations[r_id][layer],
                        activations[c_id][layer]
                    )
                    cross_sims.append(sim)
            
            # ============================================================
            # VALIDATE SIMILARITY DISTRIBUTIONS
            # ============================================================
            
            # Check if similarities are valid (not all zero)
            if len(reasoning_sims) == 0 or len(control_sims) == 0:
                print(f"⚠️ Layer {layer}: No valid similarity scores computed")
                results[layer] = {
                    'is_circuit': False,
                    'reasoning_similarity_mean': 0.0,
                    'control_similarity_mean': 0.0,
                    'cross_similarity_mean': 0.0,
                    'p_value': 1.0,
                    'cohens_d': 0.0,
                    'note': 'Failed to compute valid similarities'
                }
                continue
            
            reasoning_mean = float(np.mean(reasoning_sims))
            control_mean = float(np.mean(control_sims))
            
            # Check for degenerate control similarities (all ~0)
            if control_mean < 0.01 and np.std(control_sims) < 0.01:
                print(f"⚠️ Layer {layer}: Control similarities degenerate (mean={control_mean:.6f}, std={np.std(control_sims):.6f})")
                results[layer] = {
                    'is_circuit': False,
                    'reasoning_similarity_mean': reasoning_mean,
                    'control_similarity_mean': control_mean,
                    'cross_similarity_mean': float(np.mean(cross_sims)) if cross_sims else 0.0,
                    'p_value': 1.0,
                    'cohens_d': 0.0,
                    'note': 'Control activations degenerate - all similarities near zero'
                }
                continue
            
           
            
            try:
                # Check for sufficient variance
                reasoning_var = np.var(reasoning_sims)
                control_var = np.var(control_sims)
                combined_std = np.std(reasoning_sims + control_sims)
                
                if combined_std < 1e-8:
                    print(f"⚠️ Layer {layer}: Combined std too small ({combined_std:.6f})")
                    results[layer] = {
                        'is_circuit': False,
                        'reasoning_similarity_mean': reasoning_mean,
                        'control_similarity_mean': control_mean,
                        'cross_similarity_mean': float(np.mean(cross_sims)) if cross_sims else 0.0,
                        'p_value': 1.0,
                        'cohens_d': 0.0,
                        'note': 'Zero variance - cannot compute statistics'
                    }
                    continue
                
                # Mann-Whitney U test
                stat1, p_value1 = mannwhitneyu(reasoning_sims, control_sims, alternative='greater')
                
                # Cohen's d
                cohens_d = (reasoning_mean - control_mean) / combined_std
                
               
                
                # Compute additional metrics
                control_reasoning_gap = reasoning_mean - control_mean
                relative_difference = (reasoning_mean - control_mean) / max(control_mean, 0.01)
                
                # Check if both similarities are too high (distributed computation)
                both_high = (reasoning_mean > 0.80 and control_mean > 0.60)
                
                # Multi-criteria circuit detection
                is_circuit = (
                    p_value1 < self.significance_level and     # p < 0.01 
                    cohens_d > 0.80 and                         # Large effect size (slightly relaxed)
                    control_reasoning_gap > 0.18 and            # Absolute gap > 18% (relaxed from 0.20)
                    relative_difference > 0.33 and              # Relative increase > 33% (relaxed from 0.35)
                    reasoning_mean > 0.65 and                   # Moderate clustering
                    control_mean < 0.70 and                     # Control not too high
                    not both_high                               # NOT both high
                )

                results[layer] = {
                    'is_circuit': bool(is_circuit),
                    'reasoning_similarity_mean': reasoning_mean,
                    'control_similarity_mean': control_mean,
                    'cross_similarity_mean': float(np.mean(cross_sims)) if cross_sims else 0.0,
                    'p_value': float(p_value1),
                    'cohens_d': float(cohens_d),
                    'gap': float(control_reasoning_gap),
                    'relative_diff': float(relative_difference)
                }
                
            except Exception as e:
                print(f"⚠️ Layer {layer}: Statistical test failed: {e}")
                results[layer] = {
                    'is_circuit': False,
                    'reasoning_similarity_mean': reasoning_mean,
                    'control_similarity_mean': control_mean,
                    'cross_similarity_mean': float(np.mean(cross_sims)) if cross_sims else 0.0,
                    'p_value': 1.0,
                    'cohens_d': 0.0,
                    'note': f'Statistical test error: {str(e)}'
                }
        
        return results
