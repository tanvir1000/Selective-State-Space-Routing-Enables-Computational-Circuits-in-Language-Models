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




def setup_multi_gpu():
    """Configure GPUs"""
    if not torch.cuda.is_available():
        print("⚠️ WARNING: CUDA not available!")
        return 'cpu', None
    
    n_gpus = torch.cuda.device_count()
    print(f"\n{'='*80}")
    print(f"GPU CONFIGURATION")
    print(f"{'='*80}")
    print(f"Available GPUs: {n_gpus}")
    
    for i in range(n_gpus):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name}")
        print(f"    Memory: {props.total_memory / 1024**3:.1f} GB")
    
    torch.cuda.empty_cache()
    gc.collect()
    return 'cuda', n_gpus