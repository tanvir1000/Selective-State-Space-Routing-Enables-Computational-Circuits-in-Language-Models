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




def load_tokenizer_safe(model_name: str):
    """Load tokenizer with fallbacks"""
    print(f"Loading tokenizer for {model_name}...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        print("✓ Loaded tokenizer successfully")
        return tokenizer
    except Exception as e:
        print(f"AutoTokenizer failed: {str(e)[:100]}")
    
    if 'mamba' in model_name.lower():
        try:
            tokenizer = GPTNeoXTokenizerFast.from_pretrained("EleutherAI/gpt-neox-20b")
            print("✓ Loaded GPT-NeoX tokenizer")
            return tokenizer
        except:
            pass
    
    raise ValueError(f"Could not load tokenizer for {model_name}")

