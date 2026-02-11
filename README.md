# Selective-State-Space-Routing-Enables-Computational-Circuits-in-Language-Models

This repository implements a framework for Circuits analysis in large language models


## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/logicsame/Selective-State-Space-Routing-Enables-Computational-Circuits-in-Language-Models.git
```
### 2. Navigate to the Project Directory

```bash
cd Selective-State-Space-Routing-Enables-Computational-Circuits-in-Language-Models
```

### 3. Install Required Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run Model Benchmark
#### Gemma 2b
```bash
python benchmark/gemma_2b+it.py
```
#### llama 3.2 
```bash
python benchmark/llama_3.2_instruct.py
```
#### Phi2 
```bash
python benchmark/phi_2.py
```

#### Qwen2.5-3B
```bash
python benchmark/Qwen2.5-3B.py
```
#### state space 370m
```bash
python benchmark/state_space_370m.py
```
#### state space 790m
```bash
python benchmark/state_space_790m.py
```
#### state space 1.4b
```bash
python benchmark/state_space_1.4b.py
```
#### state space 2.8b
```bash
python benchmark/state_space_2.8b.py
```

> This will execute the all models benchmarking script using the installed framework.

## Contact

For questions or support, contact **MD Azizul Hakim** at `azizulhakim8291@gmail.com`.
