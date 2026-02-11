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
from Circuit_Dectection_Framework.utils.setup_multi_gpu import setup_multi_gpu
from Circuit_Dectection_Framework.utils.load_tokenizer_safe import load_tokenizer_safe
from Circuit_Dectection_Framework.core.UniversalActivationExtractor import UniversalActivationExtractor
from Circuit_Dectection_Framework.core.CircuitDetector import CircuitDetector
from Circuit_Dectection_Framework.core.SparseProbe import SparseProbe
from Circuit_Dectection_Framework.core.MechanisticInterpreter import MechanisticInterpreter
from Circuit_Dectection_Framework.core.GradientCircuitTracer import GradientCircuitTracer
from Circuit_Dectection_Framework.core.DataGenerator import DatasetGenerator
from Circuit_Dectection_Framework.core.GeneralizationTester import GeneralizationTester


def test_single_model(model_name: str, architecture_type: str, n_samples: int = 300):
    """Complete mechanistic analysis pipeline"""
    
    print(f"\n{'='*80}")
    print(f"TESTING: {model_name}")
    print(f"Architecture: {architecture_type}")
    print(f"{'='*80}\n")
    
    device, n_gpus = setup_multi_gpu()
    
    # Load tokenizer
    try:
        tokenizer = load_tokenizer_safe(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        print(f"❌ Failed to load tokenizer: {e}")
        return None
    
    # Load model
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        print(f"✓ Model loaded: {model.num_parameters():,} parameters\n")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None
    
    # ========================================================================
    # STEP 1: Generate tasks from multiple benchmarks
    # ========================================================================
    print("="*80)
    print("STEP 1: GENERATING MULTI-BENCHMARK TASKS")
    print("="*80 + "\n")
    
    generator = DatasetGenerator()
    
    # Custom reasoning tasks
    custom_reasoning = generator.generate_reasoning_tasks(100)
    custom_control = generator.generate_control_tasks(100)
    
    # GSM8K tasks
    try:
        gsm8k_tasks = generator.load_gsm8k_tasks(100)
    except:
        print("⚠️ Could not load GSM8K, using only custom tasks")
        gsm8k_tasks = []
    
    # Combine all reasoning tasks
    all_reasoning_tasks = custom_reasoning + gsm8k_tasks
    all_control_tasks = custom_control
    
    print(f"✓ Total reasoning tasks: {len(all_reasoning_tasks)}")
    print(f"✓ Total control tasks: {len(all_control_tasks)}\n")
    
    # ========================================================================
    # STEP 2: Extract activations
    # ========================================================================
    print("="*80)
    print("STEP 2: EXTRACTING ACTIVATIONS")
    print("="*80 + "\n")
    
    extractor = UniversalActivationExtractor(model, tokenizer, device, model_name)
    
    try:
        layers = extractor.get_layer_modules()
        n_layers = len(layers)
    except Exception as e:
        print(f"❌ Could not determine layers: {e}")
        return None
    
    layer_indices = [0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1]
    extractor.register_hooks(layer_indices)
    
    all_activations = {}
    all_labels = {}
    
    for i, task in enumerate(tqdm(all_reasoning_tasks, desc="Reasoning")):
        task_id = f"reasoning_{i}"
        acts = extractor.extract(task.prompt)
        if acts:
            all_activations[task_id] = acts
            all_labels[task_id] = task.task_type
    
    for i, task in enumerate(tqdm(all_control_tasks, desc="Control")):
        task_id = f"control_{i}"
        acts = extractor.extract(task.prompt)
        if acts:
            all_activations[task_id] = acts
            all_labels[task_id] = task.task_type
    
    
    
    
    extractor.remove_hooks()
    
    reasoning_ids = [k for k in all_activations.keys() if k.startswith('reasoning_')]
    control_ids = [k for k in all_activations.keys() if k.startswith('control_')]
    
    print(f"\n✓ Extracted: {len(reasoning_ids)} reasoning, {len(control_ids)} control\n")
    print("\n📊 ACTIVATION QUALITY CHECK:")
    for layer in layer_indices:
        valid_reasoning = sum(1 for rid in reasoning_ids if layer in all_activations[rid] and not np.any(np.isnan(all_activations[rid][layer])))
        valid_control = sum(1 for cid in control_ids if layer in all_activations[cid] and not np.any(np.isnan(all_activations[cid][layer])))
        print(f"Layer {layer}: {valid_reasoning}/{len(reasoning_ids)} reasoning, {valid_control}/{len(control_ids)} control valid")
    # ========================================================================
    # STEP 3: Detect circuits
    # ========================================================================
    print("="*80)
    print("STEP 3: CIRCUIT DETECTION")
    print("="*80 + "\n")
    
    detector = CircuitDetector()
    circuit_results = detector.detect_with_baseline(
        all_activations, reasoning_ids, control_ids
    )
    
    n_circuits = sum(1 for r in circuit_results.values() if r['is_circuit'])
    circuit_layers = [layer for layer, r in circuit_results.items() if r['is_circuit']]
    
    for layer, res in sorted(circuit_results.items()):
        status = "✓ CIRCUIT" if res['is_circuit'] else "✗ No circuit"
        print(f"Layer {layer}: {status}")
        print(f"  Reasoning: {res['reasoning_similarity_mean']:.4f}")
        print(f"  Control:   {res['control_similarity_mean']:.4f}")
        print(f"  Effect d:  {res['cohens_d']:.4f}")
        print(f"  p-value:   {res['p_value']:.6f}\n")
    
    print(f"🎯 Circuits found: {n_circuits}/{len(layer_indices)}\n")
    
    # ========================================================================
    # STEP 4: Sparse probing to identify circuit neurons
    # ========================================================================
    print("="*80)
    print("STEP 4: IDENTIFYING CIRCUIT NEURONS")
    print("="*80 + "\n")
    
    probe = SparseProbe()
    probe_results = {}
    circuit_neurons = {}
    
    for layer in layer_indices:
        result = probe.train_probe(all_activations, all_labels, layer)
        probe_results[layer] = result
        
        if layer in circuit_layers:
            circuit_neurons[layer] = result['important_neurons']
        
        print(f"Layer {layer}: Acc={result['accuracy']:.3f}, Neurons={result['n_important']}")
    
    print()
    
    # ========================================================================
    # STEP 5: Mechanistic interpretation
    # ========================================================================
    print("="*80)
    print("STEP 5: MECHANISTIC INTERPRETATION")
    print("="*80 + "\n")
    
    interpreter = MechanisticInterpreter(model, tokenizer, device)
    
    # Re-register hooks for interpretation
    extractor.register_hooks(layer_indices)
    
    mechanistic_results = {}
    
    for layer in circuit_layers:
        if layer in circuit_neurons and len(circuit_neurons[layer]) > 0:
            print(f"Analyzing layer {layer}...")
            
            # Analyze top 20 neurons
            top_neurons = circuit_neurons[layer][:20]
            
            # Neuron selectivity analysis
            selectivity = interpreter.analyze_neuron_selectivity(
                top_neurons,
                layer,
                all_reasoning_tasks + all_control_tasks,
                extractor
            )
            
            # Find example activations for top 5 neurons
            example_activations = {}
            for neuron_idx in top_neurons[:5]:
                examples = interpreter.find_example_activations(
                    neuron_idx,
                    layer,
                    all_reasoning_tasks,
                    extractor,
                    n_examples=3
                )
                example_activations[neuron_idx] = examples
            
            mechanistic_results[layer] = {
                'neuron_selectivity': selectivity,
                'example_activations': example_activations,
                'n_neurons_analyzed': len(top_neurons)
            }
    
    extractor.remove_hooks()
    print()
    
    # ========================================================================
    # STEP 6: Gradient-based circuit tracing
    # ========================================================================
    print("="*80)
    print("STEP 6: GRADIENT-BASED CIRCUIT TRACING")
    print("="*80 + "\n")
    
    tracer = GradientCircuitTracer(model, tokenizer, device)
    
    # Trace information flow for sample tasks
    gradient_tracing_results = []
    
    sample_tasks = all_reasoning_tasks[:10]  # Sample 10 tasks
    
    for task in tqdm(sample_tasks, desc="Tracing gradients"):
        flow = tracer.trace_information_flow(
            task,
            circuit_layers,
            circuit_neurons
        )
        gradient_tracing_results.append(flow)
    
    print(f"✓ Traced {len(gradient_tracing_results)} information flow patterns\n")
    
    # ========================================================================
    # STEP 7: Generalization testing
    # ========================================================================
    print("="*80)
    print("STEP 7: CROSS-BENCHMARK GENERALIZATION")
    print("="*80 + "\n")
    
    # Re-register hooks
    extractor.register_hooks(layer_indices)
    
    # Organize tasks by benchmark
    benchmarks = {
        'custom_arithmetic': [t for t in custom_reasoning if t.task_type == 'arithmetic'],
        'custom_comparison': [t for t in custom_reasoning if t.task_type == 'comparison'],
        'custom_sequential': [t for t in custom_reasoning if t.task_type == 'sequential'],
    }
    
    if gsm8k_tasks:
        benchmarks['gsm8k'] = gsm8k_tasks
    
    tester = GeneralizationTester(model, tokenizer, device)
    
    generalization_results = tester.test_cross_benchmark_consistency(
        circuit_neurons,
        benchmarks,
        extractor
    )
    
    extractor.remove_hooks()
    
    # ========================================================================
    # STEP 8: Compile and save complete results
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 8: COMPILING RESULTS")
    print("="*80 + "\n")
    
    # Cleanup
    model_params = model.num_parameters() if hasattr(model, 'num_parameters') else 0
    del model
    torch.cuda.empty_cache()
    gc.collect()
    
    # Compile comprehensive results
    complete_results = {
        'model': model_name,
        'architecture': architecture_type,
        'parameters': model_params, 
        
        # Task information
        'tasks': {
            'n_reasoning_total': len(all_reasoning_tasks),
            'n_control_total': len(all_control_tasks),
            'n_custom_reasoning': len(custom_reasoning),
            'n_gsm8k': len(gsm8k_tasks),
            'task_types': list(set(t.task_type for t in all_reasoning_tasks))
        },
        
        # Circuit detection results
        'circuit_detection': {
            'circuits_found': n_circuits,
            'total_layers_tested': len(layer_indices),
            'circuit_layers': circuit_layers,
            'layer_indices_tested': layer_indices,
            'detailed_results': circuit_results
        },
        
        # Sparse probing results
        'sparse_probing': {
            'probe_results': probe_results,
            'circuit_neurons': circuit_neurons,
            'total_circuit_neurons': sum(len(neurons) for neurons in circuit_neurons.values()),
            'sparsity_by_layer': {
                layer: probe_results[layer]['sparsity'] 
                for layer in circuit_layers if layer in probe_results
            }
        },
        
        # Mechanistic interpretation
        'mechanistic_interpretation': {
            'results_by_layer': mechanistic_results,
            'n_layers_interpreted': len(mechanistic_results)
        },
        
        # Gradient tracing
        'gradient_tracing': {
            'n_tasks_traced': len(gradient_tracing_results),
            'flow_patterns': gradient_tracing_results
        },
        
        # Generalization testing
        'generalization': {
            'cross_benchmark_results': generalization_results,
            'benchmarks_tested': list(benchmarks.keys())
        }
    }
    
    # Save results
    timestamp = __import__('datetime').datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"complete_analysis_{architecture_type}_{model_name.split('/')[-1]}_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(complete_results, f, indent=2)
    
    print(f"✓ Complete results saved to: {filename}")
    
    # Print summary
    print("\n" + "="*80)
    print("ANALYSIS SUMMARY")
    print("="*80)
    print(f"\nCircuit Detection:")
    print(f"  • Circuits found: {n_circuits}/{len(layer_indices)} layers")
    print(f"  • Average effect size: {np.mean([r['cohens_d'] for r in circuit_results.values()]):.3f}")
    print(f"\nSparse Structure:")
    print(f"  • Total circuit neurons: {sum(len(neurons) for neurons in circuit_neurons.values())}")
    print(f"  • Average sparsity: {np.mean([probe_results[l]['sparsity'] for l in circuit_layers if l in probe_results]):.3f}")
    print(f"\nMechanistic Interpretation:")
    print(f"  • Layers analyzed: {len(mechanistic_results)}")
    print(f"  • Neurons characterized: {sum(r['n_neurons_analyzed'] for r in mechanistic_results.values())}")
    print(f"\nGeneralization:")
    print(f"  • Benchmarks tested: {len(benchmarks)}")
    print(f"  • Cross-benchmark consistency measured")
    print("\n" + "="*80 + "\n")
    
    return complete_results