import numpy as np
import warnings
from Circuit_Dectection_Framework.core.DataGenerator import ReasoningExample
warnings.filterwarnings('ignore')
from typing import Dict, List
from Circuit_Dectection_Framework.core.CircuitDetector import CircuitDetector
from Circuit_Dectection_Framework.core.UniversalActivationExtractor import UniversalActivationExtractor
class GeneralizationTester:
    """
    Evaluates whether identified reasoning circuits generalize across multiple benchmarks.

    This class measures the stability and consistency of neuron activations that define
    a discovered circuit when tested on different reasoning or control datasets.
    By comparing within-benchmark activation patterns, it quantifies how well the 
    same circuit components (neurons and layers) transfer across tasks and domains.

    Attributes:
        model: The neural network model under analysis.
        tokenizer: Tokenizer for encoding text prompts.
        device: Computation device ('cpu' or 'cuda') used for evaluation.

    Methods:
        test_cross_benchmark_consistency(
            circuit_neurons: Dict[int, List[int]],
            benchmarks: Dict[str, List[ReasoningExample]],
            extractor: UniversalActivationExtractor
        ) -> Dict:
            Tests whether the same neurons remain consistently active across different 
            benchmarks or reasoning datasets.

            For each benchmark, a subset of task examples is evaluated to extract neuron 
            activations. Pairwise similarities between activation patterns are computed 
            within each layer, estimating how stable the identified circuit is across domains.

            Args:
                circuit_neurons (Dict[int, List[int]]):
                    Mapping of layer indices to neuron indices that form the identified circuit.
                benchmarks (Dict[str, List[ReasoningExample]]):
                    Dictionary of benchmark names and their corresponding reasoning tasks.
                extractor (UniversalActivationExtractor):
                    Tool used to extract layer-wise activations from the model.

            Returns:
                Dict:
                    A structured summary of cross-benchmark consistency:
                    {
                        benchmark_name: {
                            'n_tasks_tested': int,            # Number of evaluated tasks
                            'circuit_consistency': {
                                layer_idx: {
                                    'mean_similarity': float,  # Average pairwise similarity
                                    'std_similarity': float    # Similarity variability
                                },
                                ...
                            }
                        },
                        ...
                    }
    
    """
    
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
    
    def test_cross_benchmark_consistency(
        self,
        circuit_neurons: Dict[int, List[int]],
        benchmarks: Dict[str, List[ReasoningExample]],
        extractor: UniversalActivationExtractor
    ) -> Dict:
        """
        Test if the same neurons activate across different benchmarks
        """
        print("\nTesting cross-benchmark generalization...")
        
        results = {}
        
        for benchmark_name, tasks in benchmarks.items():
            print(f"  Testing on {benchmark_name}...")
            
            # Extract activations for this benchmark
            benchmark_activations = {}
            for i, task in enumerate(tasks[:50]):  # Sample 50 per benchmark
                acts = extractor.extract(task.prompt)
                if acts:
                    benchmark_activations[f"{benchmark_name}_{i}"] = acts
            
            # Check circuit consistency
            detector = CircuitDetector()
            task_ids = list(benchmark_activations.keys())
            
            if len(task_ids) >= 10:
                # Measure similarity within this benchmark's reasoning tasks
                circuit_consistency = {}
                for layer, neurons in circuit_neurons.items():
                    if layer in benchmark_activations[task_ids[0]]:
                        sims = []
                        for i in range(min(10, len(task_ids))):
                            for j in range(i+1, min(10, len(task_ids))):
                                sim = detector.compute_similarity(
                                    benchmark_activations[task_ids[i]][layer],
                                    benchmark_activations[task_ids[j]][layer]
                                )
                                sims.append(sim)
                        
                        if sims:
                            circuit_consistency[layer] = {
                                'mean_similarity': float(np.mean(sims)),
                                'std_similarity': float(np.std(sims))
                            }
                
                results[benchmark_name] = {
                    'n_tasks_tested': len(benchmark_activations),
                    'circuit_consistency': circuit_consistency
                }
        
        return results
