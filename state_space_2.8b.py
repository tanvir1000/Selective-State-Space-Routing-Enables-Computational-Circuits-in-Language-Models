import json
import numpy as np
from Circuit_Dectection_Framework.utils.evaluate_single_model import test_single_model

def main():
    """Run complete mechanistic analysis pipeline"""
    
    print(f"\n{'#'*80}")
    print("# COMPLETE MECHANISTIC ANALYSIS PIPELINE")
    print("# Nature Machine Intelligence Ready")
    print(f"{'#'*80}\n")
    
    # ========================================================================
    # Configure models to test
    # ========================================================================
    test_models = [
    
        ('state-spaces/mamba-2.8b-hf', 'ssm'),
       
    ]
    
    all_results = []
    
    for model_name, arch_type in test_models:
        print(f"\n{'='*80}")
        print(f"PROCESSING: {model_name}")
        print(f"{'='*80}\n")
        
        result = test_single_model(
            model_name, 
            arch_type, 
            n_samples=300
        )
        
        if result:
            all_results.append(result)
    
    # ========================================================================
    # Create comparative summary
    # ========================================================================
    if len(all_results) > 1:
        print("\n" + "="*80)
        print("COMPARATIVE ANALYSIS")
        print("="*80 + "\n")
        
        comparison = {
            'models_compared': len(all_results),
            'comparison_table': []
        }
        
        for result in all_results:
            model_summary = {
                'model': result['model'],
                'architecture': result['architecture'],
                'circuits_found': result['circuit_detection']['circuits_found'],
                'total_layers': result['circuit_detection']['total_layers_tested'],
                'avg_effect_size': float(np.mean([
                    r['cohens_d'] 
                    for r in result['circuit_detection']['detailed_results'].values()
                ])),
                'total_circuit_neurons': result['sparse_probing']['total_circuit_neurons'],
                'avg_sparsity': float(np.mean(list(
                    result['sparse_probing']['sparsity_by_layer'].values()
                ))) if result['sparse_probing']['sparsity_by_layer'] else 0
            }
            comparison['comparison_table'].append(model_summary)
        
        # Save comparison
        comparison_file = f"comparative_analysis_{__import__('datetime').datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(comparison_file, 'w') as f:
            json.dump(comparison, f, indent=2)
        
        print(f"✓ Comparative analysis saved to: {comparison_file}")
        
        # Print comparison table
        print("\n" + "="*80)
        print("MODEL COMPARISON TABLE")
        print("="*80 + "\n")
        
        print(f"{'Model':<30} | {'Arch':<12} | {'Circuits':<10} | {'Effect Size':<12} | {'Sparsity':<10}")
        print("-" * 100)
        
        for summary in comparison['comparison_table']:
            model_short = summary['model'].split('/')[-1][:28]
            circuits = f"{summary['circuits_found']}/{summary['total_layers']}"
            print(f"{model_short:<30} | {summary['architecture']:<12} | {circuits:<10} | {summary['avg_effect_size']:<12.3f} | {summary['avg_sparsity']:<10.3f}")
        
        print()
    
    # ========================================================================
    # Save master results file
    # ========================================================================
    master_results = {
        'experiment': 'Complete Mechanistic Analysis - Nature MI',
        'timestamp': __import__('datetime').datetime.now().isoformat(),
        'models_tested': len(all_results),
        'analyses_performed': [
            'Circuit Detection',
            'Sparse Probing',
            'Mechanistic Interpretation',
            'Gradient-based Tracing',
            'Cross-benchmark Generalization'
        ],
        'all_results': all_results
    }
    
    master_file = f"master_results_{__import__('datetime').datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(master_file, 'w') as f:
        json.dump(master_results, f, indent=2)
    
    print("="*80)
    print("PIPELINE COMPLETE")
    print("="*80)
    print(f"\n✓ Master results saved to: {master_file}")
    print(f"✓ Individual model results saved")
    if len(all_results) > 1:
        print(f"✓ Comparative analysis saved")
    print("="*80 + "\n")
    
    return all_results

if __name__ == '__main__':
    from huggingface_hub import login
    login(token="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")
    results = main()