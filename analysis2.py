"""
MULTI-FILE COMPLETE ANALYSIS TOOL
Load all individual model result files and perform comprehensive analysis

This tool:
1. Auto-discovers all complete_analysis_*.json files
2. Loads ALL experimental data (not just circuit detection)
3. Performs deep mechanistic analysis
4. Creates publication-ready visualizations
5. Generates comprehensive statistics

Usage:
  Place all your complete_analysis_*.json files in the same directory
  Run this script
  Get complete analysis!
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from collections import defaultdict
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")

class ComprehensiveMultiModelAnalyzer:
    """Deep analysis across all model result files"""
    
    def __init__(self, results_dir: str = "."):
        self.results_dir = Path(results_dir)
        self.models = {}
        self.architecture_groups = {
            'ssm': [],
            'transformer': [],
            'hybrid': [],
            'efficient': []
        }
        
    def discover_and_load_all_files(self):
        """Auto-discover and load all result files"""
        print("\n" + "="*80)
        print("AUTO-DISCOVERING AND LOADING RESULT FILES")
        print("="*80 + "\n")
        # Find all complete_analysis files
        pattern = "complete_analysis_*.json"
        files = list(self.results_dir.glob(pattern))
        
        if not files:
            print(f"⚠️ No files found matching pattern: {pattern}")
            print(f"   Looking in: {self.results_dir.absolute()}")
            print("\nPlease ensure your files are named like:")
            print("  complete_analysis_ssm_mamba-2.8b-hf_20251019_153351.json")
            return False
        
        print(f"Found {len(files)} result files:\n")
        
        # Load each file
        for file_path in sorted(files):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                model_name = data['model']
                self.models[model_name] = data
                
                # Categorize by architecture
                arch = data['architecture'].lower()
                if 'ssm' in arch:
                    self.architecture_groups['ssm'].append(model_name)
                elif 'transformer' in arch or 'dense' in arch:
                    self.architecture_groups['transformer'].append(model_name)
                elif 'hybrid' in arch:
                    self.architecture_groups['hybrid'].append(model_name)
                elif 'efficient' in arch:
                    self.architecture_groups['efficient'].append(model_name)
                
                # Print summary
                print(f"✓ {file_path.name}")
                print(f"  Model: {model_name}")
                print(f"  Architecture: {data['architecture']}")
                print(f"  Parameters: {data['parameters']:,}")
                print(f"  Circuits: {data['circuit_detection']['circuits_found']}/{data['circuit_detection']['total_layers_tested']}")
                
                # Check what data is available
                data_available = []
                if 'circuit_detection' in data:
                    data_available.append("Circuit Detection")
                if 'sparse_probing' in data:
                    data_available.append("Sparse Probing")
                if 'mechanistic_interpretation' in data:
                    data_available.append("Mechanistic Interp")
                if 'gradient_tracing' in data:
                    data_available.append("Gradient Tracing")
                if 'generalization' in data:
                    data_available.append("Generalization")
                
                print(f"  Data: {', '.join(data_available)}")
                print()
                
            except Exception as e:
                print(f"❌ Failed to load {file_path.name}: {e}\n")
        
        print(f"\n{'='*80}")
        print(f"SUMMARY: Loaded {len(self.models)} models")
        print(f"  SSM: {len(self.architecture_groups['ssm'])}")
        print(f"  Transformer: {len(self.architecture_groups['transformer'])}")
        print(f"  Hybrid: {len(self.architecture_groups['hybrid'])}")
        print(f"  Efficient: {len(self.architecture_groups['efficient'])}")
        print(f"{'='*80}\n")
        
        return True
    
    def analyze_circuit_neurons(self):
        """Deep analysis of which neurons form circuits"""
        print("="*80)
        print("CIRCUIT NEURON ANALYSIS")
        print("="*80 + "\n")
        
        neuron_data = []
        
        for model_name, data in self.models.items():
            if 'sparse_probing' not in data:
                continue
            
            sp = data['sparse_probing']
            arch = data['architecture']
            params = data['parameters']
            
            # Get circuit neurons
            if 'circuit_neurons' in sp:
                total_circuit_neurons = sp.get('total_circuit_neurons', 0)
                
                print(f"📊 {model_name.split('/')[-1]}")
                print(f"  Total circuit neurons: {total_circuit_neurons}")
                
                # Per-layer breakdown
                for layer, neurons in sp['circuit_neurons'].items():
                    if layer in sp['probe_results']:
                        probe = sp['probe_results'][layer]
                        
                        print(f"  Layer {layer}:")
                        print(f"    Circuit neurons: {len(neurons)}")
                        print(f"    Total neurons: {probe['total_neurons']}")
                        print(f"    Percentage: {len(neurons)/probe['total_neurons']*100:.1f}%")
                        print(f"    Probe accuracy: {probe['accuracy']:.3f}")
                        print(f"    Sparsity: {probe['sparsity']:.3f}")
                        
                        neuron_data.append({
                            'Model': model_name.split('/')[-1],
                            'Architecture': arch,
                            'Parameters': params,
                            'Layer': int(layer),
                            'Circuit_Neurons': len(neurons),
                            'Total_Neurons': probe['total_neurons'],
                            'Percentage': len(neurons)/probe['total_neurons']*100,
                            'Probe_Accuracy': probe['accuracy'],
                            'Sparsity': probe['sparsity']
                        })
                
                print()
        
        if neuron_data:
            df = pd.DataFrame(neuron_data)
            
            print("\n📊 CIRCUIT NEURON STATISTICS\n")
            
            # By architecture
            for arch in df['Architecture'].unique():
                arch_df = df[df['Architecture'] == arch]
                print(f"{arch}:")
                print(f"  Avg circuit neurons per layer: {arch_df['Circuit_Neurons'].mean():.1f} ± {arch_df['Circuit_Neurons'].std():.1f}")
                print(f"  Avg percentage: {arch_df['Percentage'].mean():.1f}% ± {arch_df['Percentage'].std():.1f}%")
                print(f"  Avg sparsity: {arch_df['Sparsity'].mean():.3f} ± {arch_df['Sparsity'].std():.3f}")
                print()
            
            return df
        
        print("⚠️ No sparse probing data found\n")
        return None
    
    def analyze_neuron_selectivity(self):
        """Analyze what neurons are selective for"""
        print("="*80)
        print("NEURON SELECTIVITY ANALYSIS")
        print("="*80 + "\n")
        
        selectivity_summary = []
        
        for model_name, data in self.models.items():
            if 'mechanistic_interpretation' not in data:
                continue
            
            mi = data['mechanistic_interpretation']
            
            if 'results_by_layer' not in mi:
                continue
            
            print(f"📊 {model_name.split('/')[-1]}\n")
            
            for layer, layer_data in mi['results_by_layer'].items():
                if 'neuron_selectivity' not in layer_data:
                    continue
                
                selectivity = layer_data['neuron_selectivity']
                
                # Aggregate selectivity by task type
                task_type_counts = defaultdict(int)
                task_type_strengths = defaultdict(list)
                
                for neuron_id, sel_data in selectivity.items():
                    if 'most_selective_for' in sel_data:
                        task_type = sel_data['most_selective_for']
                        task_type_counts[task_type] += 1
                        task_type_strengths[task_type].append(sel_data.get('selectivity_strength', 0))
                
                print(f"  Layer {layer} ({len(selectivity)} neurons):")
                
                for task_type in sorted(task_type_counts.keys(), key=lambda x: task_type_counts[x], reverse=True):
                    count = task_type_counts[task_type]
                    avg_strength = np.mean(task_type_strengths[task_type])
                    pct = count / len(selectivity) * 100
                    
                    print(f"    {task_type}: {count} neurons ({pct:.1f}%), strength={avg_strength:.3f}")
                    
                    selectivity_summary.append({
                        'Model': model_name.split('/')[-1],
                        'Architecture': data['architecture'],
                        'Layer': int(layer),
                        'Task_Type': task_type,
                        'N_Neurons': count,
                        'Percentage': pct,
                        'Avg_Strength': avg_strength
                    })
                
                print()
        
        if selectivity_summary:
            df = pd.DataFrame(selectivity_summary)
            
            print("\n📊 SELECTIVITY SUMMARY\n")
            
            # Overall task type distribution
            task_totals = df.groupby('Task_Type')['N_Neurons'].sum().sort_values(ascending=False)
            print("Overall task type preferences:")
            for task_type, count in task_totals.items():
                print(f"  {task_type}: {count} neurons")
            
            print()
            
            return df
        
        print("⚠️ No mechanistic interpretation data found\n")
        return None
    
    def analyze_information_flow(self):
        """Analyze gradient-based information flow"""
        print("="*80)
        print("INFORMATION FLOW ANALYSIS")
        print("="*80 + "\n")
        
        flow_summary = []
        
        for model_name, data in self.models.items():
            if 'gradient_tracing' not in data:
                continue
            
            gt = data['gradient_tracing']
            
            print(f"📊 {model_name.split('/')[-1]}")
            print(f"  Tasks traced: {gt.get('n_tasks_traced', 0)}")
            
            if 'flow_patterns' in gt and len(gt['flow_patterns']) > 0:
                # Aggregate flow information
                layer_activity = defaultdict(list)
                
                for pattern in gt['flow_patterns']:
                    if 'layer_flows' in pattern:
                        for layer, flow_data in pattern['layer_flows'].items():
                            n_active = len(flow_data.get('active_neurons', []))
                            layer_activity[layer].append(n_active)
                
                print(f"  Flow patterns analyzed: {len(gt['flow_patterns'])}")
                print(f"  Average active neurons per layer:")
                
                for layer in sorted(layer_activity.keys(), key=lambda x: int(x)):
                    avg_active = np.mean(layer_activity[layer])
                    print(f"    Layer {layer}: {avg_active:.1f} neurons")
                    
                    flow_summary.append({
                        'Model': model_name.split('/')[-1],
                        'Architecture': data['architecture'],
                        'Layer': int(layer),
                        'Avg_Active_Neurons': avg_active,
                        'N_Patterns': len(layer_activity[layer])
                    })
            
            print()
        
        if flow_summary:
            df = pd.DataFrame(flow_summary)
            return df
        
        print("⚠️ No gradient tracing data found\n")
        return None
    
    def analyze_cross_benchmark_generalization(self):
        """Analyze how circuits generalize across benchmarks"""
        print("="*80)
        print("CROSS-BENCHMARK GENERALIZATION ANALYSIS")
        print("="*80 + "\n")
        
        gen_data = []
        
        for model_name, data in self.models.items():
            if 'generalization' not in data:
                continue
            
            gen = data['generalization']
            
            print(f"📊 {model_name.split('/')[-1]}")
            
            if 'cross_benchmark_results' in gen:
                for benchmark, results in gen['cross_benchmark_results'].items():
                    print(f"\n  {benchmark}:")
                    print(f"    Tasks tested: {results.get('n_tasks_tested', 0)}")
                    
                    if 'circuit_consistency' in results:
                        for layer, consistency in results['circuit_consistency'].items():
                            mean_sim = consistency.get('mean_similarity', 0)
                            std_sim = consistency.get('std_similarity', 0)
                            
                            print(f"    Layer {layer}: {mean_sim:.3f} ± {std_sim:.3f}")
                            
                            gen_data.append({
                                'Model': model_name.split('/')[-1],
                                'Architecture': data['architecture'],
                                'Benchmark': benchmark,
                                'Layer': int(layer),
                                'Mean_Similarity': mean_sim,
                                'Std_Similarity': std_sim
                            })
            
            print()
        
        if gen_data:
            df = pd.DataFrame(gen_data)
            
            print("\n📊 GENERALIZATION SUMMARY\n")
            
            # By architecture
            for arch in df['Architecture'].unique():
                arch_df = df[df['Architecture'] == arch]
                print(f"{arch}:")
                print(f"  Mean circuit consistency: {arch_df['Mean_Similarity'].mean():.3f} ± {arch_df['Mean_Similarity'].std():.3f}")
                print(f"  Benchmarks tested: {arch_df['Benchmark'].nunique()}")
                print()
            
            return df
        
        print("⚠️ No generalization data found\n")
        return None
    
    def create_comprehensive_visualizations(self, save_dir: str = "."):
        """Create all visualizations"""
        print("="*80)
        print("CREATING COMPREHENSIVE VISUALIZATIONS")
        print("="*80 + "\n")
        
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        # Figure 1: Neuron-level analysis
        self._create_neuron_analysis_figure(save_path)
        
        # Figure 2: Selectivity patterns
        self._create_selectivity_figure(save_path)
        
        # Figure 3: Information flow
        self._create_flow_figure(save_path)
        
        print()
    
    def _create_neuron_analysis_figure(self, save_path: Path):
        """Create neuron-level analysis visualizations"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Circuit Neuron Analysis', fontsize=16, fontweight='bold')
        
        # Collect neuron data
        neuron_data = []
        for model_name, data in self.models.items():
            if 'sparse_probing' not in data:
                continue
            
            sp = data['sparse_probing']
            is_ssm = 'ssm' in data['architecture'].lower()
            
            if 'circuit_neurons' in sp:
                for layer, neurons in sp['circuit_neurons'].items():
                    if layer in sp['probe_results']:
                        probe = sp['probe_results'][layer]
                        neuron_data.append({
                            'Architecture': 'SSM' if is_ssm else 'Transformer',
                            'Model': model_name.split('/')[-1][:15],
                            'Circuit_Neurons': len(neurons),
                            'Total_Neurons': probe['total_neurons'],
                            'Percentage': len(neurons)/probe['total_neurons']*100,
                            'Sparsity': probe['sparsity'],
                            'Accuracy': probe['accuracy']
                        })
        
        if not neuron_data:
            plt.close()
            return
        
        df = pd.DataFrame(neuron_data)
        
        # 1a: Circuit neuron percentage
        ax = axes[0, 0]
        df.boxplot(column='Percentage', by='Architecture', ax=ax, patch_artist=True)
        ax.set_title('(A) Circuit Neurons Percentage', fontweight='bold')
        ax.set_xlabel('Architecture', fontweight='bold')
        ax.set_ylabel('% of Total Neurons', fontweight='bold')
        plt.sca(ax)
        plt.xticks(rotation=0)
        
        # 1b: Sparsity distribution
        ax = axes[0, 1]
        for arch, color in [('SSM', '#2ecc71'), ('Transformer', '#e74c3c')]:
            arch_df = df[df['Architecture'] == arch]
            if len(arch_df) > 0:
                ax.hist(arch_df['Sparsity'], alpha=0.6, label=arch, color=color, bins=15)
        ax.set_xlabel('Sparsity', fontweight='bold')
        ax.set_ylabel('Frequency', fontweight='bold')
        ax.set_title('(B) Sparsity Distribution', fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # 1c: Sparsity vs Accuracy
        ax = axes[1, 0]
        for arch, color in [('SSM', '#2ecc71'), ('Transformer', '#e74c3c')]:
            arch_df = df[df['Architecture'] == arch]
            if len(arch_df) > 0:
                ax.scatter(arch_df['Sparsity'], arch_df['Accuracy'], 
                          alpha=0.6, s=50, label=arch, color=color)
        ax.set_xlabel('Sparsity', fontweight='bold')
        ax.set_ylabel('Probe Accuracy', fontweight='bold')
        ax.set_title('(C) Sparsity vs Accuracy', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # 1d: Circuit neurons by model
        ax = axes[1, 1]
        models = df['Model'].unique()[:10]  # Top 10 models
        circuit_counts = [df[df['Model'] == m]['Circuit_Neurons'].mean() for m in models]
        colors = ['#2ecc71' if 'mamba' in m.lower() else '#e74c3c' for m in models]
        ax.barh(range(len(models)), circuit_counts, color=colors, alpha=0.7)
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(models, fontsize=8)
        ax.set_xlabel('Avg Circuit Neurons', fontweight='bold')
        ax.set_title('(D) Circuit Neurons by Model', fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        fig_path = save_path / 'neuron_analysis.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {fig_path}")
        plt.close()
    
    def _create_selectivity_figure(self, save_path: Path):
        """Create selectivity analysis figure"""
        # Collect selectivity data
        sel_data = []
        for model_name, data in self.models.items():
            if 'mechanistic_interpretation' not in data:
                continue
            
            mi = data['mechanistic_interpretation']
            if 'results_by_layer' not in mi:
                continue
            
            for layer, layer_data in mi['results_by_layer'].items():
                if 'neuron_selectivity' not in layer_data:
                    continue
                
                selectivity = layer_data['neuron_selectivity']
                
                for neuron_id, sel in selectivity.items():
                    if 'most_selective_for' in sel:
                        sel_data.append({
                            'Architecture': data['architecture'],
                            'Task_Type': sel['most_selective_for'],
                            'Strength': sel.get('selectivity_strength', 0)
                        })
        
        if not sel_data:
            return
        
        df = pd.DataFrame(sel_data)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create grouped bar chart
        task_types = df['Task_Type'].unique()
        x = np.arange(len(task_types))
        width = 0.35
        
        ssm_df = df[df['Architecture'].str.contains('ssm', case=False)]
        trans_df = df[df['Architecture'].str.contains('transformer|dense', case=False, regex=True)]
        
        ssm_counts = [len(ssm_df[ssm_df['Task_Type'] == tt]) for tt in task_types]
        trans_counts = [len(trans_df[trans_df['Task_Type'] == tt]) for tt in task_types]
        
        ax.bar(x - width/2, ssm_counts, width, label='SSM', color='#2ecc71', alpha=0.8)
        ax.bar(x + width/2, trans_counts, width, label='Transformer', color='#e74c3c', alpha=0.8)
        
        ax.set_xlabel('Task Type', fontweight='bold')
        ax.set_ylabel('Number of Selective Neurons', fontweight='bold')
        ax.set_title('Neuron Selectivity by Task Type', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(task_types, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        fig_path = save_path / 'selectivity_analysis.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {fig_path}")
        plt.close()
    
    def _create_flow_figure(self, save_path: Path):
        """Create information flow figure"""
        # This would create visualizations of gradient flow patterns
        # Placeholder for now
        pass
    
    def generate_publication_summary(self):
        """Generate complete publication-ready summary"""
        print("="*80)
        print("PUBLICATION-READY SUMMARY")
        print("="*80 + "\n")
        
        summary = {
            'models': {},
            'totals': {
                'models_analyzed': len(self.models),
                'circuits_detected': 0,
                'layers_tested': 0,
                'neurons_analyzed': 0,
                'probes_trained': 0,
                'tasks_traced': 0
            }
        }
        
        for model_name, data in self.models.items():
            model_summary = {
                'architecture': data['architecture'],
                'parameters': data['parameters'],
                'circuits': data['circuit_detection']['circuits_found'],
                'layers': data['circuit_detection']['total_layers_tested']
            }
            
            # Add sparse probing info
            if 'sparse_probing' in data:
                model_summary['circuit_neurons'] = data['sparse_probing'].get('total_circuit_neurons', 0)
                model_summary['sparsity'] = np.mean(list(data['sparse_probing'].get('sparsity_by_layer', {}).values())) if data['sparse_probing'].get('sparsity_by_layer') else 0
            
            # Add mechanistic interpretation info
            if 'mechanistic_interpretation' in data:
                mi = data['mechanistic_interpretation']
                if 'results_by_layer' in mi:
                    model_summary['neurons_interpreted'] = sum(
                        ld.get('n_neurons_analyzed', 0) 
                        for ld in mi['results_by_layer'].values()
                    )
            
            summary['models'][model_name] = model_summary
            
            # Update totals
            summary['totals']['circuits_detected'] += model_summary['circuits']
            summary['totals']['layers_tested'] += model_summary['layers']
            if 'circuit_neurons' in model_summary:
                summary['totals']['neurons_analyzed'] += model_summary.get('circuit_neurons', 0)
        
        # Print summary
        print("📊 EXPERIMENTAL SCOPE\n")
        print(f"Total models: {summary['totals']['models_analyzed']}")
        print(f"Total layers tested: {summary['totals']['layers_tested']}")
        print(f"Total circuits detected: {summary['totals']['circuits_detected']}")
        print(f"Total circuit neurons: {summary['totals']['neurons_analyzed']}\n")
        
        print("📊 PER-MODEL DETAILS\n")
        for model_name, ms in summary['models'].items():
            print(f"{model_name.split('/')[-1]}:")
            print(f"  Architecture: {ms['architecture']}")
            print(f"  Parameters: {ms['parameters']:,}")
            print(f"  Circuits: {ms['circuits']}/{ms['layers']}")
            if 'circuit_neurons' in ms:
                print(f"  Circuit neurons: {ms['circuit_neurons']}")
                print(f"  Sparsity: {ms.get('sparsity', 0):.3f}")
            if 'neurons_interpreted' in ms:
                print(f"  Neurons interpreted: {ms['neurons_interpreted']}")
            print()
        
        return summary

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run complete multi-file analysis"""
    print("\n" + "#"*80)
    print("# MULTI-FILE COMPREHENSIVE ANALYSIS")
    print("# Complete Mechanistic Analysis - Nature MI Ready")
    print("#"*80 + "\n")
    
    # Initialize analyzer
    analyzer = ComprehensiveMultiModelAnalyzer()
    
    # Auto-discover and load all files
    if not analyzer.discover_and_load_all_files():
        print("❌ No result files found. Exiting.")
        return
    
    # Run all analyses
    print("\n" + "="*80)
    print("RUNNING ALL ANALYSES")
    print("="*80 + "\n")
    
    analyzer.generate_publication_summary()
    analyzer.analyze_circuit_neurons()
    analyzer.analyze_neuron_selectivity()
    analyzer.analyze_information_flow()
    analyzer.analyze_cross_benchmark_generalization()
    
    # Create visualizations
    analyzer.create_comprehensive_visualizations()
    
    print("="*80)
    print("COMPLETE ANALYSIS FINISHED")
    print("="*80)
    print("\nGenerated files:")
    print("  • neuron_analysis.png - Circuit neuron patterns")
    print("  • selectivity_analysis.png - Task selectivity")
    print("\nAll mechanistic analyses complete!")
    print("="*80 + "\n")

if __name__ == '__main__':
    main()