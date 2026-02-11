"""
COMPREHENSIVE ANALYSIS & VISUALIZATION SUITE
For Nature Machine Intelligence Publication

Reads all master_results_*.json files and generates:
1. Publication-quality figures
2. Statistical comparisons
3. Detailed tables
4. LaTeX-ready outputs
5. Interactive visualizations
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from scipy import stats
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality plotting defaults
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
sns.set_palette("colorblind")

class PublicationAnalyzer:
    """Generate publication-ready analysis from multiple master results"""
    
    def __init__(self, results_dir: str = "."):
        self.results_dir = Path(results_dir)
        self.all_results = []
        self.summary_df = None
        
    def load_all_results(self):
        """Load all master_results_*.json files"""
        print("="*80)
        print("LOADING RESULTS FILES")
        print("="*80 + "\n")
        
        json_files = list(self.results_dir.glob("master_results_*.json"))
        
        # Also try complete_analysis_*.json files
        if not json_files:
            json_files = list(self.results_dir.glob("complete_analysis_*.json"))
        
        print(f"Found {len(json_files)} results files:")
        for f in json_files:
            print(f"  • {f.name}")
        
        for filepath in json_files:
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    
                    # Handle both master_results and complete_analysis formats
                    if 'all_results' in data:
                        # Master results format
                        for result in data['all_results']:
                            self.all_results.append(result)
                    elif 'model' in data:
                        # Single complete_analysis format
                        self.all_results.append(data)
                    
                print(f"✓ Loaded: {filepath.name}")
            except Exception as e:
                print(f"✗ Failed to load {filepath.name}: {e}")
        
        print(f"\n✓ Total models loaded: {len(self.all_results)}\n")
        
        if not self.all_results:
            raise ValueError("No results found! Make sure you have master_results_*.json or complete_analysis_*.json files")
        
        return self.all_results
    
    def create_summary_dataframe(self):
        """Create comprehensive summary DataFrame"""
        print("="*80)
        print("CREATING SUMMARY DATAFRAME")
        print("="*80 + "\n")
        
        rows = []
        
        for result in self.all_results:
            model_name = result['model']
            arch = result.get('architecture', 'unknown')
            params = result.get('parameters', 0)
            
            # Circuit detection metrics
            cd = result['circuit_detection']
            circuits_found = cd['circuits_found']
            total_layers = cd['total_layers_tested']
            circuit_rate = circuits_found / total_layers if total_layers > 0 else 0
            
            # Calculate average metrics from detailed results
            detailed = cd['detailed_results']
            valid_layers = [v for v in detailed.values() if 'cohens_d' in v]
            
            if valid_layers:
                avg_reasoning_sim = np.mean([v['reasoning_similarity_mean'] for v in valid_layers])
                avg_control_sim = np.mean([v['control_similarity_mean'] for v in valid_layers])
                avg_cohens_d = np.mean([v['cohens_d'] for v in valid_layers])
                avg_p_value = np.mean([v['p_value'] for v in valid_layers])
                
                # Circuit-specific metrics
                circuit_layers = [v for v in valid_layers if v.get('is_circuit', False)]
                if circuit_layers:
                    circuit_reasoning_sim = np.mean([v['reasoning_similarity_mean'] for v in circuit_layers])
                    circuit_control_sim = np.mean([v['control_similarity_mean'] for v in circuit_layers])
                    circuit_cohens_d = np.mean([v['cohens_d'] for v in circuit_layers])
                    circuit_gap = np.mean([v.get('gap', v['reasoning_similarity_mean'] - v['control_similarity_mean']) 
                       for v in circuit_layers])
                else:
                    circuit_reasoning_sim = 0
                    circuit_control_sim = 0
                    circuit_cohens_d = 0
                    circuit_gap = 0
            else:
                avg_reasoning_sim = avg_control_sim = avg_cohens_d = avg_p_value = 0
                circuit_reasoning_sim = circuit_control_sim = circuit_cohens_d = circuit_gap = 0
            
            # Sparse probing metrics
            sp = result.get('sparse_probing', {})
            total_circuit_neurons = sp.get('total_circuit_neurons', 0)
            
            probe_results = sp.get('probe_results', {})
            if probe_results:
                avg_probe_acc = np.mean([v['accuracy'] for v in probe_results.values()])
                avg_sparsity = np.mean([v['sparsity'] for v in probe_results.values()])
                total_neurons = np.mean([v['total_neurons'] for v in probe_results.values()])
            else:
                avg_probe_acc = avg_sparsity = total_neurons = 0
            
            # Mechanistic interpretation
            mi = result.get('mechanistic_interpretation', {})
            n_layers_interpreted = mi.get('n_layers_interpreted', 0)
            
            # Task information
            tasks = result.get('tasks', {})
            n_reasoning = tasks.get('n_reasoning_total', 0)
            n_control = tasks.get('n_control_total', 0)
            n_gsm8k = tasks.get('n_gsm8k', 0)
            
            row = {
                'model': model_name.split('/')[-1],
                'full_model_name': model_name,
                'architecture': arch,
                'parameters_millions': params / 1_000_000 if params > 0 else 0,
                
                # Circuit detection
                'circuits_found': circuits_found,
                'total_layers': total_layers,
                'circuit_rate': circuit_rate,
                'avg_reasoning_similarity': avg_reasoning_sim,
                'avg_control_similarity': avg_control_sim,
                'avg_cohens_d': avg_cohens_d,
                'avg_p_value': avg_p_value,
                
                # Circuit-specific metrics
                'circuit_reasoning_sim': circuit_reasoning_sim,
                'circuit_control_sim': circuit_control_sim,
                'circuit_cohens_d': circuit_cohens_d,
                'circuit_gap': circuit_gap,
                
                # Sparse probing
                'total_circuit_neurons': total_circuit_neurons,
                'avg_probe_accuracy': avg_probe_acc,
                'avg_sparsity': avg_sparsity,
                'avg_neurons_per_layer': total_neurons,
                'circuit_neuron_density': total_circuit_neurons / (total_neurons * total_layers) if total_neurons > 0 else 0,
                
                # Mechanistic interpretation
                'layers_interpreted': n_layers_interpreted,
                
                # Task info
                'n_reasoning_tasks': n_reasoning,
                'n_control_tasks': n_control,
                'n_gsm8k_tasks': n_gsm8k,
            }
            
            rows.append(row)
        
        self.summary_df = pd.DataFrame(rows)
        
        print(f"✓ Created summary with {len(self.summary_df)} models")
        print(f"✓ Metrics computed: {len(self.summary_df.columns)}")
        print(f"\nModels included:")
        for model in self.summary_df['model'].values:
            print(f"  • {model}")
        print()
        
        return self.summary_df
    
    def generate_all_visualizations(self, output_dir: str = "publication_figures"):
        """Generate all publication figures"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print("="*80)
        print("GENERATING PUBLICATION FIGURES")
        print("="*80 + "\n")
        
        figures = []
        
        # Figure 1: Circuit Detection Overview
        print("Creating Figure 1: Circuit Detection Overview...")
        fig1 = self.plot_circuit_detection_overview()
        fig1.savefig(output_path / "figure1_circuit_detection.png", bbox_inches='tight')
        figures.append("figure1_circuit_detection.png")
        plt.close()
        
        # Figure 2: Architecture Comparison
        print("Creating Figure 2: Architecture Comparison...")
        fig2 = self.plot_architecture_comparison()
        fig2.savefig(output_path / "figure2_architecture_comparison.png", bbox_inches='tight')
        figures.append("figure2_architecture_comparison.png")
        plt.close()
        
        # Figure 3: Sparsity Analysis
        print("Creating Figure 3: Sparsity Analysis...")
        fig3 = self.plot_sparsity_analysis()
        fig3.savefig(output_path / "figure3_sparsity_analysis.png", bbox_inches='tight')
        figures.append("figure3_sparsity_analysis.png")
        plt.close()
        
        # Figure 4: Effect Size Distribution
        print("Creating Figure 4: Effect Size Distribution...")
        fig4 = self.plot_effect_size_distribution()
        fig4.savefig(output_path / "figure4_effect_sizes.png", bbox_inches='tight')
        figures.append("figure4_effect_sizes.png")
        plt.close()
        
        # Figure 5: Layer-wise Circuit Analysis
        print("Creating Figure 5: Layer-wise Analysis...")
        fig5 = self.plot_layerwise_analysis()
        fig5.savefig(output_path / "figure5_layerwise_analysis.png", bbox_inches='tight')
        figures.append("figure5_layerwise_analysis.png")
        plt.close()
        
        # Figure 6: Similarity Heatmap
        print("Creating Figure 6: Similarity Heatmap...")
        fig6 = self.plot_similarity_heatmap()
        fig6.savefig(output_path / "figure6_similarity_heatmap.png", bbox_inches='tight')
        figures.append("figure6_similarity_heatmap.png")
        plt.close()
        
        # Figure 7: Statistical Significance
        print("Creating Figure 7: Statistical Significance...")
        fig7 = self.plot_statistical_significance()
        fig7.savefig(output_path / "figure7_statistics.png", bbox_inches='tight')
        figures.append("figure7_statistics.png")
        plt.close()
        
        # Figure 8: Circuit Neuron Analysis
        print("Creating Figure 8: Circuit Neuron Analysis...")
        fig8 = self.plot_circuit_neuron_analysis()
        fig8.savefig(output_path / "figure8_neuron_analysis.png", bbox_inches='tight')
        figures.append("figure8_neuron_analysis.png")
        plt.close()
        
        print(f"\n✓ All figures saved to: {output_path}/")
        print(f"✓ Total figures: {len(figures)}\n")
        
        return figures
    
    def plot_circuit_detection_overview(self):
        """Figure 1: Main circuit detection results"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        df = self.summary_df
        
        # Panel A: Circuit rate by model
        ax = axes[0, 0]
        colors = ['#2E86AB' if 'mamba' in m.lower() or 'ssm' in a.lower() 
                  else '#A23B72' for m, a in zip(df['model'], df['architecture'])]
        
        bars = ax.bar(range(len(df)), df['circuit_rate'], color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(df['model'], rotation=45, ha='right')
        ax.set_ylabel('Circuit Detection Rate')
        ax.set_title('A. Circuit Detection Rate by Model', fontweight='bold')
        ax.set_ylim(0, 1.0)
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='50% threshold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Panel B: Circuits found vs total layers
        ax = axes[0, 1]
        x = np.arange(len(df))
        width = 0.35
        
        ax.bar(x - width/2, df['circuits_found'], width, label='Circuits Found', 
               color='#06A77D', alpha=0.8, edgecolor='black')
        ax.bar(x + width/2, df['total_layers'], width, label='Total Layers',
               color='#D5C67A', alpha=0.8, edgecolor='black')
        
        ax.set_xlabel('Model')
        ax.set_ylabel('Number of Layers')
        ax.set_title('B. Circuits Found vs Total Layers', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(df['model'], rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Panel C: Effect sizes
        ax = axes[1, 0]
        ax.scatter(df['avg_control_similarity'], df['avg_reasoning_similarity'], 
                   s=200, c=colors, alpha=0.7, edgecolor='black', linewidth=2)
        
        for i, model in enumerate(df['model']):
            ax.annotate(model, (df['avg_control_similarity'].iloc[i], 
                               df['avg_reasoning_similarity'].iloc[i]),
                       xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # Diagonal line
        lim = [0, 1]
        ax.plot(lim, lim, 'k--', alpha=0.3, label='Equal similarity')
        
        ax.set_xlabel('Control Similarity (avg)')
        ax.set_ylabel('Reasoning Similarity (avg)')
        ax.set_title('C. Reasoning vs Control Similarity', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # Panel D: Cohen's d distribution
        ax = axes[1, 1]
        positions = range(len(df))
        parts = ax.violinplot([self._get_cohens_d_distribution(model) 
                               for model in df['full_model_name']], 
                              positions=positions, widths=0.7, showmeans=True)
        
        for pc, color in zip(parts['bodies'], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
        
        ax.set_xticks(positions)
        ax.set_xticklabels(df['model'], rotation=45, ha='right')
        ax.set_ylabel("Cohen's d")
        ax.set_title("D. Effect Size Distribution", fontweight='bold')
        ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='Large effect (d=0.8)')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_architecture_comparison(self):
        """Figure 2: SSM vs Transformer comparison"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        df = self.summary_df
        
        # Group by architecture
        arch_groups = df.groupby('architecture')
        
        # Panel A: Circuit rate by architecture
        ax = axes[0, 0]
        arch_stats = arch_groups.agg({
            'circuit_rate': ['mean', 'std'],
            'model': 'count'
        })
        
        architectures = arch_stats.index.tolist()
        means = arch_stats['circuit_rate']['mean'].values
        stds = arch_stats['circuit_rate']['std'].values
        counts = arch_stats['model']['count'].values
        
        colors_arch = ['#2E86AB' if 'ssm' in a.lower() else '#A23B72' for a in architectures]
        
        bars = ax.bar(architectures, means, yerr=stds, capsize=5, 
                     color=colors_arch, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'n={count}', ha='center', va='bottom', fontsize=9)
        
        ax.set_ylabel('Circuit Detection Rate')
        ax.set_title('A. Circuit Rate by Architecture', fontweight='bold')
        ax.set_ylim(0, max(means) * 1.2)
        ax.grid(axis='y', alpha=0.3)
        
        # Panel B: Effect size comparison
        ax = axes[0, 1]
        arch_effect = arch_groups['avg_cohens_d'].agg(['mean', 'std'])
        
        bars = ax.bar(architectures, arch_effect['mean'], yerr=arch_effect['std'],
                     capsize=5, color=colors_arch, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        ax.set_ylabel("Average Cohen's d")
        ax.set_title('B. Effect Size by Architecture', fontweight='bold')
        ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='Large effect')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Panel C: Sparsity comparison
        ax = axes[1, 0]
        arch_sparsity = arch_groups['avg_sparsity'].agg(['mean', 'std'])
        
        bars = ax.bar(architectures, arch_sparsity['mean'], yerr=arch_sparsity['std'],
                     capsize=5, color=colors_arch, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        ax.set_ylabel('Average Sparsity')
        ax.set_title('C. Circuit Sparsity by Architecture', fontweight='bold')
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3)
        
        # Panel D: Statistical test
        ax = axes[1, 1]
        
        # Prepare data for comparison
        comparison_metrics = ['circuit_rate', 'avg_cohens_d', 'avg_sparsity', 'circuit_gap']
        
        if len(architectures) >= 2:
            arch1, arch2 = architectures[0], architectures[1]
            group1 = df[df['architecture'] == arch1]
            group2 = df[df['architecture'] == arch2]
            
            p_values = []
            for metric in comparison_metrics:
                if len(group1) > 1 and len(group2) > 1:
                    stat, p = stats.mannwhitneyu(group1[metric].dropna(), 
                                                 group2[metric].dropna(), 
                                                 alternative='two-sided')
                    p_values.append(p)
                else:
                    p_values.append(1.0)
            
            colors_sig = ['green' if p < 0.05 else 'red' for p in p_values]
            
            bars = ax.bar(range(len(comparison_metrics)), 
                         [-np.log10(p) for p in p_values],
                         color=colors_sig, alpha=0.7, edgecolor='black')
            
            ax.set_xticks(range(len(comparison_metrics)))
            ax.set_xticklabels([m.replace('_', ' ').title() for m in comparison_metrics], 
                              rotation=45, ha='right')
            ax.set_ylabel('-log10(p-value)')
            ax.set_title(f'D. Statistical Significance\n({arch1} vs {arch2})', fontweight='bold')
            ax.axhline(y=-np.log10(0.05), color='red', linestyle='--', 
                      alpha=0.5, label='p=0.05')
            ax.axhline(y=-np.log10(0.01), color='orange', linestyle='--',
                      alpha=0.5, label='p=0.01')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Need at least 2 architectures\nfor comparison',
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
        
        plt.tight_layout()
        return fig
    
    def plot_sparsity_analysis(self):
        """Figure 3: Circuit sparsity analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        df = self.summary_df
        
        # Panel A: Sparsity vs circuit rate
        ax = axes[0, 0]
        colors = ['#2E86AB' if 'mamba' in m.lower() or 'ssm' in a.lower() 
                  else '#A23B72' for m, a in zip(df['model'], df['architecture'])]
        
        scatter = ax.scatter(df['avg_sparsity'], df['circuit_rate'],
                           s=df['parameters_millions']*2, c=colors, 
                           alpha=0.7, edgecolor='black', linewidth=1.5)
        
        for i, model in enumerate(df['model']):
            ax.annotate(model, (df['avg_sparsity'].iloc[i], df['circuit_rate'].iloc[i]),
                       xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax.set_xlabel('Average Sparsity')
        ax.set_ylabel('Circuit Detection Rate')
        ax.set_title('A. Sparsity vs Circuit Detection', fontweight='bold')
        ax.grid(alpha=0.3)
        
        # Add size legend
        sizes = [100, 500, 1000]
        labels = ['100M', '500M', '1B+']
        legend_elements = [plt.scatter([], [], s=s*2, c='gray', alpha=0.7, edgecolor='black')
                          for s in sizes]
        ax.legend(legend_elements, labels, title='Parameters', loc='best')
        
        # Panel B: Circuit neurons per model
        ax = axes[0, 1]
        bars = ax.bar(range(len(df)), df['total_circuit_neurons'], 
                     color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(df['model'], rotation=45, ha='right')
        ax.set_ylabel('Total Circuit Neurons')
        ax.set_title('B. Circuit Neurons by Model', fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Panel C: Neuron density
        ax = axes[1, 0]
        bars = ax.bar(range(len(df)), df['circuit_neuron_density'] * 100,
                     color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(df['model'], rotation=45, ha='right')
        ax.set_ylabel('Circuit Neuron Density (%)')
        ax.set_title('C. Circuit Neuron Density', fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Panel D: Probe accuracy vs sparsity
        ax = axes[1, 1]
        scatter = ax.scatter(df['avg_sparsity'], df['avg_probe_accuracy'],
                           s=200, c=colors, alpha=0.7, edgecolor='black', linewidth=2)
        
        for i, model in enumerate(df['model']):
            ax.annotate(model, (df['avg_sparsity'].iloc[i], df['avg_probe_accuracy'].iloc[i]),
                       xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax.set_xlabel('Average Sparsity')
        ax.set_ylabel('Probe Accuracy')
        ax.set_title('D. Probe Accuracy vs Sparsity', fontweight='bold')
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_effect_size_distribution(self):
        """Figure 4: Effect size (Cohen's d) distribution"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        df = self.summary_df
        
        # Panel A: Cohen's d by model
        ax = axes[0, 0]
        colors = ['#2E86AB' if 'mamba' in m.lower() or 'ssm' in a.lower() 
                  else '#A23B72' for m, a in zip(df['model'], df['architecture'])]
        
        bars = ax.bar(range(len(df)), df['avg_cohens_d'],
                     color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(df['model'], rotation=45, ha='right')
        ax.set_ylabel("Cohen's d")
        ax.set_title("A. Average Effect Size by Model", fontweight='bold')
        ax.axhline(y=0.2, color='gray', linestyle='--', alpha=0.5, label='Small (0.2)')
        ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Medium (0.5)')
        ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='Large (0.8)')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Panel B: Circuit-specific effect sizes
        ax = axes[0, 1]
        bars = ax.bar(range(len(df)), df['circuit_cohens_d'],
                     color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(df['model'], rotation=45, ha='right')
        ax.set_ylabel("Cohen's d (Circuit Layers Only)")
        ax.set_title("B. Effect Size in Circuit Layers", fontweight='bold')
        ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.5)
        ax.grid(axis='y', alpha=0.3)
        
        # Panel C: Circuit gap
        ax = axes[1, 0]
        bars = ax.bar(range(len(df)), df['circuit_gap'],
                     color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(df['model'], rotation=45, ha='right')
        ax.set_ylabel('Reasoning - Control Gap')
        ax.set_title('C. Similarity Gap in Circuits', fontweight='bold')
        ax.axhline(y=0.18, color='red', linestyle='--', alpha=0.5, label='Threshold (0.18)')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Panel D: Effect size vs circuit rate
        ax = axes[1, 1]
        scatter = ax.scatter(df['avg_cohens_d'], df['circuit_rate'],
                           s=200, c=colors, alpha=0.7, edgecolor='black', linewidth=2)
        
        for i, model in enumerate(df['model']):
            ax.annotate(model, (df['avg_cohens_d'].iloc[i], df['circuit_rate'].iloc[i]),
                       xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax.set_xlabel("Average Cohen's d")
        ax.set_ylabel('Circuit Detection Rate')
        ax.set_title('D. Effect Size vs Detection Rate', fontweight='bold')
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_layerwise_analysis(self):
        """Figure 5: Layer-wise circuit analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Extract layer-wise data for each model
        for result in self.all_results[:min(4, len(self.all_results))]:
            model_name = result['model'].split('/')[-1]
            detailed = result['circuit_detection']['detailed_results']
            
            layers = sorted([int(k) for k in detailed.keys()])
            
            # Panel A: Reasoning similarity by layer
            ax = axes[0, 0]
            reasoning_sims = [detailed[str(l)]['reasoning_similarity_mean'] for l in layers]
            ax.plot(layers, reasoning_sims, marker='o', label=model_name, linewidth=2)
            
            # Panel B: Control similarity by layer
            ax = axes[0, 1]
            control_sims = [detailed[str(l)]['control_similarity_mean'] for l in layers]
            ax.plot(layers, control_sims, marker='s', label=model_name, linewidth=2)
            
            # Panel C: Cohen's d by layer
            ax = axes[1, 0]
            cohens_ds = [detailed[str(l)]['cohens_d'] for l in layers]
            ax.plot(layers, cohens_ds, marker='^', label=model_name, linewidth=2)
            
            # Panel D: Circuit detection by layer
            ax = axes[1, 1]
            is_circuit = [1 if detailed[str(l)]['is_circuit'] else 0 for l in layers]
            ax.plot(layers, is_circuit, marker='D', label=model_name, linewidth=2)
        
        # Configure Panel A
        ax = axes[0, 0]
        ax.set_xlabel('Layer Index')
        ax.set_ylabel('Reasoning Similarity')
        ax.set_title('A. Reasoning Similarity Across Layers', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # Configure Panel B
        ax = axes[0, 1]
        ax.set_xlabel('Layer Index')
        ax.set_ylabel('Control Similarity')
        ax.set_title('B. Control Similarity Across Layers', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # Configure Panel C
        ax = axes[1, 0]
        ax.set_xlabel('Layer Index')
        ax.set_ylabel("Cohen's d")
        ax.set_title("C. Effect Size Across Layers", fontweight='bold')
        ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='Large effect')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # Configure Panel D
        ax = axes[1, 1]
        ax.set_xlabel('Layer Index')
        ax.set_ylabel('Circuit Detected (1=Yes, 0=No)')
        ax.set_title('D. Circuit Detection Across Layers', fontweight='bold')
        ax.set_ylim(-0.1, 1.1)
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_similarity_heatmap(self):
        """Figure 6: Similarity heatmap across models and layers"""
        n_models = len(self.all_results)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Prepare data matrices
        max_layers = max([len(r['circuit_detection']['detailed_results']) 
                         for r in self.all_results])
        
        reasoning_matrix = np.zeros((n_models, max_layers))
        control_matrix = np.zeros((n_models, max_layers))
        
        model_labels = []
        
        for i, result in enumerate(self.all_results):
            model_labels.append(result['model'].split('/')[-1])
            detailed = result['circuit_detection']['detailed_results']
            
            for j, layer_key in enumerate(sorted(detailed.keys(), key=int)):
                if j < max_layers:
                    reasoning_matrix[i, j] = detailed[layer_key]['reasoning_similarity_mean']
                    control_matrix[i, j] = detailed[layer_key]['control_similarity_mean']
        
        # Panel A: Reasoning similarity heatmap
        ax = axes[0]
        im1 = ax.imshow(reasoning_matrix, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)
        ax.set_yticks(range(n_models))
        ax.set_yticklabels(model_labels)
        ax.set_xlabel('Layer Index')
        ax.set_ylabel('Model')
        ax.set_title('A. Reasoning Similarity Heatmap', fontweight='bold')
        plt.colorbar(im1, ax=ax, label='Similarity')
        
        # Panel B: Control similarity heatmap
        ax = axes[1]
        im2 = ax.imshow(control_matrix, aspect='auto', cmap='YlGnBu', vmin=0, vmax=1)
        ax.set_yticks(range(n_models))
        ax.set_yticklabels(model_labels)
        ax.set_xlabel('Layer Index')
        ax.set_ylabel('Model')
        ax.set_title('B. Control Similarity Heatmap', fontweight='bold')
        plt.colorbar(im2, ax=ax, label='Similarity')
        
        plt.tight_layout()
        return fig
    
    def plot_statistical_significance(self):
        """Figure 7: Statistical significance analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        df = self.summary_df
        
        # Panel A: P-value distribution
        ax = axes[0, 0]
        colors = ['#2E86AB' if 'mamba' in m.lower() or 'ssm' in a.lower() 
                  else '#A23B72' for m, a in zip(df['model'], df['architecture'])]
        
        bars = ax.bar(range(len(df)), -np.log10(df['avg_p_value'] + 1e-10),
                     color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(df['model'], rotation=45, ha='right')
        ax.set_ylabel('-log10(p-value)')
        ax.set_title('A. Statistical Significance by Model', fontweight='bold')
        ax.axhline(y=-np.log10(0.05), color='red', linestyle='--', alpha=0.5, label='p=0.05')
        ax.axhline(y=-np.log10(0.01), color='orange', linestyle='--', alpha=0.5, label='p=0.01')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Panel B: P-value vs effect size
        ax = axes[0, 1]
        scatter = ax.scatter(-np.log10(df['avg_p_value'] + 1e-10), df['avg_cohens_d'],
                           s=200, c=colors, alpha=0.7, edgecolor='black', linewidth=2)
        
        for i, model in enumerate(df['model']):
            ax.annotate(model, (-np.log10(df['avg_p_value'].iloc[i] + 1e-10), 
                               df['avg_cohens_d'].iloc[i]),
                       xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax.set_xlabel('-log10(p-value)')
        ax.set_ylabel("Cohen's d")
        ax.set_title('B. Significance vs Effect Size', fontweight='bold')
        ax.axvline(x=-np.log10(0.05), color='red', linestyle='--', alpha=0.5)
        ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.5)
        ax.grid(alpha=0.3)
        
        # Panel C: Confidence intervals for circuit rate
        ax = axes[1, 0]
        
        for i, (idx, row) in enumerate(df.iterrows()):
            # Calculate 95% CI using Wilson score interval
            n = row['total_layers']
            p = row['circuit_rate']
            
            z = 1.96  # 95% confidence
            denominator = 1 + z**2/n
            centre = (p + z**2/(2*n)) / denominator
            adjustment = z * np.sqrt((p*(1-p)/n + z**2/(4*n**2))) / denominator
            
            ci_low = max(0, centre - adjustment)
            ci_high = min(1, centre + adjustment)
            
            ax.errorbar(i, p, yerr=[[p - ci_low], [ci_high - p]], 
                       fmt='o', markersize=10, capsize=5, capthick=2,
                       color=colors[i], ecolor=colors[i], alpha=0.8)
        
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(df['model'], rotation=45, ha='right')
        ax.set_ylabel('Circuit Detection Rate')
        ax.set_title('C. Circuit Rate with 95% CI', fontweight='bold')
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3)
        
        # Panel D: Layer-wise significance
        ax = axes[1, 1]
        
        for result in self.all_results[:min(4, len(self.all_results))]:
            model_name = result['model'].split('/')[-1]
            detailed = result['circuit_detection']['detailed_results']
            
            layers = sorted([int(k) for k in detailed.keys()])
            p_values = [detailed[str(l)]['p_value'] for l in layers]
            
            ax.plot(layers, [-np.log10(p + 1e-10) for p in p_values], 
                   marker='o', label=model_name, linewidth=2)
        
        ax.set_xlabel('Layer Index')
        ax.set_ylabel('-log10(p-value)')
        ax.set_title('D. Significance Across Layers', fontweight='bold')
        ax.axhline(y=-np.log10(0.05), color='red', linestyle='--', alpha=0.5, label='p=0.05')
        ax.axhline(y=-np.log10(0.01), color='orange', linestyle='--', alpha=0.5, label='p=0.01')
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_circuit_neuron_analysis(self):
        """Figure 8: Circuit neuron analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        df = self.summary_df
        
        # Panel A: Total circuit neurons
        ax = axes[0, 0]
        colors = ['#2E86AB' if 'mamba' in m.lower() or 'ssm' in a.lower() 
                  else '#A23B72' for m, a in zip(df['model'], df['architecture'])]
        
        bars = ax.bar(range(len(df)), df['total_circuit_neurons'],
                     color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(df['model'], rotation=45, ha='right')
        ax.set_ylabel('Total Circuit Neurons')
        ax.set_title('A. Total Circuit Neurons by Model', fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Panel B: Circuit neurons vs parameters
        ax = axes[0, 1]
        scatter = ax.scatter(df['parameters_millions'], df['total_circuit_neurons'],
                           s=df['circuit_rate']*500, c=colors, 
                           alpha=0.7, edgecolor='black', linewidth=2)
        
        for i, model in enumerate(df['model']):
            ax.annotate(model, (df['parameters_millions'].iloc[i], 
                               df['total_circuit_neurons'].iloc[i]),
                       xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax.set_xlabel('Model Parameters (Millions)')
        ax.set_ylabel('Total Circuit Neurons')
        ax.set_title('B. Circuit Neurons vs Model Size', fontweight='bold')
        ax.grid(alpha=0.3)
        
        # Panel C: Circuit neuron density
        ax = axes[1, 0]
        bars = ax.bar(range(len(df)), df['circuit_neuron_density'] * 100,
                     color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(df['model'], rotation=45, ha='right')
        ax.set_ylabel('Circuit Neuron Density (%)')
        ax.set_title('C. Circuit Neuron Density', fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Panel D: Neurons per circuit layer
        ax = axes[1, 1]
        
        neurons_per_layer = []
        for _, row in df.iterrows():
            if row['circuits_found'] > 0:
                avg_neurons = row['total_circuit_neurons'] / row['circuits_found']
            else:
                avg_neurons = 0
            neurons_per_layer.append(avg_neurons)
        
        bars = ax.bar(range(len(df)), neurons_per_layer,
                     color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(df['model'], rotation=45, ha='right')
        ax.set_ylabel('Avg Neurons per Circuit Layer')
        ax.set_title('D. Average Circuit Neurons per Layer', fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def _get_cohens_d_distribution(self, model_name: str):
        """Get Cohen's d values for all layers of a model"""
        for result in self.all_results:
            if result['model'] == model_name:
                detailed = result['circuit_detection']['detailed_results']
                return [v['cohens_d'] for v in detailed.values() if 'cohens_d' in v]
        return [0]
    
    def generate_latex_tables(self, output_dir: str = "publication_figures"):
        """Generate LaTeX-formatted tables for publication"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print("="*80)
        print("GENERATING LATEX TABLES")
        print("="*80 + "\n")
        
        df = self.summary_df
        
        # Table 1: Main Results
        print("Creating Table 1: Main Results...")
        
        latex_table1 = "\\begin{table}[h]\n\\centering\n\\caption{Circuit Detection Results Across Models}\n"
        latex_table1 += "\\label{tab:main_results}\n"
        latex_table1 += "\\begin{tabular}{lccccc}\n\\hline\n"
        latex_table1 += "Model & Architecture & Circuits & Total Layers & Detection Rate & Cohen's d \\\\\n\\hline\n"
        
        for _, row in df.iterrows():
            latex_table1 += f"{row['model']} & {row['architecture']} & "
            latex_table1 += f"{row['circuits_found']} & {row['total_layers']} & "
            latex_table1 += f"{row['circuit_rate']:.3f} & {row['avg_cohens_d']:.3f} \\\\\n"
        
        latex_table1 += "\\hline\n\\end{tabular}\n\\end{table}\n"
        
        with open(output_path / "table1_main_results.tex", 'w') as f:
            f.write(latex_table1)
        
        # Table 2: Sparsity Analysis
        print("Creating Table 2: Sparsity Analysis...")
        
        latex_table2 = "\\begin{table}[h]\n\\centering\n\\caption{Circuit Sparsity Analysis}\n"
        latex_table2 += "\\label{tab:sparsity}\n"
        latex_table2 += "\\begin{tabular}{lcccc}\n\\hline\n"
        latex_table2 += "Model & Circuit Neurons & Sparsity & Probe Acc & Density (\\%) \\\\\n\\hline\n"
        
        for _, row in df.iterrows():
            latex_table2 += f"{row['model']} & {row['total_circuit_neurons']} & "
            latex_table2 += f"{row['avg_sparsity']:.3f} & {row['avg_probe_accuracy']:.3f} & "
            latex_table2 += f"{row['circuit_neuron_density']*100:.2f} \\\\\n"
        
        latex_table2 += "\\hline\n\\end{tabular}\n\\end{table}\n"
        
        with open(output_path / "table2_sparsity.tex", 'w') as f:
            f.write(latex_table2)
        
        # Table 3: Statistical Significance
        print("Creating Table 3: Statistical Analysis...")
        
        latex_table3 = "\\begin{table}[h]\n\\centering\n\\caption{Statistical Significance Analysis}\n"
        latex_table3 += "\\label{tab:statistics}\n"
        latex_table3 += "\\begin{tabular}{lcccc}\n\\hline\n"
        latex_table3 += "Model & Avg p-value & Effect Size & Circuit Gap & Rel. Diff \\\\\n\\hline\n"
        
        for _, row in df.iterrows():
            latex_table3 += f"{row['model']} & "
            latex_table3 += f"{row['avg_p_value']:.4f} & {row['avg_cohens_d']:.3f} & "
            latex_table3 += f"{row['circuit_gap']:.3f} & -- \\\\\n"
        
        latex_table3 += "\\hline\n\\end{tabular}\n\\end{table}\n"
        
        with open(output_path / "table3_statistics.tex", 'w') as f:
            f.write(latex_table3)
        
        print(f"\n✓ LaTeX tables saved to: {output_path}/\n")
    
    def generate_comprehensive_report(self, output_dir: str = "publication_figures"):
        """Generate a comprehensive text report"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print("="*80)
        print("GENERATING COMPREHENSIVE REPORT")
        print("="*80 + "\n")
        
        df = self.summary_df
        
        report = []
        report.append("="*80)
        report.append("COMPREHENSIVE MECHANISTIC ANALYSIS REPORT")
        report.append("Nature Machine Intelligence - Ready for Publication")
        report.append("="*80)
        report.append("")
        
        # Executive Summary
        report.append("EXECUTIVE SUMMARY")
        report.append("-" * 80)
        report.append(f"Models Analyzed: {len(df)}")
        report.append(f"Total Layers Tested: {df['total_layers'].sum()}")
        report.append(f"Total Circuits Found: {df['circuits_found'].sum()}")
        report.append(f"Average Circuit Rate: {df['circuit_rate'].mean():.3f} ± {df['circuit_rate'].std():.3f}")
        report.append(f"Average Effect Size (Cohen's d): {df['avg_cohens_d'].mean():.3f} ± {df['avg_cohens_d'].std():.3f}")
        report.append(f"Total Circuit Neurons: {df['total_circuit_neurons'].sum()}")
        report.append("")
        
        # Per-Model Analysis
        report.append("DETAILED MODEL ANALYSIS")
        report.append("-" * 80)
        
        for i, (_, row) in enumerate(df.iterrows(), 1):
            report.append(f"\n{i}. {row['model']}")
            report.append(f"   Architecture: {row['architecture']}")
            report.append(f"   Parameters: {row['parameters_millions']:.1f}M")
            report.append(f"   ")
            report.append(f"   Circuit Detection:")
            report.append(f"      • Circuits Found: {row['circuits_found']}/{row['total_layers']} ({row['circuit_rate']:.1%})")
            report.append(f"      • Effect Size (Cohen's d): {row['avg_cohens_d']:.3f}")
            report.append(f"      • Statistical Significance (p): {row['avg_p_value']:.6f}")
            report.append(f"   ")
            report.append(f"   Similarity Metrics:")
            report.append(f"      • Reasoning Similarity: {row['avg_reasoning_similarity']:.3f}")
            report.append(f"      • Control Similarity: {row['avg_control_similarity']:.3f}")
            report.append(f"      • Circuit Gap: {row['circuit_gap']:.3f}")
            report.append(f"   ")
            report.append(f"   Circuit Sparsity:")
            report.append(f"      • Total Circuit Neurons: {row['total_circuit_neurons']}")
            report.append(f"      • Sparsity: {row['avg_sparsity']:.3f}")
            report.append(f"      • Neuron Density: {row['circuit_neuron_density']*100:.2f}%")
            report.append(f"      • Probe Accuracy: {row['avg_probe_accuracy']:.3f}")
            report.append(f"   ")
            report.append(f"   Task Coverage:")
            report.append(f"      • Reasoning Tasks: {row['n_reasoning_tasks']}")
            report.append(f"      • Control Tasks: {row['n_control_tasks']}")
            report.append(f"      • GSM8K Tasks: {row['n_gsm8k_tasks']}")
        
        # Architecture Comparison
        report.append("\n")
        report.append("ARCHITECTURE COMPARISON")
        report.append("-" * 80)
        
        arch_groups = df.groupby('architecture')
        for arch_name, group in arch_groups:
            report.append(f"\n{arch_name.upper()} Architecture:")
            report.append(f"   Models: {len(group)}")
            report.append(f"   Avg Circuit Rate: {group['circuit_rate'].mean():.3f} ± {group['circuit_rate'].std():.3f}")
            report.append(f"   Avg Effect Size: {group['avg_cohens_d'].mean():.3f} ± {group['avg_cohens_d'].std():.3f}")
            report.append(f"   Avg Sparsity: {group['avg_sparsity'].mean():.3f} ± {group['avg_sparsity'].std():.3f}")
            report.append(f"   Total Circuit Neurons: {group['total_circuit_neurons'].sum()}")
        
        # Statistical Tests
        if len(df['architecture'].unique()) >= 2:
            report.append("\n")
            report.append("STATISTICAL COMPARISON")
            report.append("-" * 80)
            
            archs = df['architecture'].unique()
            arch1, arch2 = archs[0], archs[1]
            group1 = df[df['architecture'] == arch1]
            group2 = df[df['architecture'] == arch2]
            
            metrics = ['circuit_rate', 'avg_cohens_d', 'avg_sparsity']
            
            report.append(f"\nComparing {arch1} vs {arch2}:")
            
            for metric in metrics:
                if len(group1) > 1 and len(group2) > 1:
                    stat, p = stats.mannwhitneyu(group1[metric].dropna(), 
                                                 group2[metric].dropna(), 
                                                 alternative='two-sided')
                    
                    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
                    
                    report.append(f"   {metric}: p={p:.4f} {sig}")
        
        # Key Findings
        report.append("\n")
        report.append("KEY FINDINGS")
        report.append("-" * 80)
        
        report.append(f"\n1. Circuit Detection:")
        report.append(f"   • {df['circuits_found'].sum()} circuits detected across {len(df)} models")
        report.append(f"   • Average detection rate: {df['circuit_rate'].mean():.1%}")
        report.append(f"   • Effect sizes range from {df['avg_cohens_d'].min():.3f} to {df['avg_cohens_d'].max():.3f}")
        
        report.append(f"\n2. Circuit Sparsity:")
        report.append(f"   • Total {df['total_circuit_neurons'].sum()} circuit neurons identified")
        report.append(f"   • Average sparsity: {df['avg_sparsity'].mean():.1%}")
        report.append(f"   • Average neuron density: {(df['circuit_neuron_density']*100).mean():.2f}%")
        
        report.append(f"\n3. Architecture Differences:")
        for arch_name, group in arch_groups:
            report.append(f"   • {arch_name}: {group['circuit_rate'].mean():.1%} circuit rate, ")
            report.append(f"     d={group['avg_cohens_d'].mean():.3f}")
        
        report.append("\n")
        report.append("="*80)
        report.append("END OF REPORT")
        report.append("="*80)
        
        # Save report
        report_text = "\n".join(report)
        
        with open(output_path / "comprehensive_report.txt", 'w') as f:
            f.write(report_text)
        
        print(f"✓ Comprehensive report saved to: {output_path}/comprehensive_report.txt\n")
        
        # Also print to console
        print(report_text)
        
        return report_text
    
    def export_to_excel(self, output_dir: str = "publication_figures"):
        """Export all data to Excel for further analysis"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print("="*80)
        print("EXPORTING TO EXCEL")
        print("="*80 + "\n")
        
        excel_file = output_path / "complete_analysis_data.xlsx"
        
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            # Sheet 1: Summary
            self.summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Sheet 2: Layer-wise details
            layer_data = []
            for result in self.all_results:
                model = result['model'].split('/')[-1]
                detailed = result['circuit_detection']['detailed_results']
                
                for layer, metrics in detailed.items():
                    row = {
                        'model': model,
                        'layer': int(layer),
                        **metrics
                    }
                    layer_data.append(row)
            
            layer_df = pd.DataFrame(layer_data)
            layer_df.to_excel(writer, sheet_name='Layer_Details', index=False)
            
            # Sheet 3: Architecture comparison
            arch_comparison = self.summary_df.groupby('architecture').agg({
                'circuit_rate': ['mean', 'std', 'min', 'max'],
                'avg_cohens_d': ['mean', 'std', 'min', 'max'],
                'avg_sparsity': ['mean', 'std', 'min', 'max'],
                'total_circuit_neurons': 'sum'
            })
            arch_comparison.to_excel(writer, sheet_name='Architecture_Comparison')
        
        print(f"✓ Excel file saved to: {excel_file}\n")

def main():
    """Main execution function"""
    print("\n" + "#"*80)
    print("# COMPREHENSIVE ANALYSIS & VISUALIZATION SUITE")
    print("# For Nature Machine Intelligence Publication")
    print("#"*80 + "\n")
    
    # Initialize analyzer
    analyzer = PublicationAnalyzer(results_dir=".")
    
    # Load all results
    analyzer.load_all_results()
    
    # Create summary dataframe
    analyzer.create_summary_dataframe()
    
    # Generate all visualizations
    figures = analyzer.generate_all_visualizations(output_dir="publication_figures")
    
    # Generate LaTeX tables
    analyzer.generate_latex_tables(output_dir="publication_figures")
    
    # Generate comprehensive report
    analyzer.generate_comprehensive_report(output_dir="publication_figures")
    
    # Export to Excel
    analyzer.export_to_excel(output_dir="publication_figures")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print("  • 8 publication-quality figures (PNG, 300 DPI)")
    print("  • 3 LaTeX tables")
    print("  • 1 comprehensive text report")
    print("  • 1 Excel file with all data")
    print("\nAll files saved to: publication_figures/")
    print("\n✓ Ready for Nature Machine Intelligence submission!")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()