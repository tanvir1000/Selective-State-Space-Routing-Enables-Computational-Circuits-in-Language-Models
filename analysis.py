"""
COMPREHENSIVE ANALYSIS OF CIRCUIT DETECTION RESULTS
Nature MI Publication Ready

This script loads all model results and performs:
1. Cross-model statistical analysis
2. Architectural comparison
3. Scaling behavior analysis
4. Circuit property characterization
5. Publication-ready visualizations
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")

class ResultsAnalyzer:
    """Comprehensive analysis of circuit detection experiments"""
    
    def __init__(self, results_dir: str = "."):
        self.results_dir = Path(results_dir)
        self.results = {}
        self.architecture_groups = {
            'ssm': [],
            'transformer': [],
            'hybrid': [],
            'efficient': []
        }
        
    def load_all_results(self, file_paths: List[str] = None):
        """Load all JSON result files"""
        print("="*80)
        print("LOADING EXPERIMENTAL RESULTS")
        print("="*80 + "\n")
        
        if file_paths is None:
            # Auto-discover JSON files
            file_paths = list(self.results_dir.glob("complete_analysis_*.json"))
        
        for file_path in file_paths:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    model_name = data['model']
                    self.results[model_name] = data
                    
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
                    
                    print(f"✓ Loaded: {model_name}")
                    print(f"  Architecture: {data['architecture']}")
                    print(f"  Parameters: {data['parameters']:,}")
                    print(f"  Circuits: {data['circuit_detection']['circuits_found']}/{data['circuit_detection']['total_layers_tested']}")
                    print()
                    
            except Exception as e:
                print(f"⚠️ Failed to load {file_path}: {e}\n")
        
        print(f"Total models loaded: {len(self.results)}")
        print(f"  SSM: {len(self.architecture_groups['ssm'])}")
        print(f"  Transformer: {len(self.architecture_groups['transformer'])}")
        print(f"  Hybrid: {len(self.architecture_groups['hybrid'])}")
        print(f"  Efficient: {len(self.architecture_groups['efficient'])}\n")
        
        return self.results
    
    def create_summary_dataframe(self) -> pd.DataFrame:
        """Create comprehensive summary table"""
        print("="*80)
        print("CREATING SUMMARY STATISTICS")
        print("="*80 + "\n")
        
        summary_data = []
        
        for model_name, data in self.results.items():
            cd = data['circuit_detection']
            detailed = cd['detailed_results']
            
            # Extract key metrics
            circuits_found = cd['circuits_found']
            total_layers = cd['total_layers_tested']
            circuit_rate = circuits_found / total_layers if total_layers > 0 else 0
            
            # Compute statistics across all layers
            all_cohens_d = [v['cohens_d'] for v in detailed.values()]
            all_reasoning_sim = [v['reasoning_similarity_mean'] for v in detailed.values()]
            all_control_sim = [v['control_similarity_mean'] for v in detailed.values()]
            all_gaps = [v.get('gap', v['reasoning_similarity_mean'] - v['control_similarity_mean']) 
                       for v in detailed.values()]
            
            # Circuit-specific statistics
            circuit_layers = [k for k, v in detailed.items() if v['is_circuit']]
            circuit_cohens_d = [detailed[k]['cohens_d'] for k in circuit_layers] if circuit_layers else [0]
            circuit_gaps = [detailed[k].get('gap', detailed[k]['reasoning_similarity_mean'] - detailed[k]['control_similarity_mean']) 
                           for k in circuit_layers] if circuit_layers else [0]
            
            summary_data.append({
                'Model': model_name.split('/')[-1],
                'Full_Model_Name': model_name,
                'Architecture': data['architecture'],
                'Parameters': data['parameters'],
                'Circuits_Found': circuits_found,
                'Total_Layers': total_layers,
                'Circuit_Rate': circuit_rate,
                
                # Overall statistics
                'Mean_Cohens_d': np.mean(all_cohens_d),
                'Std_Cohens_d': np.std(all_cohens_d),
                'Mean_Reasoning_Sim': np.mean(all_reasoning_sim),
                'Mean_Control_Sim': np.mean(all_control_sim),
                'Mean_Gap': np.mean(all_gaps),
                
                # Circuit-specific statistics
                'Circuit_Mean_Cohens_d': np.mean(circuit_cohens_d),
                'Circuit_Mean_Gap': np.mean(circuit_gaps),
                
                # Architecture classification
                'Is_SSM': 'ssm' in data['architecture'].lower(),
                'Is_Transformer': 'transformer' in data['architecture'].lower() or 'dense' in data['architecture'].lower(),
            })
        
        df = pd.DataFrame(summary_data)
        df = df.sort_values(['Architecture', 'Parameters'])
        
        print(df[['Model', 'Architecture', 'Circuit_Rate', 'Mean_Cohens_d', 'Mean_Gap']].to_string(index=False))
        print()
        
        return df
    
    def analyze_architectural_differences(self, df: pd.DataFrame):
        """Statistical comparison between architectures"""
        print("="*80)
        print("ARCHITECTURAL COMPARISON ANALYSIS")
        print("="*80 + "\n")
        
        ssm_models = df[df['Is_SSM']]
        transformer_models = df[df['Is_Transformer']]
        
        print(f"📊 SUMMARY STATISTICS\n")
        print(f"State-Space Models (n={len(ssm_models)}):")
        print(f"  Circuit Formation Rate: {ssm_models['Circuit_Rate'].mean():.2%} ± {ssm_models['Circuit_Rate'].std():.2%}")
        print(f"  Mean Cohen's d: {ssm_models['Mean_Cohens_d'].mean():.3f} ± {ssm_models['Mean_Cohens_d'].std():.3f}")
        print(f"  Mean Gap: {ssm_models['Mean_Gap'].mean():.3f} ± {ssm_models['Mean_Gap'].std():.3f}")
        print(f"  Mean Reasoning Similarity: {ssm_models['Mean_Reasoning_Sim'].mean():.3f}")
        print(f"  Mean Control Similarity: {ssm_models['Mean_Control_Sim'].mean():.3f}\n")
        
        print(f"Transformer Models (n={len(transformer_models)}):")
        print(f"  Circuit Formation Rate: {transformer_models['Circuit_Rate'].mean():.2%} ± {transformer_models['Circuit_Rate'].std():.2%}")
        print(f"  Mean Cohen's d: {transformer_models['Mean_Cohens_d'].mean():.3f} ± {transformer_models['Mean_Cohens_d'].std():.3f}")
        print(f"  Mean Gap: {transformer_models['Mean_Gap'].mean():.3f} ± {transformer_models['Mean_Gap'].std():.3f}")
        print(f"  Mean Reasoning Similarity: {transformer_models['Mean_Reasoning_Sim'].mean():.3f}")
        print(f"  Mean Control Similarity: {transformer_models['Mean_Control_Sim'].mean():.3f}\n")
        
        # Statistical tests
        print(f"🔬 STATISTICAL TESTS\n")
        
        # Circuit formation rate comparison
        if len(ssm_models) > 0 and len(transformer_models) > 0:
            # Mann-Whitney U test for circuit rates
            stat, p_circuit = stats.mannwhitneyu(
                ssm_models['Circuit_Rate'], 
                transformer_models['Circuit_Rate'],
                alternative='greater'
            )
            print(f"Circuit Formation Rate:")
            print(f"  Mann-Whitney U test: U={stat:.2f}, p={p_circuit:.6f}")
            print(f"  {'✓ SIGNIFICANT' if p_circuit < 0.05 else '✗ Not significant'} (α=0.05)\n")
            
            # Cohen's d comparison
            stat, p_cohens = stats.mannwhitneyu(
                ssm_models['Mean_Cohens_d'],
                transformer_models['Mean_Cohens_d'],
                alternative='greater'
            )
            print(f"Effect Size (Cohen's d):")
            print(f"  Mann-Whitney U test: U={stat:.2f}, p={p_cohens:.6f}")
            print(f"  {'✓ SIGNIFICANT' if p_cohens < 0.05 else '✗ Not significant'} (α=0.05)\n")
            
            # Gap comparison
            stat, p_gap = stats.mannwhitneyu(
                ssm_models['Mean_Gap'],
                transformer_models['Mean_Gap'],
                alternative='greater'
            )
            print(f"Reasoning-Control Gap:")
            print(f"  Mann-Whitney U test: U={stat:.2f}, p={p_gap:.6f}")
            print(f"  {'✓ SIGNIFICANT' if p_gap < 0.05 else '✗ Not significant'} (α=0.05)\n")
        
        # Effect size between architectures
        if len(ssm_models) > 0 and len(transformer_models) > 0:
            circuit_rate_diff = ssm_models['Circuit_Rate'].mean() - transformer_models['Circuit_Rate'].mean()
            pooled_std = np.sqrt((ssm_models['Circuit_Rate'].var() + transformer_models['Circuit_Rate'].var()) / 2)
            effect_size = circuit_rate_diff / pooled_std if pooled_std > 0 else float('inf')
            
            print(f"📈 EFFECT SIZE ANALYSIS\n")
            print(f"Circuit Formation Rate Difference:")
            print(f"  Absolute difference: {circuit_rate_diff:.2%}")
            print(f"  Cohen's d (architecture): {effect_size:.3f}")
            print(f"  Interpretation: {'Huge' if abs(effect_size) > 1.2 else 'Large' if abs(effect_size) > 0.8 else 'Medium' if abs(effect_size) > 0.5 else 'Small'}\n")
    
    def analyze_scaling_behavior(self, df: pd.DataFrame):
        """Analyze how circuits scale with model size"""
        print("="*80)
        print("SCALING BEHAVIOR ANALYSIS")
        print("="*80 + "\n")
        
        for arch_type in ['ssm', 'transformer']:
            models = df[df['Is_SSM']] if arch_type == 'ssm' else df[df['Is_Transformer']]
            
            if len(models) > 1:
                print(f"📊 {arch_type.upper()} Models:\n")
                
                # Sort by parameters
                models = models.sort_values('Parameters')
                
                # Correlation between size and circuit formation
                if models['Parameters'].var() > 0:
                    corr, p_value = stats.spearmanr(models['Parameters'], models['Circuit_Rate'])
                    print(f"  Parameter Count vs Circuit Rate:")
                    print(f"    Spearman ρ = {corr:.3f}, p = {p_value:.4f}")
                    print(f"    {'✓ Significant correlation' if p_value < 0.05 else '✗ No significant correlation'}\n")
                    
                    corr2, p_value2 = stats.spearmanr(models['Parameters'], models['Mean_Cohens_d'])
                    print(f"  Parameter Count vs Effect Size:")
                    print(f"    Spearman ρ = {corr2:.3f}, p = {p_value2:.4f}")
                    print(f"    {'✓ Significant correlation' if p_value2 < 0.05 else '✗ No significant correlation'}\n")
                
                # Print scaling trajectory
                print(f"  Scaling Trajectory:")
                for _, row in models.iterrows():
                    print(f"    {row['Parameters']/1e9:.2f}B params: {row['Circuit_Rate']:.1%} circuit rate, d={row['Mean_Cohens_d']:.3f}")
                print()
    
    def analyze_layer_wise_patterns(self):
        """Analyze patterns across layers"""
        print("="*80)
        print("LAYER-WISE PATTERN ANALYSIS")
        print("="*80 + "\n")
        
        for arch_type in ['SSM', 'Transformer']:
            print(f"📊 {arch_type} Models:\n")
            
            is_ssm = arch_type == 'SSM'
            relevant_models = [name for name in self.results.keys() 
                             if self.results[name]['architecture'].lower().find('ssm' if is_ssm else 'transformer') >= 0 or
                                self.results[name]['architecture'].lower().find('dense') >= 0 and not is_ssm]
            
            if not relevant_models:
                print("  No models in this category\n")
                continue
            
            # Aggregate statistics by relative layer position
            layer_stats = {
                'early': {'cohens_d': [], 'reasoning_sim': [], 'control_sim': [], 'is_circuit': []},
                'middle': {'cohens_d': [], 'reasoning_sim': [], 'control_sim': [], 'is_circuit': []},
                'late': {'cohens_d': [], 'reasoning_sim': [], 'control_sim': [], 'is_circuit': []}
            }
            
            for model_name in relevant_models:
                data = self.results[model_name]
                layers = data['circuit_detection']['layer_indices_tested']
                detailed = data['circuit_detection']['detailed_results']
                
                n_layers = len(layers)
                for i, layer in enumerate(layers):
                    layer_str = str(layer)
                    if layer_str not in detailed:
                        continue
                    
                    # Categorize by position
                    relative_pos = i / (n_layers - 1) if n_layers > 1 else 0.5
                    if relative_pos < 0.33:
                        category = 'early'
                    elif relative_pos < 0.67:
                        category = 'middle'
                    else:
                        category = 'late'
                    
                    layer_stats[category]['cohens_d'].append(detailed[layer_str]['cohens_d'])
                    layer_stats[category]['reasoning_sim'].append(detailed[layer_str]['reasoning_similarity_mean'])
                    layer_stats[category]['control_sim'].append(detailed[layer_str]['control_similarity_mean'])
                    layer_stats[category]['is_circuit'].append(1 if detailed[layer_str]['is_circuit'] else 0)
            
            # Print statistics
            for position in ['early', 'middle', 'late']:
                stats_dict = layer_stats[position]
                if stats_dict['cohens_d']:
                    print(f"  {position.capitalize()} Layers:")
                    print(f"    Circuit Rate: {np.mean(stats_dict['is_circuit']):.2%}")
                    print(f"    Mean Cohen's d: {np.mean(stats_dict['cohens_d']):.3f} ± {np.std(stats_dict['cohens_d']):.3f}")
                    print(f"    Reasoning Sim: {np.mean(stats_dict['reasoning_sim']):.3f}")
                    print(f"    Control Sim: {np.mean(stats_dict['control_sim']):.3f}")
                    print(f"    Gap: {np.mean(stats_dict['reasoning_sim']) - np.mean(stats_dict['control_sim']):.3f}\n")
            
            print()
    
    def create_visualizations(self, df: pd.DataFrame, save_dir: str = "."):
        """Create publication-quality visualizations"""
        print("="*80)
        print("GENERATING VISUALIZATIONS")
        print("="*80 + "\n")
        
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        # Figure 1: Circuit Formation Rate by Architecture
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Circuit Detection Analysis: Comprehensive Overview', fontsize=16, fontweight='bold')
        
        # 1a: Circuit formation rate
        ax = axes[0, 0]
        arch_order = ['ssm', 'transformer', 'hybrid', 'efficient']
        arch_colors = {'ssm': '#2ecc71', 'transformer': '#e74c3c', 'hybrid': '#f39c12', 'efficient': '#3498db'}
        
        circuit_rates = []
        arch_labels = []
        colors = []
        
        for arch in arch_order:
            arch_models = df[df['Architecture'].str.lower().str.contains(arch)]
            if len(arch_models) > 0:
                circuit_rates.append(arch_models['Circuit_Rate'].values)
                arch_labels.append(f"{arch.upper()}\n(n={len(arch_models)})")
                colors.append(arch_colors.get(arch, '#95a5a6'))
        
        bp = ax.boxplot(circuit_rates, labels=arch_labels, patch_artist=True, showmeans=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_ylabel('Circuit Formation Rate', fontweight='bold')
        ax.set_title('(A) Circuit Formation by Architecture', fontweight='bold')
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)
        ax.grid(axis='y', alpha=0.3)
        
        # 1b: Effect size comparison
        ax = axes[0, 1]
        ssm_cohens = df[df['Is_SSM']]['Mean_Cohens_d'].values
        trans_cohens = df[df['Is_Transformer']]['Mean_Cohens_d'].values
        
        ax.boxplot([ssm_cohens, trans_cohens], 
                   labels=['SSM', 'Transformer'],
                   patch_artist=True,
                   showmeans=True,
                   boxprops=dict(facecolor='#3498db', alpha=0.7))
        ax.set_ylabel("Cohen's d (Effect Size)", fontweight='bold')
        ax.set_title('(B) Effect Size Distribution', fontweight='bold')
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax.axhline(y=0.8, color='green', linestyle='--', alpha=0.3, label='Large effect')
        ax.grid(axis='y', alpha=0.3)
        ax.legend()
        
        # 1c: Similarity patterns
        ax = axes[1, 0]
        width = 0.35
        x = np.arange(2)
        
        ssm_reasoning = df[df['Is_SSM']]['Mean_Reasoning_Sim'].mean()
        ssm_control = df[df['Is_SSM']]['Mean_Control_Sim'].mean()
        trans_reasoning = df[df['Is_Transformer']]['Mean_Reasoning_Sim'].mean()
        trans_control = df[df['Is_Transformer']]['Mean_Control_Sim'].mean()
        
        ax.bar(x - width/2, [ssm_reasoning, trans_reasoning], width, label='Reasoning', color='#2ecc71', alpha=0.8)
        ax.bar(x + width/2, [ssm_control, trans_control], width, label='Control', color='#e74c3c', alpha=0.8)
        
        ax.set_ylabel('Mean Similarity', fontweight='bold')
        ax.set_title('(C) Similarity Patterns', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(['SSM', 'Transformer'])
        ax.legend()
        ax.set_ylim(0, 1)
        ax.grid(axis='y', alpha=0.3)
        
        # 1d: Scaling behavior
        ax = axes[1, 1]
        for is_ssm, label, color, marker in [(True, 'SSM', '#2ecc71', 'o'), 
                                               (False, 'Transformer', '#e74c3c', 's')]:
            models = df[df['Is_SSM'] == is_ssm]
            if len(models) > 0:
                ax.scatter(models['Parameters'] / 1e9, 
                          models['Circuit_Rate'],
                          s=100, alpha=0.7, label=label, color=color, marker=marker)
        
        ax.set_xlabel('Parameters (Billions)', fontweight='bold')
        ax.set_ylabel('Circuit Formation Rate', fontweight='bold')
        ax.set_title('(D) Scaling Behavior', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
        
        plt.tight_layout()
        fig_path = save_path / 'comprehensive_analysis.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {fig_path}")
        plt.close()
        
        # Figure 2: Layer-wise heatmap
        self._create_layer_heatmap(save_path)
        
        print()
    
    def _create_layer_heatmap(self, save_path: Path):
        """Create heatmap of circuit detection across layers"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Layer-wise Circuit Detection Patterns', fontsize=14, fontweight='bold')
        
        for idx, (arch_type, ax) in enumerate([('SSM', axes[0]), ('Transformer', axes[1])]):
            is_ssm = arch_type == 'SSM'
            relevant_models = [name for name in self.results.keys() 
                             if (('ssm' in self.results[name]['architecture'].lower()) == is_ssm)]
            
            if not relevant_models:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center')
                ax.set_title(f'{arch_type} Models')
                continue
            
            # Create matrix: models x layers
            max_layers = max(len(self.results[m]['circuit_detection']['layer_indices_tested']) 
                           for m in relevant_models)
            matrix = np.zeros((len(relevant_models), max_layers))
            matrix[:] = np.nan
            
            model_labels = []
            for i, model_name in enumerate(relevant_models):
                model_labels.append(model_name.split('/')[-1][:20])
                data = self.results[model_name]
                detailed = data['circuit_detection']['detailed_results']
                
                for j, layer in enumerate(data['circuit_detection']['layer_indices_tested']):
                    if str(layer) in detailed:
                        matrix[i, j] = 1 if detailed[str(layer)]['is_circuit'] else 0
            
            # Plot heatmap
            sns.heatmap(matrix, 
                       ax=ax,
                       cmap=['#e74c3c', '#2ecc71'],
                       cbar_kws={'label': 'Circuit Detected'},
                       yticklabels=model_labels,
                       xticklabels=[f'L{i}' for i in range(max_layers)],
                       linewidths=0.5,
                       linecolor='white',
                       vmin=0, vmax=1,
                       mask=np.isnan(matrix))
            
            ax.set_title(f'{arch_type} Models', fontweight='bold')
            ax.set_xlabel('Layer Index', fontweight='bold')
            ax.set_ylabel('Model', fontweight='bold')
        
        plt.tight_layout()
        fig_path = save_path / 'layerwise_heatmap.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {fig_path}")
        plt.close()
    
    def generate_publication_table(self, df: pd.DataFrame, save_path: str = "publication_table.tex"):
        """Generate LaTeX table for publication"""
        print("="*80)
        print("GENERATING PUBLICATION TABLE")
        print("="*80 + "\n")
        
        # Select and format columns
        table_df = df[['Model', 'Architecture', 'Parameters', 'Circuit_Rate', 
                      'Mean_Cohens_d', 'Mean_Gap']].copy()
        
        table_df['Parameters'] = table_df['Parameters'].apply(lambda x: f"{x/1e9:.2f}B")
        table_df['Circuit_Rate'] = table_df['Circuit_Rate'].apply(lambda x: f"{x:.1%}")
        table_df['Mean_Cohens_d'] = table_df['Mean_Cohens_d'].apply(lambda x: f"{x:.3f}")
        table_df['Mean_Gap'] = table_df['Mean_Gap'].apply(lambda x: f"{x:.3f}")
        
        # Rename columns
        table_df.columns = ['Model', 'Architecture', 'Parameters', 'Circuit Rate', 
                           "Cohen's d", 'R-C Gap']
        
        # Generate LaTeX
        latex = table_df.to_latex(index=False, escape=False, column_format='llrrrr')
        
        with open(save_path, 'w') as f:
            f.write(latex)
        
        print(f"✓ Saved LaTeX table: {save_path}\n")
        print("Preview:")
        print(table_df.to_string(index=False))
        print()

# ============================================================================
# MAIN ANALYSIS EXECUTION
# ============================================================================

def main():
    """Run comprehensive analysis"""
    print("\n" + "#"*80)
    print("# COMPREHENSIVE RESULTS ANALYSIS")
    print("# Nature Machine Intelligence Publication Ready")
    print("#"*80 + "\n")
    
    # Initialize analyzer
    analyzer = ResultsAnalyzer()
    
    # Load all results
    # OPTION 1: Manually specify files
    result_files = [
        "complete_analysis_Dense_Llama-3.2-3B-Instruct_20251019_144159.json",
        "complete_analysis_Effiecient_gemma-2b-it_20251019_152241.json",
        "complete_analysis_Hybrid_phi-2_20251019_143233.json",
        "complete_analysis_ssm_mamba-1.4b-hf_20251019_090158.json",
        "complete_analysis_ssm_mamba-2.8b-hf_20251019_153351.json",
        "complete_analysis_ssm_mamba-370m-hf_20251019_090410.json",
        "complete_analysis_ssm_mamba-790m-hf_20251019_090442.json",
        "complete_analysis_transformer_Qwen2.5-3B_20251019_143042.json"
    ]
    
    # OPTION 2: Or auto-discover (uncomment if you want auto-discovery)
    # result_files = None  # Will auto-discover all complete_analysis_*.json files
    
    # For demonstration, let's load from the data you provided
    # You'll need to save your JSON data to files first
    
    print("⚠️ NOTE: Please save your JSON results to files, then specify them above")
    print("   or place them in the current directory with pattern: complete_analysis_*.json\n")
    
    # Uncomment this when you have files:
    analyzer.load_all_results(result_files)
    
    # Create summary dataframe
    df = analyzer.create_summary_dataframe()
    
    # Run analyses
    analyzer.analyze_architectural_differences(df)
    analyzer.analyze_scaling_behavior(df)
    analyzer.analyze_layer_wise_patterns()
    
    # Create visualizations
    analyzer.create_visualizations(df)
    
    # Generate publication table
    analyzer.generate_publication_table(df)
    
    print("="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nGenerated files:")
    print("  • comprehensive_analysis.png - Main figure")
    print("  • layerwise_heatmap.png - Layer patterns")
    print("  • publication_table.tex - LaTeX table")
    print("="*80 + "\n")

if __name__ == '__main__':
    main()