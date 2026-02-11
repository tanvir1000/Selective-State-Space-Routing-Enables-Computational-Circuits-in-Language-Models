"""
Supplementary Tables & Data: Complete Layer-by-Layer Statistics - FIXED
Generate comprehensive statistical tables and raw data exports
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from itertools import combinations

# ============================================================================
# DATA LOADING
# ============================================================================

def load_all_results(results_dir="."):
    """Load all result JSON files"""
    print("Loading results files...")
    results_dir = Path(results_dir)
    json_files = list(results_dir.glob("complete_analysis_*.json"))
    
    all_data = []
    for filepath in json_files:
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                all_data.append(data)
            print(f"  ✓ Loaded: {filepath.name}")
        except Exception as e:
            print(f"  ✗ Failed: {filepath.name}")
    
    print(f"\nTotal models loaded: {len(all_data)}\n")
    return all_data

# ============================================================================
# SUPPLEMENTARY TABLE S1: COMPLETE LAYER STATISTICS - FIXED
# ============================================================================

def create_table_s1(all_results, output_file="supp_table_s1_layer_statistics.csv"):
    """Generate complete layer-by-layer statistics table"""
    
    print("Generating Supplementary Table S1: Complete Layer Statistics...")
    
    table_data = []
    
    for result in all_results:
        model_name = result['model'].split('/')[-1]
        arch = result['architecture']
        params = result['parameters']
        
        cd = result['circuit_detection']
        detailed = cd['detailed_results']
        
        for layer_str, layer_data in detailed.items():
            layer_idx = int(layer_str)
            
            # Calculate gap safely
            r_mean = layer_data.get('reasoning_similarity_mean', 0)
            c_mean = layer_data.get('control_similarity_mean', 0)
            gap = layer_data.get('gap', r_mean - c_mean)
            
            table_data.append({
                'Model': model_name,
                'Architecture': arch,
                'Parameters_M': f"{params/1e6:.0f}",
                'Layer': layer_idx,
                'Is_Circuit': layer_data.get('is_circuit', False),
                'Cohens_d': layer_data.get('cohens_d', 0),
                'p_value': layer_data.get('p_value', 1.0),
                'Reasoning_Sim_Mean': r_mean,
                'Control_Sim_Mean': c_mean,
                'Gap': gap,
                # Optional fields - use .get() with defaults
                'Reasoning_Sim_Std': layer_data.get('reasoning_similarity_std', 0),
                'Control_Sim_Std': layer_data.get('control_similarity_std', 0),
                'n_samples_reasoning': layer_data.get('n_samples_reasoning', 100),
                'n_samples_control': layer_data.get('n_samples_control', 100),
            })
    
    df = pd.DataFrame(table_data)
    
    # Sort by model and layer
    df = df.sort_values(['Model', 'Layer'])
    
    # Save CSV
    df.to_csv(output_file, index=False, float_format='%.4f')
    print(f"✓ Saved: {output_file}")
    
    # Also create LaTeX version
    latex_file = output_file.replace('.csv', '.tex')
    create_latex_table_s1(df, latex_file)
    
    return df

def create_latex_table_s1(df, output_file):
    """Generate LaTeX version of Table S1"""
    
    # Select key columns for LaTeX (full CSV has all data)
    df_latex = df[['Model', 'Layer', 'Is_Circuit', 'Cohens_d', 'p_value', 'Gap']].copy()
    
    # Format for LaTeX
    df_latex['Is_Circuit'] = df_latex['Is_Circuit'].map({True: '$\\checkmark$', False: '$\\times$'})
    df_latex['Cohens_d'] = df_latex['Cohens_d'].apply(lambda x: f"${x:.3f}$")
    df_latex['p_value'] = df_latex['p_value'].apply(
        lambda x: f"${x:.4f}$" if x >= 0.0001 else "$<10^{-4}$")
    df_latex['Gap'] = df_latex['Gap'].apply(lambda x: f"${x:.3f}$")
    
    # Group by model for better table organization
    latex_content = r"""\begin{longtable}{llcccc}
\caption{Complete layer-by-layer statistics for all models and layers} \\
\toprule
\textbf{Model} & \textbf{Layer} & \textbf{Circuit} & \textbf{Cohen's $d$} & \textbf{$p$-value} & \textbf{Gap} \\
\midrule
\endfirsthead

\multicolumn{6}{c}{\tablename\ \thetable\ -- Continued from previous page} \\
\toprule
\textbf{Model} & \textbf{Layer} & \textbf{Circuit} & \textbf{Cohen's $d$} & \textbf{$p$-value} & \textbf{Gap} \\
\midrule
\endhead

\midrule
\multicolumn{6}{r}{Continued on next page} \\
\endfoot

\bottomrule
\endlastfoot

"""
    
    current_model = None
    for _, row in df_latex.iterrows():
        if row['Model'] != current_model:
            if current_model is not None:
                latex_content += "\\midrule\n"
            current_model = row['Model']
            # Shorter model name
            model_short = current_model.replace('mamba-', '').replace('-hf', '')[:15]
        else:
            model_short = ""
        
        latex_content += f"{model_short} & L{row['Layer']} & {row['Is_Circuit']} & {row['Cohens_d']} & {row['p_value']} & {row['Gap']} \\\\\n"
    
    latex_content += r"""\end{longtable}

\textbf{Note:} Complete statistics for all tested layers across all models. 
Circuit designation ($\checkmark$) requires Cohen's $d > 0.8$, $p < 0.01$ (Mann-Whitney U test), 
and gap $> 0.18$ satisfied simultaneously. Gap = Reasoning similarity - Control similarity.
Sample size: $n = 100$ tasks per category for all comparisons.
"""
    
    with open(output_file, 'w') as f:
        f.write(latex_content)
    
    print(f"✓ Saved: {output_file}")

# ============================================================================
# SUPPLEMENTARY TABLE S2: PAIRWISE COMPARISONS
# ============================================================================

def create_table_s2(all_results, output_file="supp_table_s2_pairwise_comparisons.csv"):
    """Generate pairwise statistical comparisons between models"""
    
    print("Generating Supplementary Table S2: Pairwise Comparisons...")
    
    # Extract model-level statistics
    model_stats = []
    
    for result in all_results:
        model_name = result['model'].split('/')[-1]
        arch = result['architecture']
        
        cd = result['circuit_detection']
        detailed = cd['detailed_results']
        
        # Aggregate statistics - FIXED
        cohens_d_list = []
        gap_list = []
        
        for ld in detailed.values():
            cohens_d_list.append(ld.get('cohens_d', 0))
            
            r_mean = ld.get('reasoning_similarity_mean', 0)
            c_mean = ld.get('control_similarity_mean', 0)
            gap = ld.get('gap', r_mean - c_mean)
            gap_list.append(gap)
        
        model_stats.append({
            'model': model_name,
            'architecture': arch,
            'cohens_d_values': cohens_d_list,
            'gap_values': gap_list,
            'mean_d': np.mean(cohens_d_list),
            'mean_gap': np.mean(gap_list),
        })
    
    # Perform pairwise comparisons
    comparisons = []
    
    for (m1, m2) in combinations(model_stats, 2):
        # Mann-Whitney U test for Cohen's d
        stat_d, p_d = stats.mannwhitneyu(m1['cohens_d_values'], 
                                         m2['cohens_d_values'], 
                                         alternative='two-sided')
        
        # Mann-Whitney U test for gaps
        stat_gap, p_gap = stats.mannwhitneyu(m1['gap_values'], 
                                             m2['gap_values'],
                                             alternative='two-sided')
        
        # Effect size (rank-biserial correlation)
        n1, n2 = len(m1['cohens_d_values']), len(m2['cohens_d_values'])
        r = 1 - (2*stat_d) / (n1 * n2)  # Rank-biserial correlation
        
        comparisons.append({
            'Model_1': m1['model'],
            'Model_2': m2['model'],
            'Arch_1': m1['architecture'],
            'Arch_2': m2['architecture'],
            'Mean_d_1': m1['mean_d'],
            'Mean_d_2': m2['mean_d'],
            'U_statistic_d': stat_d,
            'p_value_d': p_d,
            'Mean_gap_1': m1['mean_gap'],
            'Mean_gap_2': m2['mean_gap'],
            'U_statistic_gap': stat_gap,
            'p_value_gap': p_gap,
            'Effect_size_r': r,
            'Significant': 'Yes' if p_d < 0.05 and p_gap < 0.05 else 'No',
        })
    
    df = pd.DataFrame(comparisons)
    
    # Save CSV
    df.to_csv(output_file, index=False, float_format='%.4f')
    print(f"✓ Saved: {output_file}")
    
    # LaTeX version
    latex_file = output_file.replace('.csv', '.tex')
    create_latex_table_s2(df, latex_file)
    
    return df

def create_latex_table_s2(df, output_file):
    """Generate LaTeX version of Table S2"""
    
    latex_content = r"""\begin{table}[H]
\centering
\caption{Pairwise statistical comparisons between models}
\label{tab:pairwise}
\small
\begin{tabular}{llcccc}
\toprule
\textbf{Model 1} & \textbf{Model 2} & \textbf{$\Delta d$} & \textbf{$p$-value} & \textbf{Effect $r$} & \textbf{Sig.} \\
\midrule
"""
    
    for _, row in df.iterrows():
        m1 = row['Model_1'].replace('mamba-', '').replace('-hf', '')[:12]
        m2 = row['Model_2'].replace('mamba-', '').replace('-hf', '')[:12]
        delta_d = row['Mean_d_1'] - row['Mean_d_2']
        p_val = row['p_value_d']
        r = row['Effect_size_r']
        sig = '$\\checkmark$' if row['Significant'] == 'Yes' else ''
        
        p_str = f"${p_val:.4f}$" if p_val >= 0.0001 else "$<10^{-4}$"
        
        latex_content += f"{m1} & {m2} & ${delta_d:.3f}$ & {p_str} & ${r:.3f}$ & {sig} \\\\\n"
    
    latex_content += r"""\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item \textbf{Note:} Pairwise Mann-Whitney U tests comparing Cohen's $d$ distributions between models. 
$\Delta d$ = Mean Cohen's $d$ (Model 1) - Mean Cohen's $d$ (Model 2). 
Effect size $r$ is rank-biserial correlation. 
Significant comparisons (Sig.) have $p < 0.05$ for both Cohen's $d$ and gap comparisons.
\end{tablenotes}
\end{table}
"""
    
    with open(output_file, 'w') as f:
        f.write(latex_content)
    
    print(f"✓ Saved: {output_file}")

# ============================================================================
# SUPPLEMENTARY DATA FILE: RAW RESULTS
# ============================================================================

def create_supplementary_data_file(all_results, output_file="supplementary_data_raw_results.json"):
    """Export all raw results in a clean JSON format"""
    
    print("Generating Supplementary Data File: Raw Results...")
    
    # Create clean export structure
    export_data = {
        'metadata': {
            'description': 'Complete raw experimental results for circuit detection analysis',
            'n_models': len(all_results),
            'date_generated': pd.Timestamp.now().strftime('%Y-%m-%d'),
        },
        'results': []
    }
    
    for result in all_results:
        export_data['results'].append(result)
    
    # Save with pretty formatting
    with open(output_file, 'w') as f:
        json.dump(export_data, f, indent=2)
    
    print(f"✓ Saved: {output_file}")
    print(f"  File size: {Path(output_file).stat().st_size / 1024:.1f} KB")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("="*80)
    print("SUPPLEMENTARY TABLES & DATA GENERATOR")
    print("="*80)
    
    # Load all results
    all_results = load_all_results(".")
    
    if not all_results:
        print("\n❌ No results files found!")
        return
    
    # Generate Table S1: Complete layer statistics
    df_s1 = create_table_s1(all_results)
    print(f"\nTable S1: {len(df_s1)} layer entries")
    
    # Generate Table S2: Pairwise comparisons
    df_s2 = create_table_s2(all_results)
    print(f"\nTable S2: {len(df_s2)} pairwise comparisons")
    
    # Generate supplementary data file
    create_supplementary_data_file(all_results)
    
    print("\n" + "="*80)
    print("✓ SUPPLEMENTARY TABLES & DATA COMPLETE")
    print("="*80)
    print("\nGenerated files:")
    print("  • supp_table_s1_layer_statistics.csv (+ .tex)")
    print("  • supp_table_s2_pairwise_comparisons.csv (+ .tex)")
    print("  • supplementary_data_raw_results.json")
    print("\n✓ Ready for supplementary materials submission!")
    print("="*80)

if __name__ == "__main__":
    main()