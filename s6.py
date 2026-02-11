"""
Supplementary Figure S5: Hybrid & Efficient Architectures Extended Analysis
Shows why Phi-2 (hybrid) and Gemma (efficient) also fail to form circuits
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import matplotlib.gridspec as gridspec
from matplotlib import rcParams

# ============================================================================
# PUBLICATION SETTINGS
# ============================================================================

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica'],
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

COLORS = {
    'ssm': '#2E7D32',
    'attention': '#C62828',
    'hybrid': '#EF6C00',
    'efficient': '#1565C0',
    'circuit_yes': '#4CAF50',
    'circuit_no': '#EF5350',
}

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
        except:
            pass
    
    print(f"\nTotal models loaded: {len(all_data)}\n")
    return all_data

def extract_hybrid_efficient_data(all_results):
    """Extract data for Phi-2 and Gemma models"""
    
    target_models = ['phi-2', 'gemma-2b-it']
    model_data = {}
    
    for result in all_results:
        model_name = result['model'].split('/')[-1].lower()
        is_target = any(target in model_name for target in target_models)
        
        if is_target:
            model_key = 'phi-2' if 'phi' in model_name else 'gemma-2b'
            cd = result['circuit_detection']
            detailed = cd['detailed_results']
            
            layer_data = []
            for layer_str, ld in detailed.items():
                layer_idx = int(layer_str)
                layer_data.append({
                    'layer': layer_idx,
                    'cohens_d': ld.get('cohens_d', 0),
                    'p_value': ld.get('p_value', 1.0),
                    'reasoning_sim': ld.get('reasoning_similarity_mean', 0),
                    'control_sim': ld.get('control_similarity_mean', 0),
                    'gap': ld.get('gap', ld.get('reasoning_similarity_mean', 0) - ld.get('control_similarity_mean', 0)),
                    'is_circuit': ld.get('is_circuit', False),
                })
            
            model_data[model_key] = {
                'architecture': result['architecture'],
                'parameters': result['parameters'],
                'layers': pd.DataFrame(layer_data),
                'circuits_found': cd['circuits_found'],
                'total_layers': cd['total_layers_tested'],
            }
    
    return model_data

# ============================================================================
# PANEL A: PHI-2 HYBRID ARCHITECTURE ANALYSIS
# ============================================================================

def plot_panel_a(ax, model_data):
    """Panel A: Phi-2 circuit detection summary"""
    
    if 'phi-2' not in model_data:
        ax.text(0.5, 0.5, 'Phi-2 data not found', ha='center', va='center',
               transform=ax.transAxes)
        return
    
    phi_data = model_data['phi-2']
    df = phi_data['layers']
    layers = df['layer'].values
    is_circuit = df['is_circuit'].values
    cohens_d = df['cohens_d'].values
    colors = [COLORS['circuit_yes'] if c else COLORS['circuit_no'] for c in is_circuit]
    
    bars = ax.bar(range(len(layers)), cohens_d, color=colors, alpha=0.8,
                  edgecolor='white', linewidth=1.5)
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.4)
    ax.axhline(y=0.8, color=COLORS['ssm'], linestyle='--', linewidth=1.5,
              alpha=0.6, label='Circuit threshold (d=0.8)')
    ax.axhline(y=-0.8, color=COLORS['attention'], linestyle='--', linewidth=1.5,
              alpha=0.6)
    
    ax.set_xlabel('Layer Index', fontsize=10, labelpad=8)
    ax.set_ylabel("Cohen's d", fontsize=10, labelpad=8)
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels([f'L{l}' for l in layers], fontsize=8)
    
    circuits = phi_data['circuits_found']
    total = phi_data['total_layers']
    summary_text = f"Circuits: {circuits}/{total} (0%)\nHybrid Architecture"
    ax.text(0.98, 0.98, summary_text, transform=ax.transAxes,
           ha='right', va='top', fontsize=8,
           bbox=dict(boxstyle='round', facecolor=COLORS['hybrid'], 
                    alpha=0.3, edgecolor=COLORS['hybrid'], linewidth=2))
    
    ax.legend(loc='center', framealpha=0.95, fontsize=7, edgecolor='lightgray')
    ax.grid(axis='y', linestyle='-', alpha=0.1, linewidth=0.3)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.text(-0.15, 1.12, 'A. Phi-2: Hybrid Architecture', transform=ax.transAxes,
           fontsize=12, fontweight='bold', va='bottom')
   

# ============================================================================
# PANEL B: GEMMA EFFICIENT ARCHITECTURE ANALYSIS
# ============================================================================

def plot_panel_b(ax, model_data):
    """Panel B: Gemma circuit detection summary"""
    
    if 'gemma-2b' not in model_data:
        ax.text(0.5, 0.5, 'Gemma data not found', ha='center', va='center',
               transform=ax.transAxes)
        return
    
    gemma_data = model_data['gemma-2b']
    df = gemma_data['layers']
    layers = df['layer'].values
    is_circuit = df['is_circuit'].values
    cohens_d = df['cohens_d'].values
    colors = [COLORS['circuit_yes'] if c else COLORS['circuit_no'] for c in is_circuit]
    
    bars = ax.bar(range(len(layers)), cohens_d, color=colors, alpha=0.8,
                  edgecolor='white', linewidth=1.5)
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.4)
    ax.axhline(y=0.8, color=COLORS['ssm'], linestyle='--', linewidth=1.5,
              alpha=0.6, label='Circuit threshold (d=0.8)')
    ax.axhline(y=-0.8, color=COLORS['attention'], linestyle='--', linewidth=1.5,
              alpha=0.6)
    
    ax.set_xlabel('Layer Index', fontsize=10, labelpad=8)
    ax.set_ylabel("Cohen's d", fontsize=10, labelpad=8)
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels([f'L{l}' for l in layers], fontsize=8)
    
    circuits = gemma_data['circuits_found']
    total = gemma_data['total_layers']
    summary_text = f"Circuits: {circuits}/{total} (0%)\nEfficient Architecture"
    ax.text(0.98, 0.98, summary_text, transform=ax.transAxes,
           ha='right', va='top', fontsize=8,
           bbox=dict(boxstyle='round', facecolor=COLORS['efficient'],
                    alpha=0.3, edgecolor=COLORS['efficient'], linewidth=2))
    
    ax.legend(loc='center', framealpha=0.95, fontsize=7, edgecolor='lightgray')
    ax.grid(axis='y', linestyle='-', alpha=0.1, linewidth=0.3)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.text(-0.15, 1.12, 'B. Gemma-2B: Efficient Architecture', transform=ax.transAxes,
           fontsize=12, fontweight='bold', va='bottom')
   

# ============================================================================
# PANEL C: SIMILARITY PATTERNS COMPARISON
# ============================================================================

def plot_panel_c(ax, model_data):
    """Panel C: Compare similarity patterns across hybrid/efficient models"""
    
    models_to_plot = []
    for model_key in ['phi-2', 'gemma-2b']:
        if model_key in model_data:
            df = model_data[model_key]['layers']
            models_to_plot.append((model_key, df))
    
    if not models_to_plot:
        return
    
    x_offset = 0
    for model_key, df in models_to_plot:
        x = np.arange(len(df)) + x_offset
        color = COLORS['hybrid'] if model_key == 'phi-2' else COLORS['efficient']
        label = 'Phi-2' if model_key == 'phi-2' else 'Gemma-2B'
        
        ax.plot(x, df['reasoning_sim'].values, marker='o', markersize=6,
               linewidth=2, label=f'{label} Reasoning', color=color, alpha=0.8)
        ax.plot(x, df['control_sim'].values, marker='s', markersize=6,
               linewidth=2, linestyle='--', label=f'{label} Control',
               color=color, alpha=0.5)
        x_offset += len(df) + 1
    
    ax.set_xlabel('Layer (by model)', fontsize=10, labelpad=8)
    ax.set_ylabel('Cosine Similarity', fontsize=10, labelpad=8)
    ax.legend(loc='best', framealpha=0.95, fontsize=7, ncol=2, edgecolor='lightgray')
    ax.grid(axis='y', linestyle='-', alpha=0.1, linewidth=0.3)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.text(-0.15, 1.12, 'C. Similarity Patterns Show No Specialization', transform=ax.transAxes,
           fontsize=12, fontweight='bold', va='bottom')


# ============================================================================
# MAIN FIGURE GENERATION (PANELS A–C ONLY)
# ============================================================================

def create_supp_figure_s5(all_results, output_path="supp_figure_s6_additional_archs.pdf"):
    """Create Supplementary Figure S5 (Panels A–C only)"""
    
    print("\nGenerating Supplementary Figure S5: Additional Architectures (Panels A–C)...")
    model_data = extract_hybrid_efficient_data(all_results)
    
    if not model_data:
        print("⚠️ No Phi-2 or Gemma data found!")
        return
    
    print(f"Found data for: {list(model_data.keys())}")
    
    fig = plt.figure(figsize=(190/25.4, 120/25.4))
    gs = gridspec.GridSpec(2, 2, figure=fig,
                          hspace=0.55, wspace=0.4,
                          left=0.1, right=0.96,
                          top=0.93, bottom=0.08)
    
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])  # C spans bottom row
    
    plot_panel_a(ax_a, model_data)
    plot_panel_b(ax_b, model_data)
    plot_panel_c(ax_c, model_data)
    
    fig.suptitle('Hybrid and Efficient Architectures Also Fail to Form Circuits',
                fontsize=12, fontweight='bold', y=1.05)
    
    print(f"Saving to {output_path}...")
    plt.savefig(output_path, dpi=600, bbox_inches='tight',
                format='pdf', pad_inches=0.1, facecolor='white')
    png_path = output_path.replace('.pdf', '.png')
    plt.savefig(png_path, dpi=600, bbox_inches='tight',
                format='png', pad_inches=0.1, facecolor='white')
    
    print(f"✓ Saved PDF: {output_path}")
    print(f"✓ Saved PNG: {png_path}")
    plt.close()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("="*80)
    print("SUPPLEMENTARY FIGURE S5 GENERATOR - ADDITIONAL ARCHITECTURES (A–C)")
    print("="*80)
    
    all_results = load_all_results(".")
    if not all_results:
        print("\n❌ No results files found!")
        return
    
    create_supp_figure_s5(all_results)
    
    print("\n" + "="*80)
    print("✓ SUPPLEMENTARY FIGURE S5 COMPLETE (Panels A–C)")
    print("="*80)
    print("\nPanels generated:")
    print("  • Panel A: Phi-2 hybrid architecture analysis")
    print("  • Panel B: Gemma efficient architecture analysis")
    print("  • Panel C: Similarity patterns comparison")
    print("\n✓ Ready for supplementary materials!")
    print("="*80)

if __name__ == "__main__":
    main()
