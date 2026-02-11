"""
Figure 2: Circuit Detection Results - NATURE MACHINE INTELLIGENCE VERSION
Professional publication-quality visualization with enhanced typography and clean layout
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import matplotlib.patches as mpatches
from matplotlib import rcParams
import matplotlib.gridspec as gridspec

# ============================================================================
# ENHANCED NATURE MACHINE INTELLIGENCE PUBLICATION SETTINGS
# ============================================================================

# Enhanced Nature-quality parameters
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica'],
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'axes.labelweight': 'normal',
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.titlesize': 12,
    'figure.titleweight': 'bold',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'axes.linewidth': 0.6,
    'grid.linewidth': 0.3,
    'lines.linewidth': 1.0,
    'patch.linewidth': 0.6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.minor.width': 0.3,
    'ytick.minor.width': 0.3,
    'xtick.major.size': 3.5,
    'ytick.major.size': 3.5,
})

# Professional Nature color palette
COLORS = {
    'ssm': '#2E7D32',           # Nature green for SSM
    'ssm_light': '#A5D6A7',     # Light green
    'attention': '#C62828',     # Nature red for Attention
    'attention_light': '#EF9A9A', # Light red
    'hybrid': '#EF6C00',        # Nature orange for Hybrid
    'hybrid_light': '#FFCC80',  # Light orange
    'efficient': '#1565C0',     # Nature blue for Efficient
    'efficient_light': '#90CAF9', # Light blue
    'neutral': '#455A64',       # Dark gray
    'grid': '#ECEFF1',         # Very light gray for grid
    'significance_high': '#1B5E20',
    'significance_med': '#388E3C', 
    'significance_low': '#FF9800',
    'significance_none': '#EF5350',
}

# Architecture groupings
ARCH_GROUPS = {
    'ssm': ['ssm'],
    'attention': ['transformer', 'Dense'],
    'hybrid': ['Hybrid'],
    'efficient': ['Effiecient'],
}

# ============================================================================
# DATA LOADING (unchanged)
# ============================================================================

def load_all_results(results_dir="."):
    """Load all master_results JSON files"""
    print("Loading results files...")
    
    results_dir = Path(results_dir)
    json_files = list(results_dir.glob("master_results_*.json"))
    
    if not json_files:
        json_files = list(results_dir.glob("complete_analysis_*.json"))
    
    all_data = []
    
    for filepath in json_files:
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                
                if 'all_results' in data:
                    for result in data['all_results']:
                        all_data.append(result)
                elif 'model' in data:
                    all_data.append(data)
                    
            print(f"  ✓ Loaded: {filepath.name}")
        except Exception as e:
            print(f"  ✗ Failed: {filepath.name} - {e}")
    
    print(f"\nTotal models loaded: {len(all_data)}\n")
    return all_data

def extract_summary_data(all_results):
    """Extract summary statistics from results"""
    summary = []
    
    for result in all_results:
        model_name = result['model'].split('/')[-1]
        arch = result['architecture']
        
        # Determine architecture group
        arch_group = 'other'
        for group, keywords in ARCH_GROUPS.items():
            if any(kw.lower() in arch.lower() for kw in keywords):
                arch_group = group
                break
        
        cd = result['circuit_detection']
        detailed = cd['detailed_results']
        
        circuits_found = cd['circuits_found']
        total_layers = cd['total_layers_tested']
        circuit_rate = circuits_found / total_layers if total_layers > 0 else 0
        
        cohens_d_list = []
        p_values = []
        reasoning_sims = []
        control_sims = []
        gaps = []
        
        for layer_key, layer_data in detailed.items():
            if 'cohens_d' in layer_data:
                cohens_d_list.append(layer_data['cohens_d'])
                p_values.append(layer_data.get('p_value', 1.0))
                reasoning_sims.append(layer_data['reasoning_similarity_mean'])
                control_sims.append(layer_data['control_similarity_mean'])
                
                gap = layer_data.get('gap', 
                    layer_data['reasoning_similarity_mean'] - layer_data['control_similarity_mean'])
                gaps.append(gap)
        
        summary.append({
            'model': model_name,
            'model_short': model_name.replace('mamba-', '').replace('-hf', ''),
            'architecture': arch,
            'arch_group': arch_group,
            'circuits_found': circuits_found,
            'total_layers': total_layers,
            'circuit_rate': circuit_rate,
            'mean_cohens_d': np.mean(cohens_d_list) if cohens_d_list else 0,
            'std_cohens_d': np.std(cohens_d_list) if cohens_d_list else 0,
            'cohens_d_list': cohens_d_list,
            'mean_p_value': np.mean(p_values) if p_values else 1.0,
            'p_values': p_values,
            'mean_reasoning_sim': np.mean(reasoning_sims) if reasoning_sims else 0,
            'mean_control_sim': np.mean(control_sims) if control_sims else 0,
            'reasoning_sims': reasoning_sims,
            'control_sims': control_sims,
            'mean_gap': np.mean(gaps) if gaps else 0,
        })
    
    return pd.DataFrame(summary)

# ============================================================================
# PANEL A: ENHANCED CIRCUIT DETECTION RATE
# ============================================================================

def plot_panel_a(ax, df):
    """Panel A: Circuit detection rate by model architecture"""
    
    # Sort by architecture group and circuit rate
    arch_order = ['ssm', 'attention', 'hybrid', 'efficient']
    df_sorted = df.sort_values(['arch_group', 'circuit_rate'], 
                              key=lambda x: x.map({k: i for i, k in enumerate(arch_order)}))
    
    x_pos = np.arange(len(df_sorted))
    
    # Enhanced color mapping
    colors = []
    edge_colors = []
    for _, row in df_sorted.iterrows():
        arch = row['arch_group']
        colors.append(COLORS.get(arch + '_light', COLORS[arch]))
        edge_colors.append(COLORS.get(arch, COLORS['neutral']))
    
    # Create clean bars
    bars = ax.bar(x_pos, df_sorted['circuit_rate'] * 100,
                  color=colors, edgecolor=edge_colors, linewidth=0.6,
                  alpha=0.85, zorder=3, width=0.65)
    
    # Enhanced styling
    ax.set_ylabel('Circuit Detection Rate (%)', fontsize=10, labelpad=10)
    # ax.set_xlabel('Model Architecture', fontsize=10, labelpad=10)
    
    # Clean X-axis labels
    ax.set_xticks(x_pos)
    labels = [f"{row['model_short']}" for _, row in df_sorted.iterrows()]
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    
    # Y-axis limits
    max_rate = df_sorted['circuit_rate'].max() * 100
    ax.set_ylim(0, min(105, max(100, max_rate * 1.1)))
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    
    # Subtle reference line
    ax.axhline(y=50, color=COLORS['neutral'], linestyle=':', 
              linewidth=0.6, alpha=0.4, zorder=1)
    
    # Clean value annotations
    for i, (bar, rate) in enumerate(zip(bars, df_sorted['circuit_rate'])):
        height = bar.get_height()
        if height > 5:  # Only label bars above threshold
            va = 'bottom' if height < 85 else 'top'
            y_offset = 1.5 if va == 'bottom' else -1.5
            color = 'black' if height < 85 else 'white'
            
            ax.text(bar.get_x() + bar.get_width()/2., height + y_offset,
                   f'{rate*100:.0f}%', ha='center', va=va, fontsize=8, 
                   fontweight='bold', color=color)
    
    # Clean grid
    ax.yaxis.grid(True, linestyle='-', alpha=0.1, color=COLORS['grid'], linewidth=0.4)
    ax.set_axisbelow(True)
    
    # Clean spines
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_linewidth(0.6)
        ax.spines[spine].set_color(COLORS['neutral'])
    
    # Panel label
    ax.text(-0.1, 1.02, 'A. Circuit detection rate', transform=ax.transAxes,
           fontsize=10, fontweight='bold', va='bottom')

# ============================================================================
# PANEL B: ENHANCED EFFECT SIZE DISTRIBUTION
# ============================================================================

def plot_panel_b(ax, df):
    """Panel B: Distribution of effect sizes across architectures"""
    
    plot_data = []
    positions = []
    colors_list = []
    edge_colors_list = []
    labels = []
    
    pos = 0
    for arch_group in ['ssm', 'attention', 'hybrid', 'efficient']:
        group_df = df[df['arch_group'] == arch_group]
        
        if len(group_df) > 0:
            all_cohens = []
            for _, row in group_df.iterrows():
                all_cohens.extend(row['cohens_d_list'])
            
            if all_cohens:
                plot_data.append(all_cohens)
                positions.append(pos)
                colors_list.append(COLORS.get(arch_group + '_light', COLORS[arch_group]))
                edge_colors_list.append(COLORS.get(arch_group, COLORS['neutral']))
                
                # Calculate summary statistics
                n_models = len(group_df)
                n_layers = len(all_cohens)
                labels.append(f"{arch_group.upper()}\n(n={n_models}M,{n_layers}L)")
                pos += 1
    
    # Create clean violin plot
    if plot_data:
        parts = ax.violinplot(plot_data, positions=positions, widths=0.65,
                              showmeans=True, showmedians=False, showextrema=True)
        
        # Style violins
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors_list[i])
            pc.set_alpha(0.7)
            pc.set_edgecolor(edge_colors_list[i])
            pc.set_linewidth(0.6)
        
        # Style statistical elements
        for partname in ('cbars', 'cmins', 'cmaxes'):
            if partname in parts:
                vp = parts[partname]
                vp.set_edgecolor(COLORS['neutral'])
                vp.set_linewidth(0.6)
                vp.set_alpha(0.7)
        
        # Style mean lines
        parts['cmeans'].set_edgecolor('#FFFFFF')
        parts['cmeans'].set_linewidth(1.5)
    
    # Enhanced styling
    ax.set_ylabel("Cohen's d (Effect Size)", fontsize=10, labelpad=10)
    # ax.set_xlabel('Architecture Group', fontsize=10, labelpad=10)
    
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=8)
    
    # Clean reference lines
    ax.axhline(y=0, color=COLORS['neutral'], linestyle='-', 
              linewidth=0.6, alpha=0.5, zorder=1)
    
    effect_levels = [0.2, 0.5, 0.8]
    effect_labels = ['Small effect', 'Medium effect', 'Large effect']
    effect_colors = ['#FFF59D', '#FFD54F', '#FFA726']
    
    for level, label, color in zip(effect_levels, effect_labels, effect_colors):
        ax.axhline(y=level, color=color, linestyle='--', 
                  linewidth=0.8, alpha=0.6, label=label)
        ax.axhline(y=-level, color=color, linestyle='--', 
                  linewidth=0.8, alpha=0.6)
    
    # Clean legend
    ax.legend(loc='upper right', framealpha=0.95, fontsize=7, 
             edgecolor='lightgray', fancybox=False)
    
    # Professional grid
    ax.yaxis.grid(True, linestyle='-', alpha=0.08, color=COLORS['grid'], linewidth=0.3)
    ax.set_axisbelow(True)
    
    # Clean spines
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_linewidth(0.6)
        ax.spines[spine].set_color(COLORS['neutral'])
    
    # Panel label
    ax.text(-0.1, 1.02, 'B. Effect size distribution', transform=ax.transAxes,
           fontsize=10, fontweight='bold', va='bottom')

# ============================================================================
# PANEL C: ENHANCED SIMILARITY PATTERNS
# ============================================================================

def plot_panel_c(ax, df):
    """Panel C: Reasoning vs control similarity patterns"""
    
    # Enhanced scatter plot
    markers = ['o', 's', '^', 'D']
    marker_size = 70
    
    for i, arch_group in enumerate(['ssm', 'attention', 'hybrid', 'efficient']):
        group_df = df[df['arch_group'] == arch_group]
        
        if len(group_df) > 0:
            color = COLORS.get(arch_group + '_light', COLORS[arch_group])
            edge_color = COLORS.get(arch_group, COLORS['neutral'])
            
            ax.scatter(group_df['mean_control_sim'], 
                      group_df['mean_reasoning_sim'],
                      s=marker_size, alpha=0.8,
                      color=color, marker=markers[i],
                      edgecolor=edge_color, linewidth=0.6,
                      label=arch_group.upper(),
                      zorder=3)
            
            # Clean annotations
            for _, row in group_df.iterrows():
                ax.annotate(row['model_short'], 
                           (row['mean_control_sim'], row['mean_reasoning_sim']),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=7, alpha=0.8, fontweight='normal',
                           bbox=dict(boxstyle='round,pad=0.2', 
                                   facecolor='white', alpha=0.9, 
                                   edgecolor='lightgray', linewidth=0.3))
    
    # Clean diagonal line
    lim_min = min(df['mean_control_sim'].min(), df['mean_reasoning_sim'].min()) - 0.02
    lim_max = max(df['mean_control_sim'].max(), df['mean_reasoning_sim'].max()) + 0.02
    ax.plot([lim_min, lim_max], [lim_min, lim_max], 
           color=COLORS['neutral'], linestyle='--', 
           linewidth=0.8, alpha=0.4, label='Identity line', zorder=1)
    
    # Professional styling
    ax.set_xlabel('Control Task Similarity', fontsize=10, labelpad=10)
    ax.set_ylabel('Reasoning Task Similarity', fontsize=10, labelpad=10)
    
    # Auto-adjust limits
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    
    # Clean legend
    ax.legend(loc='lower right', framealpha=0.95, fontsize=7,
             edgecolor='lightgray', fancybox=False, markerscale=0.8)
    
    # Professional grid
    ax.grid(True, linestyle='-', alpha=0.08, color=COLORS['grid'], linewidth=0.3)
    ax.set_axisbelow(True)
    
    # Equal aspect ratio
    ax.set_aspect('equal', adjustable='box')
    
    # Clean spines
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color(COLORS['neutral'])
    
    # Panel label
    ax.text(-0.1, 1.02, 'C. Similarity pattern', transform=ax.transAxes,
           fontsize=10, fontweight='bold', va='bottom')

# ============================================================================
# PANEL D: ENHANCED STATISTICAL SIGNIFICANCE
# ============================================================================

def plot_panel_d(ax, df):
    """Panel D: Statistical significance across models"""
    
    # Prepare data
    df_sorted = df.sort_values(['arch_group', 'mean_p_value'])
    x_pos = np.arange(len(df_sorted))
    
    # Calculate transformed p-values
    log_p_values = -np.log10(df_sorted['mean_p_value'] + 1e-10)
    
    # Clean color coding
    bar_colors = []
    edge_colors = []
    
    for log_p in log_p_values:
        if log_p >= -np.log10(0.001):
            bar_colors.append(COLORS['significance_high'])
            edge_colors.append('#1B5E20')
        elif log_p >= -np.log10(0.01):
            bar_colors.append(COLORS['significance_med'])
            edge_colors.append('#2E7D32')
        elif log_p >= -np.log10(0.05):
            bar_colors.append(COLORS['significance_low'])
            edge_colors.append('#EF6C00')
        else:
            bar_colors.append(COLORS['significance_none'])
            edge_colors.append('#C62828')
    
    # Create clean bar plot
    bars = ax.bar(x_pos, log_p_values,
                  color=bar_colors, edgecolor=edge_colors,
                  linewidth=0.6, alpha=0.85, zorder=3, width=0.65)
    
    # Enhanced styling
    ax.set_ylabel('-log\u2081\u2080(p-value)', fontsize=10, labelpad=10)
    # ax.set_xlabel('Model Architecture', fontsize=10, labelpad=10)
    
    # Clean X-axis labels
    ax.set_xticks(x_pos)
    labels = [f"{row['model_short']}" for _, row in df_sorted.iterrows()]
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    
    # Clean reference lines
    significance_levels = [0.05, 0.01, 0.001]
    level_colors = ['#FFA726', '#388E3C', '#1B5E20']
    
    for level, color in zip(significance_levels, level_colors):
        y_val = -np.log10(level)
        ax.axhline(y=y_val, color=color, linestyle='--',
                  linewidth=0.8, alpha=0.6, zorder=2)
        
        # Clean significance labels
        ax.text(len(df_sorted) + 0.5, y_val, f'p = {level}', 
               fontsize=7, color=color, va='center', ha='left')
    
    # Clean value annotations
    for i, (bar, log_p) in enumerate(zip(bars, log_p_values)):
        if log_p > 0.5:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
                   f'{log_p:.1f}', ha='center', va='bottom', fontsize=7,
                   fontweight='bold', color=edge_colors[i])
    
    # Enhanced grid
    ax.yaxis.grid(True, linestyle='-', alpha=0.08, color=COLORS['grid'], linewidth=0.3)
    ax.set_axisbelow(True)
    
    # Clean spines
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_linewidth(0.6)
        ax.spines[spine].set_color(COLORS['neutral'])
    
    # Panel label
    ax.text(-0.1, 1.02, 'D. Statistical significance ', transform=ax.transAxes,
           fontsize=10, fontweight='bold', va='bottom')

# ============================================================================
# MAIN FIGURE GENERATION - ENHANCED NATURE MI STANDARDS
# ============================================================================

def create_figure2(df, output_path="figure2_circuit_detection.pdf"):
    """Create professional Figure 2 meeting enhanced Nature MI standards"""
    
    print("\nGenerating enhanced Figure 2 for Nature Machine Intelligence...")
    
    # Create figure with optimal dimensions
    fig = plt.figure(figsize=(180/25.4, 180/25.4))  # 180mm x 180mm
    
    # Enhanced grid specification
    gs = gridspec.GridSpec(2, 2, figure=fig,
                          hspace=0.4, wspace=0.35,
                          left=0.1, right=0.98,
                          top=0.92, bottom=0.08)
    
    # Create subplots
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    
    # Generate all panels
    plot_panel_a(ax_a, df)
    plot_panel_b(ax_b, df)
    plot_panel_c(ax_c, df)
    plot_panel_d(ax_d, df)
    
    # Professional main title
    fig.suptitle('Circuit Detection and Statistical Analysis Across Model Architectures',
                fontsize=12, fontweight='bold', y=0.98)
    
    # Save with enhanced quality settings
    print(f"Saving to {output_path}...")
    plt.savefig(output_path, dpi=600, bbox_inches='tight',
                format='pdf', 
                pad_inches=0.1, facecolor='white', edgecolor='none')
    
    # Also save high-resolution PNG
    png_path = output_path.replace('.pdf', '.png')
    plt.savefig(png_path, dpi=600, bbox_inches='tight', format='png',
                pad_inches=0.1, facecolor='white', edgecolor='none')
    
    print(f"✓ Saved enhanced PDF: {output_path}")
    print(f"✓ Saved high-resolution PNG: {png_path}")
    
    plt.close()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("="*80)
    print("ENHANCED FIGURE 2 GENERATOR - NATURE MACHINE INTELLIGENCE STANDARDS")
    print("="*80)
    
    all_results = load_all_results(".")
    
    if not all_results:
        print("\n❌ No results files found!")
        print("Please ensure master_results_*.json files are in the current directory.")
        return
    
    df = extract_summary_data(all_results)
    
    print("\nDataset Summary:")
    print(f"  Total models: {len(df)}")
    for arch in ['ssm', 'attention', 'hybrid', 'efficient']:
        count = len(df[df['arch_group']==arch])
        if count > 0:
            arch_df = df[df['arch_group']==arch]
            mean_rate = arch_df['circuit_rate'].mean() * 100
            print(f"  {arch.upper()}: {count} models, {mean_rate:.1f}% avg detection")
    
    create_figure2(df)
    
    print("\n" + "="*80)
    print("✓ ENHANCED FIGURE GENERATION COMPLETE")
    print("="*80)
    print("\nGenerated files meet enhanced Nature Machine Intelligence standards:")
    print("  • Clean typography and text adjustments")
    print("  • Professional color scheme")
    print("  • Optimized spacing and layout")
    print("  • High-resolution output (600 DPI)")
    print("  • figure2_circuit_detection.pdf (publication-ready)")
    print("  • figure2_circuit_detection.png (for review)")
    print("\n✓ Ready for submission to Nature Machine Intelligence!")
    print("="*80)

if __name__ == "__main__":
    main()