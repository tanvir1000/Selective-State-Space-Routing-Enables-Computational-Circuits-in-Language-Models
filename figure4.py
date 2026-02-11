"""
Figure 4: Layer-wise Circuit Distribution - NATURE MACHINE INTELLIGENCE VERSION
Professional visualization showing circuit formation consistency across network depth
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
    'ssm': '#2E7D32',
    'ssm_light': '#A5D6A7',
    'ssm_dark': '#1B5E20',
    'attention': '#C62828',
    'attention_light': '#EF9A9A',
    'hybrid': '#EF6C00',
    'hybrid_light': '#FFCC80',
    'efficient': '#1565C0',
    'neutral': '#455A64',
    'grid': '#ECEFF1',
    'circuit_yes': '#2E7D32',
    'circuit_no': '#C62828',
    'circuit_weak': '#FFA726',
}

# ============================================================================
# DATA LOADING AND EXTRACTION
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
            print(f"  ✗ Failed: {filepath.name} - {e}")
    
    print(f"\nTotal models loaded: {len(all_data)}\n")
    return all_data

def extract_layer_data(all_results):
    """Extract layer-by-layer circuit data"""
    layer_data = []
    
    for result in all_results:
        model_name = result['model'].split('/')[-1]
        arch = result['architecture']
        is_ssm = 'ssm' in arch.lower()
        
        cd = result['circuit_detection']
        detailed = cd['detailed_results']
        
        # Get total layers
        n_layers = cd['total_layers_tested']
        layer_indices = cd.get('layer_indices_tested', [])
        
        # Process each layer
        for layer_key, layer_result in detailed.items():
            layer_idx = int(layer_key)
            
            # Determine relative position
            if n_layers > 0:
                relative_position = layer_idx / (n_layers - 1) if n_layers > 1 else 0.5
            else:
                relative_position = 0.5
            
            # Categorize layer depth
            if relative_position < 0.25:
                depth_category = 'Early'
            elif relative_position < 0.75:
                depth_category = 'Middle'
            else:
                depth_category = 'Late'
            
            is_circuit = layer_result.get('is_circuit', False)
            cohens_d = layer_result.get('cohens_d', 0)
            p_value = layer_result.get('p_value', 1.0)
            reasoning_sim = layer_result['reasoning_similarity_mean']
            control_sim = layer_result['control_similarity_mean']
            gap = layer_result.get('gap', reasoning_sim - control_sim)
            
            layer_data.append({
                'model': model_name,
                'model_short': model_name.replace('mamba-', '').replace('-hf', ''),
                'architecture': arch,
                'is_ssm': is_ssm,
                'layer_idx': layer_idx,
                'n_layers': n_layers,
                'relative_position': relative_position,
                'depth_category': depth_category,
                'is_circuit': is_circuit,
                'cohens_d': cohens_d,
                'p_value': p_value,
                'reasoning_sim': reasoning_sim,
                'control_sim': control_sim,
                'gap': gap,
            })
    
    return pd.DataFrame(layer_data)

# ============================================================================
# PANEL A: CIRCUIT DETECTION HEATMAP BY LAYER
# ============================================================================

# ============================================================================
# PANEL A: CIRCUIT DETECTION HEATMAP BY LAYER - FIXED VERSION
# ============================================================================

def plot_panel_a(ax, df):
    """Panel A: Heatmap showing circuit detection across layers and models - FIXED"""
    
    # Print debug info
    print("\n🔍 DEBUG - Available models in dataframe:")
    print(df['model_short'].unique())
    print(df['model'].unique())
    
    # Define display names and how to match them in data
    models_config = [
        ('Llama-3.2-3B-In', ['Llama-3.2-3B', 'Llama-3.2-3B-In']),
        ('gemma-2b-it', ['gemma-2b-it', 'gemma']),
        ('phi-2', ['phi-2', 'phi']),
        ('Mamba-1.4B', ['1.4b', 'mamba-1.4b']),
        ('Mamba-2.8B', ['2.8b', 'mamba-2.8b']),
        ('Mamba-370M', ['370m', 'mamba-370m']),
        ('Mamba-790M', ['790m', 'mamba-790m']),
        ('Qwen2.5-3B', ['Qwen2.5-3B', 'Qwen']),
    ]
    
    display_names = [name for name, _ in models_config]
    
    # Layer labels
    layer_labels = ['Layer 1\n(Early)', 'Layer 2', 'Layer 3\n(Mid)', 'Layer 4', 'Layer 5\n(Late)']
    n_layers = 5
    
    # Initialize matrix
    matrix = np.full((len(models_config), n_layers), np.nan)
    
    # Fill matrix
    for i, (display_name, search_terms) in enumerate(models_config):
        # Find matching rows using any of the search terms
        model_df = pd.DataFrame()
        
        for term in search_terms:
            # Search in both model and model_short columns
            mask = (df['model_short'].str.contains(term, case=False, na=False) | 
                   df['model'].str.contains(term, case=False, na=False))
            matching = df[mask]
            
            if len(matching) > 0:
                model_df = matching
                print(f"✓ Found {len(matching)} layers for {display_name} using term '{term}'")
                break
        
        if len(model_df) == 0:
            print(f"⚠️ No data found for {display_name}")
            continue
        
        # Sort by layer index and take first 5 layers
        model_df = model_df.sort_values('layer_idx').head(n_layers)
        
        # Fill matrix
        for j, (_, row) in enumerate(model_df.iterrows()):
            if j < n_layers:
                matrix[i, j] = 1.0 if row['is_circuit'] else 0.0
                print(f"  Layer {j}: {'Circuit ✓' if row['is_circuit'] else 'No circuit ✗'}")
    
    # Create masked array for NaN values
    masked_matrix = np.ma.masked_invalid(matrix)
    
    # Create heatmap
    cmap = plt.cm.RdYlGn.copy()
    cmap.set_bad(color='lightgray')  # Color for NaN values
    
    im = ax.imshow(masked_matrix, cmap=cmap, aspect='auto', 
                   vmin=0, vmax=1, interpolation='nearest')
    
    # Set ticks
    ax.set_xticks(np.arange(n_layers))
    ax.set_xticklabels(layer_labels, fontsize=7, rotation=0)
    ax.set_yticks(np.arange(len(display_names)))
    ax.set_yticklabels(display_names, fontsize=8)
    
    # Add text annotations
    for i in range(len(display_names)):
        for j in range(n_layers):
            if not np.isnan(matrix[i, j]):
                value = matrix[i, j]
                
                if value == 1.0:
                    # Circuit detected - use bold checkmark
                    text = '✓'
                    text_color = '#FFFFFF'  # White
                    font_size = 14
                    font_weight = 'bold'
                elif value == 0.0:
                    # No circuit - use bold X
                    text = '✗'
                    text_color = '#000000'  # Black
                    font_size = 13
                    font_weight = 'bold'
                else:
                    # Partial circuit
                    text = '◐'
                    text_color = '#333333'
                    font_size = 12
                    font_weight = 'normal'
                
                ax.text(j, i, text, ha='center', va='center',
                       color=text_color, fontsize=font_size, 
                       fontweight=font_weight,
                       family='DejaVu Sans')  # Explicit font family
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02, shrink=0.8)
    cbar.set_label('Circuit Detection Rate', fontsize=8, labelpad=-20)
    cbar.ax.tick_params(labelsize=7)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['No Circuit', 'Circuit'])
    
    # Labels
    ax.set_xlabel('Layer Position', fontsize=9, labelpad=8)
    ax.set_ylabel('Model', fontsize=9, labelpad=8)
    
    # Panel label
    ax.text(-0.2, 1.02, 'A. Circuit detection by depth', 
           transform=ax.transAxes, fontsize=10, fontweight='bold', va='bottom',ha='center')

# ============================================================================
# PANEL B: EFFECT SIZE ACROSS LAYERS - FIXED VERSION
# ============================================================================

def plot_panel_b(ax, df):
    """Panel B: Effect size (Cohen's d) across relative layer positions"""
    
    # SSM models
    ssm_df = df[df['is_ssm'] == True]
    
    # Attention models
    att_df = df[df['is_ssm'] == False]
    
    # Plot SSM models individually with distinct markers/colors
    ssm_models = ssm_df['model_short'].unique()
    ssm_colors = ['#1B5E20', '#2E7D32', '#388E3C', '#43A047', '#4CAF50']  # Different greens
    
    for i, model in enumerate(ssm_models):
        model_df = ssm_df[ssm_df['model_short'] == model].sort_values('relative_position')
        color = ssm_colors[i % len(ssm_colors)]
        
        ax.plot(model_df['relative_position'], model_df['cohens_d'],
               marker='o', markersize=4, linewidth=1.2, alpha=0.8,
               color=color, label=f'SSM {model}' if len(ssm_models) <= 3 else None)
    
    # Plot attention models (averaged) if available
    if len(att_df) > 0:
        att_grouped = att_df.groupby('relative_position')['cohens_d'].mean().reset_index()
        ax.plot(att_grouped['relative_position'], att_grouped['cohens_d'],
               marker='s', markersize=4, linewidth=1.5, alpha=0.8,
               color=COLORS['attention'], linestyle='--', label='Attention (avg)')
    
    # Reference lines
    ax.axhline(y=0, color=COLORS['neutral'], linestyle='-', 
              linewidth=0.6, alpha=0.4)
    ax.axhline(y=0.8, color=COLORS['ssm_dark'], linestyle='--',
              linewidth=0.8, alpha=0.5, label='Large effect (d=0.8)')
    
    # Styling
    ax.set_xlabel('Relative Layer Position', fontsize=9, labelpad=8)
    ax.set_ylabel("Cohen's d (Effect Size)", fontsize=9, labelpad=20)
    ax.set_xlim(-0.05, 1.05)
    
    # Legend - simplified for clarity
    if len(ssm_models) <= 3:
        ax.legend(loc='upper right', framealpha=0.95, fontsize=6,
                 edgecolor='lightgray', fancybox=False)
    else:
        # Just show architecture types, not individual models
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color=COLORS['ssm'], marker='o', linestyle='-', 
                   markersize=4, label='SSM Models'),
            Line2D([0], [0], color=COLORS['attention'], marker='s', linestyle='--', 
                   markersize=4, label='Attention')
        ]
        ax.legend(handles=legend_elements, loc='upper right', framealpha=0.95, 
                 fontsize=6, edgecolor='lightgray', fancybox=False)
    
    # Grid
    ax.grid(True, linestyle='-', alpha=0.1, color=COLORS['grid'], linewidth=0.3)
    ax.set_axisbelow(True)
    
    # Spines
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_linewidth(0.6)
        ax.spines[spine].set_color(COLORS['neutral'])
    
    # Panel label
    ax.text(-0.15, 1.02, 'B. Effect size across depth', 
           transform=ax.transAxes, fontsize=10, fontweight='bold', va='bottom', ha='center')

# Also update the main figure creation to provide more space:
def create_figure4(df, output_path="figure4_layer_distribution.pdf"):
    """Create professional Figure 4 for Nature MI - FIXED VERSION"""
    
    print("\nGenerating Figure 4: Layer-wise Circuit Distribution...")
    
    # Create figure with slightly adjusted dimensions
    fig = plt.figure(figsize=(190/25.4, 190/25.4))  # Slightly larger: 190mm x 190mm
    
    # Grid specification with more horizontal space
    gs = gridspec.GridSpec(2, 2, figure=fig,
                          hspace=0.45, wspace=0.5,  # Increased wspace
                          left=0.10, right=0.98,    # Adjusted margins
                          top=0.92, bottom=0.08)
    
    # Create subplots
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    
    # Generate panels
    plot_panel_a(ax_a, df)
    plot_panel_b(ax_b, df)
    plot_panel_c(ax_c, df)
    plot_panel_d(ax_d, df)
    
    # Main title
    fig.suptitle('Circuits Form Consistently Across Network Depth',
                fontsize=12, fontweight='bold', y=0.98)
    
    # Save
    print(f"Saving to {output_path}...")
    plt.savefig(output_path, dpi=600, bbox_inches='tight',
                format='pdf', pad_inches=0.1, 
                facecolor='white', edgecolor='none')
    
    png_path = output_path.replace('.pdf', '.png')
    plt.savefig(png_path, dpi=600, bbox_inches='tight',
                format='png', pad_inches=0.1,
                facecolor='white', edgecolor='none')
    
    print(f"✓ Saved PDF: {output_path}")
    print(f"✓ Saved PNG: {png_path}")
    
    plt.close()
# ============================================================================
# PANEL B: EFFECT SIZE ACROSS LAYERS
# ============================================================================

def plot_panel_b(ax, df):
    """Panel B: Effect size (Cohen's d) across relative layer positions"""
    
    # SSM models
    ssm_df = df[df['is_ssm'] == True]
    
    # Attention models
    att_df = df[df['is_ssm'] == False]
    
    # Plot SSM models individually
    ssm_models = ssm_df['model_short'].unique()
    
    for i, model in enumerate(ssm_models):
        model_df = ssm_df[ssm_df['model_short'] == model].sort_values('relative_position')
        
        ax.plot(model_df['relative_position'], model_df['cohens_d'],
               marker='o', markersize=5, linewidth=1.5, alpha=0.7,
               color=COLORS['ssm'], label='SSM' if i == 0 else None)
    
    # Plot attention models (averaged)
    if len(att_df) > 0:
        att_grouped = att_df.groupby('relative_position')['cohens_d'].mean().reset_index()
        ax.plot(att_grouped['relative_position'], att_grouped['cohens_d'],
               marker='s', markersize=5, linewidth=1.5, alpha=0.7,
               color=COLORS['attention'], linestyle='--', label='Attention')
    
    # Reference lines
    ax.axhline(y=0, color=COLORS['neutral'], linestyle='-', 
              linewidth=0.6, alpha=0.4)
    ax.axhline(y=0.8, color=COLORS['ssm_dark'], linestyle='--',
              linewidth=0.8, alpha=0.5, label='Large effect (d=0.8)')
    
    # Styling
    ax.set_xlabel('Relative Layer Position', fontsize=10, labelpad=10)
    ax.set_ylabel("Cohen's d (Effect Size)", fontsize=10, labelpad=10)
    ax.set_xlim(-0.05, 1.05)
    
    # Legend
    ax.legend(loc='best', framealpha=0.95, fontsize=7,
             edgecolor='lightgray', fancybox=False)
    
    # Grid
    ax.grid(True, linestyle='-', alpha=0.1, color=COLORS['grid'], linewidth=0.3)
    ax.set_axisbelow(True)
    
    # Spines
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_linewidth(0.6)
        ax.spines[spine].set_color(COLORS['neutral'])
    
    # Panel label
    ax.text(-0.15, 1.02, 'B. Effect size across depth', 
           transform=ax.transAxes, fontsize=10, fontweight='bold', va='bottom')

# ============================================================================
# PANEL C: SIMILARITY GAP BY DEPTH CATEGORY
# ============================================================================

def plot_panel_c(ax, df):
    """Panel C: Grouped bar chart of similarity gaps by depth category"""
    
    # Categories
    depth_categories = ['Early', 'Middle', 'Late']
    
    # SSM data
    ssm_df = df[df['is_ssm'] == True]
    ssm_gaps = []
    ssm_stds = []
    
    for category in depth_categories:
        cat_df = ssm_df[ssm_df['depth_category'] == category]
        if len(cat_df) > 0:
            ssm_gaps.append(cat_df['gap'].mean())
            ssm_stds.append(cat_df['gap'].std())
        else:
            ssm_gaps.append(0)
            ssm_stds.append(0)
    
    # Attention data
    att_df = df[df['is_ssm'] == False]
    att_gaps = []
    att_stds = []
    
    for category in depth_categories:
        cat_df = att_df[att_df['depth_category'] == category]
        if len(cat_df) > 0:
            att_gaps.append(abs(cat_df['gap'].mean()))  # Absolute value
            att_stds.append(cat_df['gap'].std())
        else:
            att_gaps.append(0)
            att_stds.append(0)
    
    # Bar positions
    x = np.arange(len(depth_categories))
    width = 0.35
    
    # Create bars
    bars1 = ax.bar(x - width/2, ssm_gaps, width, yerr=ssm_stds,
                   label='SSM', color=COLORS['ssm_light'], 
                   edgecolor=COLORS['ssm'], linewidth=1.0,
                   capsize=4, alpha=0.8, error_kw={'linewidth': 1.0})
    
    bars2 = ax.bar(x + width/2, att_gaps, width, yerr=att_stds,
                   label='Attention', color=COLORS['attention_light'],
                   edgecolor=COLORS['attention'], linewidth=1.0,
                   capsize=4, alpha=0.8, error_kw={'linewidth': 1.0})
    
    # Reference line
    ax.axhline(y=0.18, color=COLORS['ssm_dark'], linestyle='--',
              linewidth=0.8, alpha=0.5, label='Circuit threshold')
    
    # Annotations
    for i, (bar, gap) in enumerate(zip(bars1, ssm_gaps)):
        if gap > 0.01:
            ax.text(bar.get_x() + bar.get_width()/2., gap + ssm_stds[i] + 0.01,
                   f'{gap:.3f}', ha='center', va='bottom', 
                   fontsize=7, fontweight='bold', color=COLORS['ssm_dark'])
    
    # Styling
    ax.set_ylabel('Similarity Gap\n(Reasoning - Control)', fontsize=10, labelpad=10)
    ax.set_xlabel('Layer Depth Category', fontsize=10, labelpad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(depth_categories, fontsize=9)
    ax.set_ylim(0, 0.25)
    
    # Legend
    ax.legend(loc='center', framealpha=0.95, fontsize=7,
             edgecolor='lightgray', fancybox=False)
    
    # Grid
    ax.yaxis.grid(True, linestyle='-', alpha=0.1, color=COLORS['grid'], linewidth=0.3)
    ax.set_axisbelow(True)
    
    # Spines
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_linewidth(0.6)
        ax.spines[spine].set_color(COLORS['neutral'])
    
    # Panel label
    ax.text(-0.15, 1.02, 'C. Gap by layer depth', 
           transform=ax.transAxes, fontsize=10, fontweight='bold', va='bottom',ha='center')

# ============================================================================
# PANEL D: SUMMARY STATISTICS TABLE
# ============================================================================

def plot_panel_d(ax, df):
    """Panel D: Summary statistics table by depth category"""
    
    # Calculate statistics
    depth_categories = ['Early\n(0-25%)', 'Middle\n(25-75%)', 'Late\n(75-100%)']
    depth_ranges = [('Early', 0, 0.25), ('Middle', 0.25, 0.75), ('Late', 0.75, 1.01)]
    
    table_data = []
    
    for cat_name, cat_label, low, high in [(c, d, *r) for (c, d), (_, *r) in 
                                            zip([(c, d) for c, d in zip(['Early', 'Middle', 'Late'], depth_categories)],
                                                depth_ranges)]:
        # SSM stats
        ssm_df = df[(df['is_ssm'] == True) & 
                   (df['relative_position'] >= low) & 
                   (df['relative_position'] < high)]
        
        if len(ssm_df) > 0:
            ssm_circuit_rate = ssm_df['is_circuit'].mean() * 100
            ssm_mean_d = ssm_df['cohens_d'].mean()
            ssm_mean_gap = ssm_df['gap'].mean()
            ssm_layers = len(ssm_df)
        else:
            ssm_circuit_rate = 0
            ssm_mean_d = 0
            ssm_mean_gap = 0
            ssm_layers = 0
        
        # Attention stats
        att_df = df[(df['is_ssm'] == False) & 
                   (df['relative_position'] >= low) & 
                   (df['relative_position'] < high)]
        
        if len(att_df) > 0:
            att_circuit_rate = att_df['is_circuit'].mean() * 100
            att_layers = len(att_df)
        else:
            att_circuit_rate = 0
            att_layers = 0
        
        table_data.append([
            cat_label,
            f'{ssm_circuit_rate:.0f}%',
            f'{ssm_mean_d:.2f}',
            f'{ssm_mean_gap:.3f}',
            f'{att_circuit_rate:.0f}%',
        ])
    
    # Create table
    columns = ['Depth', 'SSM\nCircuit\nRate', 'SSM\nEffect\nSize', 'SSM\nGap', 'Attn\nCircuit\nRate']
    
    table = ax.table(cellText=table_data, colLabels=columns,
                    cellLoc='center', loc='center',
                    bbox=[0, 0, 1, 1])
    
    # Style table
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 2)
    
    # Color header
    for i in range(len(columns)):
        cell = table[(0, i)]
        cell.set_facecolor(COLORS['neutral'])
        cell.set_text_props(weight='bold', color='white')
    
    # Color cells based on values
    for i in range(1, len(table_data) + 1):
        # SSM circuit rate
        cell = table[(i, 1)]
        rate = float(table_data[i-1][1].strip('%'))
        if rate >= 80:
            cell.set_facecolor(COLORS['ssm_light'])
        elif rate > 0:
            cell.set_facecolor(COLORS['circuit_weak'])
        
        # Attention circuit rate
        cell = table[(i, 4)]
        rate = float(table_data[i-1][4].strip('%'))
        if rate == 0:
            cell.set_facecolor(COLORS['attention_light'])
    
    # Remove axes
    ax.axis('off')
    
    # Panel label
    ax.text(-0.15, 1.02, 'D. Summary statistics', 
           transform=ax.transAxes, fontsize=10, fontweight='bold', va='bottom',ha='center')

# ============================================================================
# MAIN FIGURE GENERATION
# ============================================================================

def create_figure4(df, output_path="figure4_layer_distribution.pdf"):
    """Create professional Figure 4 for Nature MI"""
    
    print("\nGenerating Figure 4: Layer-wise Circuit Distribution...")
    
    # Create figure
    fig = plt.figure(figsize=(180/25.4, 180/25.4))  # 180mm x 180mm
    
    # Grid specification
    gs = gridspec.GridSpec(2, 2, figure=fig,
                          hspace=0.45, wspace=0.4,
                          left=0.12, right=0.98,
                          top=0.92, bottom=0.08)
    
    # Create subplots
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    
    # Generate panels
    plot_panel_a(ax_a, df)
    plot_panel_b(ax_b, df)
    plot_panel_c(ax_c, df)
    plot_panel_d(ax_d, df)
    
    # Main title
    fig.suptitle('Circuits Form Consistently Across Network Depth',
                fontsize=12, fontweight='bold', y=0.98)
    
    # Save
    print(f"Saving to {output_path}...")
    plt.savefig(output_path, dpi=600, bbox_inches='tight',
                format='pdf', pad_inches=0.1, 
                facecolor='white', edgecolor='none')
    
    png_path = output_path.replace('.pdf', '.png')
    plt.savefig(png_path, dpi=600, bbox_inches='tight',
                format='png', pad_inches=0.1,
                facecolor='white', edgecolor='none')
    
    print(f"✓ Saved PDF: {output_path}")
    print(f"✓ Saved PNG: {png_path}")
    
    plt.close()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("="*80)
    print("FIGURE 4 GENERATOR - LAYER-WISE CIRCUIT DISTRIBUTION")
    print("="*80)
    
    all_results = load_all_results(".")
    
    if not all_results:
        print("\n❌ No results files found!")
        return
    
    df = extract_layer_data(all_results)
    
    print("\nLayer-wise Data Summary:")
    print(f"  Total layer measurements: {len(df)}")
    print(f"  SSM layers: {len(df[df['is_ssm']==True])}")
    print(f"  Attention layers: {len(df[df['is_ssm']==False])}")
    
    # Statistics by depth
    print("\n  By depth category:")
    for category in ['Early', 'Middle', 'Late']:
        cat_df = df[df['depth_category'] == category]
        ssm_cat = cat_df[cat_df['is_ssm'] == True]
        att_cat = cat_df[cat_df['is_ssm'] == False]
        
        if len(ssm_cat) > 0:
            ssm_rate = ssm_cat['is_circuit'].mean() * 100
            ssm_d = ssm_cat['cohens_d'].mean()
            print(f"    {category}: SSM {ssm_rate:.0f}% circuits, d={ssm_d:.2f}")
        
        if len(att_cat) > 0:
            att_rate = att_cat['is_circuit'].mean() * 100
            print(f"    {category}: Attention {att_rate:.0f}% circuits")
    
    create_figure4(df)
    
    print("\n" + "="*80)
    print("✓ FIGURE 4 GENERATION COMPLETE")
    print("="*80)
    print("\nKey findings visualized:")
    print("  • Panel A: Circuit detection heatmap across all layers")
    print("  • Panel B: Effect sizes remain high across depth for SSM")
    print("  • Panel C: Specialization gaps by layer category")
    print("  • Panel D: Summary statistics table")
    print("\n✓ Ready for Nature Machine Intelligence submission!")
    print("="*80)

if __name__ == "__main__":
    main()