"""
PANEL C: Architecture vs Circuit Formation - Improved Label Positioning
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.lines import Line2D

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica'],
    'font.size': 9,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

COLORS = {
    'ssm': '#2E7D32',
    'attention': '#C62828',
}

def load_all_results(results_dir="."):
    """Load all result JSON files"""
    results_dir = Path(results_dir)
    json_files = list(results_dir.glob("complete_analysis_*.json"))
    
    all_data = []
    for filepath in json_files:
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                all_data.append(data)
        except:
            pass
    
    return all_data

def position_labels_avoid_overlap(points, min_distance=0.28):
    """
    Aggressive label positioning with iterative collision resolution
    points: list of (x, y, label, arch_type)
    """
    if not points:
        return []
    
    # Sort by x position, then by y position
    sorted_points = sorted(points, key=lambda p: (p[0], p[1]))
    
    positioned = []
    for x, y, label, arch_type in sorted_points:
        label_y = y
        
        # Iteratively adjust until no collision
        max_iterations = 10
        for iteration in range(max_iterations):
            collision = False
            for px, py, plabel, parch, placed_y in positioned:
                # Check all previously positioned labels
                dist = abs(label_y - placed_y)
                if dist < min_distance:
                    collision = True
                    # Move label away, alternating up/down
                    if arch_type == 'ssm':
                        label_y = placed_y + min_distance
                    else:
                        label_y = placed_y - min_distance
                    break
            
            if not collision:
                break
        
        positioned.append((x, y, label, arch_type, label_y))
    
    return positioned

def create_panel_c(all_results, output_path="panel_c_scatter.pdf"):
    """Create standalone Panel C: Architecture determines circuits"""
    
    fig, ax = plt.subplots(1, 1, figsize=(140/25.4, 55/25.4))
    
    # Extract data
    ssm_data = []
    att_data = []
    
    for result in all_results:
        model_name = result['model'].split('/')[-1][:12]
        arch = result['architecture']
        
        cd = result['circuit_detection']
        circuits = cd['circuits_found']
        total = cd['total_layers_tested']
        rate = (circuits / total * 100) if total > 0 else 0
        
        # Average Cohen's d
        detailed = cd['detailed_results']
        cohens_d_list = [ld['cohens_d'] for ld in detailed.values()]
        avg_d = np.mean(cohens_d_list)
        
        if 'ssm' in arch.lower():
            ssm_data.append((rate, avg_d, model_name))
        else:
            att_data.append((rate, avg_d, model_name))
    
    # Background shading for regions
    ax.axvspan(-10, 45, alpha=0.04, color=COLORS['attention'], zorder=1)
    ax.axvspan(55, 110, alpha=0.04, color=COLORS['ssm'], zorder=1)
    ax.axhspan(-2, -0.1, alpha=0.04, color=COLORS['attention'], zorder=1)
    ax.axhspan(0.5, 2, alpha=0.04, color=COLORS['ssm'], zorder=1)
    
    # Combine and position labels
    all_points = [(x, y, name, 'ssm') for x, y, name in ssm_data] + \
                 [(x, y, name, 'attention') for x, y, name in att_data]
    positioned = position_labels_avoid_overlap(all_points, min_distance=0.25)
    
    # Plot SSM models (circles)
    for rate, avg_d, name in ssm_data:
        ax.scatter(rate, avg_d, s=180, color=COLORS['ssm'], marker='o',
                  edgecolor='white', linewidth=2.5, alpha=0.85, zorder=3)
    
    # Plot Attention models (squares)
    for rate, avg_d, name in att_data:
        ax.scatter(rate, avg_d, s=180, color=COLORS['attention'], marker='s',
                  edgecolor='white', linewidth=2.5, alpha=0.85, zorder=3)
    
    # Add labels with positioning
    for x, y, label, arch_type, label_y in positioned:
        offset_y = label_y - y
        
        # Draw thin connector if label moved
        if abs(offset_y) > 0.08:
            ax.annotate('', xy=(x + 2, label_y), xytext=(x, y),
                       arrowprops=dict(arrowstyle='-', lw=0.4, alpha=0.2, color='gray'),
                       zorder=2)
        
        # Place label
        ax.text(x + 2.8, label_y, label, fontsize=7.5, alpha=0.8, 
               ha='left', va='center', fontweight='500')
    
    # Reference lines
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.3, zorder=2)
    ax.axhline(y=0.8, color=COLORS['ssm'], linestyle='--',
              linewidth=2, alpha=0.6, label='Large effect (d=0.8)', zorder=2)
    ax.axvline(x=50, color='gray', linestyle=':', linewidth=1.5, alpha=0.4, zorder=2)
    
    # Clean labels
    ax.set_xlabel('Circuit Detection Rate (%)', fontsize=11, fontweight='normal', labelpad=10)
    ax.set_ylabel("Effect Size (Cohen's d)", fontsize=11, fontweight='normal', labelpad=10)
    
    # Axis limits
    ax.set_xlim(-8, 115)
    ax.set_ylim(-1.1, 1.65)
    
    # Region labels
    ax.text(20, 1.42, 'No Circuits', ha='center', fontsize=10,
           style='italic', color=COLORS['attention'], alpha=0.6, fontweight='bold')
    ax.text(82, 1.42, 'Consistent Circuits', ha='center', fontsize=10,
           style='italic', color=COLORS['ssm'], alpha=0.6, fontweight='bold')
    
    # Legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['ssm'],
               markersize=11, label='State-Space', markeredgecolor='white', markeredgewidth=2),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=COLORS['attention'],
               markersize=11, label='Attention', markeredgecolor='white', markeredgewidth=2)
    ]
    ax.legend(handles=legend_elements, loc='lower right', framealpha=0.98,
             fontsize=9, edgecolor='lightgray', fancybox=False)
    
    # Clean grid
    ax.grid(True, linestyle='-', alpha=0.12, linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)
    
    # Clean spines
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_linewidth(0.8)
        ax.spines[spine].set_color('#333333')
    
    # Tick parameters
    ax.tick_params(axis='both', which='major', labelsize=9, length=4, width=0.8)
    
    # Title
    ax.set_title('Architecture Determines Circuit Formation', fontsize=12,
                fontweight='bold', pad=15, color='#333333')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save
    print(f"Saving Panel C to {output_path}...")
    plt.savefig(output_path, dpi=900, bbox_inches='tight',
                format='pdf', pad_inches=0.05, facecolor='white')
    
    png_path = output_path.replace('.pdf', '.png')
    plt.savefig(png_path, dpi=600, bbox_inches='tight',
                format='png', pad_inches=0.05, facecolor='white')
    
    print(f"✓ Saved PDF: {output_path}")
    print(f"✓ Saved PNG: {png_path}")
    
    plt.close()

if __name__ == "__main__":
    print("="*60)
    print("PANEL C GENERATOR - ARCHITECTURE VS CIRCUITS")
    print("="*60)
    
    all_results = load_all_results(".")
    
    if not all_results:
        print("\n❌ No results files found!")
    else:
        create_panel_c(all_results)
        print("\n✓ Panel C ready for draw.io assembly!")
    
    print("="*60)