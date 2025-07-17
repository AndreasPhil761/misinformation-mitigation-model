import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import scienceplots
from scipy.interpolate import CubicSpline, interp1d
from scipy.signal import savgol_filter

plt.style.use(['science', 'ieee'])

def create_smooth_interpolation(x_data, y_data, method='cubic_spline'):
    
    # Create smooth interpolation through data points
    # Remove any invalid data points and sort by x
    valid_mask = ~(np.isnan(x_data) | np.isnan(y_data))
    if np.sum(valid_mask) < 2:
        return None, None
    
    x_valid = x_data[valid_mask]
    y_valid = y_data[valid_mask]
    
    # Sort by x values
    sort_idx = np.argsort(x_valid)
    x_sorted = x_valid[sort_idx]
    y_sorted = y_valid[sort_idx]
    
    if method == 'cubic_spline':
        # Create cubic spline
        cs = CubicSpline(x_sorted, y_sorted)
        x_smooth = np.linspace(x_sorted.min(), x_sorted.max(), 200)
        y_smooth = cs(x_smooth)
        
    elif method == 'cubic':
        # Cubic interpolation
        f = interp1d(x_sorted, y_sorted, kind='cubic')
        x_smooth = np.linspace(x_sorted.min(), x_sorted.max(), 200)
        y_smooth = f(x_smooth)
        
    elif method == 'savgol':
        # Smoothing (works on original points)
        if len(y_sorted) > 5:  # Need enough points
            window_length = min(len(y_sorted) // 2 * 2 + 1, 7)  # Odd number
            y_smooth = savgol_filter(y_sorted, window_length, 3)
            x_smooth = x_sorted
        else:
            x_smooth, y_smooth = x_sorted, y_sorted
            
    else:  # linear
        f = interp1d(x_sorted, y_sorted, kind='linear')
        x_smooth = np.linspace(x_sorted.min(), x_sorted.max(), 200)
        y_smooth = f(x_smooth)
    
    return x_smooth, y_smooth

def create_misinformation_rho_plot(p_values, mf_results, mb_results, figure_name):
    
    # Create a plot showing misinformation metric vs rho (p) for both approaches
    
    
    # Create figure
    fig, ax = plt.subplots(figsize=(4, 4))
    
    # Define colors
    blue_color = "#006FC9"
    red_color = "#FF7C01"
    
    # Remove any NaN values
    mf_valid = ~np.isnan(mf_results)
    mb_valid = ~np.isnan(mb_results)
    
    # Plot the data points
    ax.scatter(p_values[mf_valid], mf_results[mf_valid], 
              color=blue_color, marker='+', s=50, alpha=0.8, 
              linewidth=2)
    
    ax.scatter(p_values[mb_valid], mb_results[mb_valid], 
              color=red_color, marker='x', s=50, alpha=0.8, 
              linewidth=2)
    
    # Store handles for legend
    legend_handles = []
    legend_labels = []
    
    # Add scatter plot handles to legend
    scatter_mf = ax.scatter([], [], color=blue_color, marker='+', s=50, 
                           alpha=0.8, linewidth=2)
    scatter_mb = ax.scatter([], [], color=red_color, marker='x', s=50, 
                           alpha=0.8, linewidth=2)
    
    legend_handles.extend([scatter_mf, scatter_mb])
    legend_labels.extend(['mf-mitig', 'mb-mitig'])
    
    # Create smooth interpolation lines
    if np.sum(mf_valid) > 2:
        x_smooth_mf, y_smooth_mf = create_smooth_interpolation(
            p_values[mf_valid], mf_results[mf_valid], method='cubic_spline'
        )
        if x_smooth_mf is not None:
            line_mf = ax.plot(x_smooth_mf, y_smooth_mf, color=blue_color, 
                             linestyle='--', alpha=0.7, linewidth=1.5)[0]
            legend_handles.append(line_mf)
            legend_labels.append('mf-mitig-trend-line')
    
    if np.sum(mb_valid) > 2:
        x_smooth_mb, y_smooth_mb = create_smooth_interpolation(
            p_values[mb_valid], mb_results[mb_valid], method='cubic_spline'
        )
        if x_smooth_mb is not None:
            line_mb = ax.plot(x_smooth_mb, y_smooth_mb, color=red_color, 
                             linestyle='--', alpha=0.7, linewidth=1.5)[0]
            legend_handles.append(line_mb)
            legend_labels.append('mb-mitig-trend-line')
    
    # Customize plot
    ax.set_xlabel(r'$\rho$ parameter', fontsize='large')
    ax.set_ylabel(r'$\mathcal{M}$ - misinformation metric', fontsize='large')
    
    # Set axis limits
    ax.set_xlim(p_values.min() - 0.1, p_values.max() + 0.1)
    
    # Add minor ticks
    ax.minorticks_on()
    ax.tick_params(which='major', length=4, width=0.8)
    ax.tick_params(which='minor', length=2, width=0.6)
    
    # Style spines
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    
    # Create legend
    legend = ax.legend(legend_handles, legend_labels, 
                      loc='upper center',
                      bbox_to_anchor=(0.5, -0.08),
                      ncol=2,
                      frameon=False,
                      fontsize='large',
                      columnspacing=0.5)
    
    # Save and show

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    fig.savefig(figure_name, dpi=1000, bbox_inches='tight')
    
    # Print some statistics
    
    print(f"\nResults Summary:")
    print(f"Model Free - Mean: {np.nanmean(mf_results):.4f}, Std: {np.nanstd(mf_results):.4f}")
    print(f"Model Based - Mean: {np.nanmean(mb_results):.4f}, Std: {np.nanstd(mb_results):.4f}")
    
    # Show the image
    try:
        img = Image.open(figure_name)
        img.show()
    except:
        print("Image saved but couldn't display automatically")
    
    return fig