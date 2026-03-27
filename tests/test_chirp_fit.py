import numpy as np
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
import warnings

def analyze_z_pz_correlation(dist, n_bins=100, min_particles=10, 
                            stat_method='weighted_mean', smoothing=0.1):
    """
    Uniform binning + weighted spline interpolation for z-pz correlation analysis.
    
    Parameters:
    ----------
    dist : ParticleDistribution instance
        Particle distribution data
    n_bins : int
        Number of uniform bins in z direction
    min_particles : int
        Minimum particles per bin (bins with fewer are discarded)
    stat_method : str
        'weighted_mean' or 'median' for bin statistics
    smoothing : float
        Spline smoothing parameter (0 = interpolation, >0 = smoothing)
    
    Returns:
    -------
    spline_func : callable
        Fitted spline function f(z) = pz
    bin_stats : dict
        Statistics for each bin (z_center, pz_stat, weight, n_particles)
    """
    
    # Extract particle data
    z_data = dist.z.reshape(-1)
    pz_data = dist.pz.reshape(-1)
    
    # Use particle weights if available, otherwise uniform weights
    if hasattr(dist, 'w') and dist.w is not None:
        weights = dist.w.reshape(-1)
    else:
        weights = np.ones_like(z_data)
    
    # Create uniform bins in z
    z_min, z_max = z_data.min(), z_data.max()
    bin_edges = np.linspace(z_min, z_max, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    
    # Initialize arrays for bin statistics
    pz_stats = np.full(n_bins, np.nan)
    bin_weights = np.zeros(n_bins)  # For spline fitting (number of particles)
    n_particles = np.zeros(n_bins, dtype=int)
    
    # Calculate statistics for each bin
    for i in range(n_bins):
        mask = (z_data >= bin_edges[i]) & (z_data < bin_edges[i + 1])
        if i == n_bins - 1:  # Include right edge for last bin
            mask = (z_data >= bin_edges[i]) & (z_data <= bin_edges[i + 1])
        
        if np.sum(mask) >= min_particles:
            n_particles[i] = np.sum(mask)
            bin_weights[i] = n_particles[i]  # Use particle count as weight
            
            if stat_method == 'weighted_mean':
                # Weighted average using particle weights
                pz_stats[i] = np.average(pz_data[mask], weights=weights[mask])
            elif stat_method == 'median':
                # Simple median (unweighted)
                pz_stats[i] = np.median(pz_data[mask])
    
    # Filter out bins with insufficient particles
    valid_mask = ~np.isnan(pz_stats)
    z_valid = bin_centers[valid_mask]
    pz_valid = pz_stats[valid_mask]
    weights_valid = bin_weights[valid_mask]
    n_valid = n_particles[valid_mask]
    
    if len(z_valid) < 3:
        raise ValueError(f"Only {len(z_valid)} valid bins after filtering. Try reducing min_particles.")
    
    # Fit weighted spline
    spline_func = UnivariateSpline(z_valid, pz_valid, 
                                  w=np.sqrt(weights_valid),  # sqrt for statistical weighting
                                  s=len(z_valid) * smoothing)
    
    # Prepare bin statistics for return
    bin_stats = {
        'z_centers': bin_centers,
        'pz_stats': pz_stats,
        'bin_weights': bin_weights,
        'n_particles': n_particles,
        'valid_mask': valid_mask
    }
    
    return spline_func, bin_stats

def plot_analysis(dist, spline_func, bin_stats):
    """
    Plot original distribution, binned statistics, and fitted spline.
    
    Parameters:
    ----------
    dist : ParticleDistribution instance
        Particle distribution data
    spline_func : callable
        Fitted spline function from analyze_z_pz_correlation
    bin_stats : dict
        Bin statistics from analyze_z_pz_correlation
    """
    
    from partdist.pd3d.viz import hist2d_pd3d
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Original 2D histogram (z-pz)
    ax1 = axes[0, 0]
    hist2d_pd3d(dist, x='z', y='pz', ax=ax1, color_threshold=1e-2, cmap='jet')
    ax1.set_title('Original Distribution (z-pz)')
    # ax1.set_xlabel('z [m]')
    # ax1.set_ylabel('pz [eV/c]')
    
    # Plot 2: Binned statistics overlay
    ax2 = axes[0, 1]
    hist2d_pd3d(dist, x='z', y='pz', ax=ax2, color_threshold=1e-2, alpha=0.5, cmap='jet')
    
    # Plot valid bins with size proportional to particle count
    valid_mask = bin_stats['valid_mask']
    z_valid = bin_stats['z_centers'][valid_mask]
    pz_valid = bin_stats['pz_stats'][valid_mask]
    n_valid = bin_stats['n_particles'][valid_mask]
    
    scatter = ax2.scatter(z_valid, pz_valid*1e-6, 
                         s=np.sqrt(n_valid) * 2,  # Size proportional to sqrt(particle count)
                         c='red', edgecolors='black', 
                         alpha=0.8, label='Binned Statistics')
    
    # Plot spline
    z_smooth = np.linspace(z_valid.min(), z_valid.max(), 1000)
    pz_smooth = spline_func(z_smooth)
    ax2.plot(z_smooth, pz_smooth*1e-6, 'g-', linewidth=2, label='Fitted Spline')
    
    ax2.set_title('Binned Statistics + Spline Fit')
    # ax2.set_xlabel('z [m]')
    # ax2.set_ylabel('pz [eV/c]')
    ax2.legend()
    
    # Plot 3: Spline function alone
    ax3 = axes[1, 0]
    ax3.plot(z_smooth, pz_smooth, 'b-', linewidth=2, label='Spline: f(z) = pz')
    ax3.scatter(z_valid, pz_valid, s=20, c='red', alpha=0.6, 
               label='Binned Data Points')
    ax3.set_title('Spline Function')
    ax3.set_xlabel('z [m]')
    ax3.set_ylabel('pz [eV/c]')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Particle count per bin
    ax4 = axes[1, 1]
    bins = np.arange(len(bin_stats['n_particles']))
    colors = ['red' if m else 'gray' for m in bin_stats['valid_mask']]
    ax4.bar(bins, bin_stats['n_particles'], color=colors, alpha=0.7)
    ax4.axhline(y=bin_stats['n_particles'][bin_stats['valid_mask']].min(), 
                color='green', linestyle='--', 
                label=f'Min valid: {bin_stats["n_particles"][bin_stats["valid_mask"]].min()}')
    ax4.axhline(y=bin_stats['n_particles'][bin_stats['valid_mask']].mean(), 
                color='blue', linestyle='--', 
                label=f'Mean valid: {bin_stats["n_particles"][bin_stats["valid_mask"]].mean():.1f}')
    ax4.set_xlabel('Bin Index')
    ax4.set_ylabel('Number of Particles')
    ax4.set_title('Particle Count per Bin')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Add statistics text
    total_particles = len(dist.z.reshape(-1))
    valid_bins = np.sum(valid_mask)
    stats_text = f'Total Particles: {total_particles:,}\n'
    stats_text += f'Valid Bins: {valid_bins}/{len(valid_mask)}\n'
    stats_text += f'Min Particles per Valid Bin: {n_valid.min()}\n'
    stats_text += f'Max Particles per Valid Bin: {n_valid.max()}'
    
    fig.text(0.02, 0.02, stats_text, fontsize=10, 
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray"))
    
    plt.tight_layout()
    return fig

# Main test code
if __name__ == "__main__":
    # Import the actual data
    from test_io3d import dist_from_astra as dist
    
    # Run analysis with default parameters
    spline_func, bin_stats = analyze_z_pz_correlation(
        dist, 
        n_bins=100,
        min_particles=100,  # Adjust based on your data
        stat_method='weighted_mean',
        smoothing=0.1
    )
    
    # Plot results
    fig = plot_analysis(dist, spline_func, bin_stats)
    plt.show()
    
    # Print summary
    print("Analysis Summary:")
    print(f"Total particles: {len(dist.z.reshape(-1)):,}")
    print(f"Valid bins: {np.sum(bin_stats['valid_mask'])}/{len(bin_stats['valid_mask'])}")
    print(f"Min particles in valid bin: {bin_stats['n_particles'][bin_stats['valid_mask']].min()}")
    print(f"Max particles in valid bin: {bin_stats['n_particles'][bin_stats['valid_mask']].max()}")
    # print(f"Spline smoothing parameter: {spline_func.get_smoothing_factor():.3e}")
    
    # Example: Evaluate spline at specific z values
    z_test = np.array([dist.z.min(), dist.z.max()])
    pz_test = spline_func(z_test)
    print(f"\nSpline evaluation at z bounds:")
    print(f"  z={z_test[0]:.3e} m -> pz={pz_test[0]:.3e} eV/c")
    print(f"  z={z_test[1]:.3e} m -> pz={pz_test[1]:.3e} eV/c")