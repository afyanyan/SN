def plot_telescoping_errors(self, x, y_mean, y_std, ax=None, color='blue', 
                           sigma_levels=[1, 2], alphas=[0.3, 0.15], 
                           labels=None):

    
        if ax is None:
            ax = plt.gca()
            if isinstance(ax, np.ndarray):
                    ax = ax.flatten()[0]
    
    # Flatten arrays if needed -- Safety check to ensure arrays are 1D before plottin preventing shape mismatch errors -- makes data 1D
        y_mean = y_mean.flatten() if hasattr(y_mean, 'flatten') else y_mean
        y_std = y_std.flatten() if hasattr(y_std, 'flatten') else y_std
    
    # Auto-generate labels if not provided
        if labels is None:  
            labels = [f'{sigma}σ' for sigma in sigma_levels]
    
    # Plot bands from largest to smallest (so smaller bands appear on top)
        fills = []
        for sigma, alpha, label in zip(reversed(sigma_levels), 
                                    reversed(alphas), 
                                    reversed(labels)):
            upper = y_mean + sigma * y_std
            lower = y_mean - sigma * y_std
        
            fill = ax.fill_between(x, lower, upper, 
                                alpha=alpha, color=color, label=label)
            fills.append(fill)
    
        return fills[::-1]
 