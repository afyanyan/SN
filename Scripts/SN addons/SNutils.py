import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class SN:
    def __init__(self,
                 lc,
                 bkg
                 ):
        self.lc = lc
        self.bkg = bkg

    def read_data(self):
        df_bkg = pd.read_csv(self.bkg)
        df_lc = pd.read_csv(self.lc)
    
        t = df_lc['MJD']
        self.epoch = np.min(t)
        self.time = np.array(t - self.epoch)
        self.mag = np.array(df_lc['mag'])
        self.mag_err = np.array(df_lc['mag_err'])
        self.ave_BKG = np.mean(df_bkg['mag'])
        return self.time, self.mag

    # Function to plot data
    def plot_data(self):
        plt.plot(self.time, self.mag,'.', color = 'red', label = 'Data Points') 
        
        #last null detection for qvh
        plt.plot(60521.48711, 19.9, 'o', 
            color='purple', markersize=8, label='Last non-detection')
        
        plt.xlabel('t [days from the discovery]')
        plt.ylabel('SAP Flux Normalized (e-/s)')
        plt.gca().invert_yaxis()
        plt.legend()
        plt.grid()
        plt.savefig(self.lc.split('.')[0]+'.png')
        return

    # Function to estimate shock break out time
    def shock_breakout(self, Xp, yp, ystd):
        self.t_sbo = Xp[np.argmin(np.abs(yp - self.ave_BKG)[:len(Xp)//2])]
        self.t_sbo_l = Xp[np.argmin(np.abs(yp - ystd - self.ave_BKG)[:len(Xp)//2])]
        self.t_sbo_r = Xp[np.argmin(np.abs(yp + ystd - self.ave_BKG)[:len(Xp)//2])]
        print ("The predicted shock break out time is between",self.t_sbo_l+self.epoch," and ",self.t_sbo_r+self.epoch," in MJD")
        return 


     # Function to plot the fit
    def plot_fit(self, Xp, yp, ystd):
        plt.figure(figsize=(8,5))
        error_scale = 10
        plt.errorbar(self.time+self.epoch, self.mag, yerr=self.mag_err * error_scale, 
                fmt='r.', label='Observed data (errors ×({error_scale}))', capsize=8, alpha=1, elinewidth=3 , ecolor= 'black' , markersize=8)    
        #This adds the error bars to the points

        #last null detection for qvh
        plt.plot(60521.48711, 19.9, 'o', 
            color='purple', markersize=10, label='Last non-detection')
        plt.axhline(self.ave_BKG, color='black', linestyle='-',label="BKG")
        self.plot_telescoping_errors(Xp+self.epoch, yp, ystd,  color='blue',sigma_levels=[1, 2], alphas=[0.3, 0.15],labels=['1σ GPR', '2σ GPR'])
        plt.plot(Xp+self.epoch, yp, color='blue', alpha=0.25)
        plt.xlabel('Days')
        plt.ylabel('mag')
        plt.ylim([np.min(self.mag)-1,np.max(self.ave_BKG)+1])
        plt.gca().invert_yaxis()
        plt.plot(self.t_sbo+self.epoch, self.ave_BKG, marker="*", markersize=15, label="t_SBO",color='y')
        plt.grid()
        plt.legend()
        plot_title = "Shock break out time from %.2f to %.2f MJD" % (self.t_sbo_l+self.epoch, self.t_sbo_r+self.epoch)
        plt.title(plot_title)
        plt.savefig(self.lc.split('.')[0]+'-fit.png')

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
 