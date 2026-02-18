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
        print (df_bkg)
        self.ave_BKG = np.mean(df_bkg['mag'])
        return self.time, self.mag

    # Function to plot data
    def plot_data(self):
        plt.plot(self.time, self.mag,'.', color = 'red', label = 'Data Points') 
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
        plt.plot(self.time+self.epoch, self.mag, 'r.', label='Observed data')
        plt.axhline(self.ave_BKG, color='black', linestyle='-',label="BKG")
        plt.plot(Xp+self.epoch, yp, color='blue', alpha=0.25)
        plt.fill_between(Xp+self.epoch, yp-ystd, yp+ystd, alpha=0.1,color='b')
        plt.xlabel('Days')
        plt.ylabel('mag')
        plt.ylim([np.min(self.mag)-1,np.max(self.ave_BKG)+1])
        plt.gca().invert_yaxis()
        plt.plot(self.t_sbo+self.epoch, self.ave_BKG, marker="*", markersize=15, label="t_SBO",     color='y')
        df_bkg = pd.read_csv(self.bkg)
        plt.scatter(df_bkg['MJD'], df_bkg['mag'], color='black', label='Last Null Detection')
        plt.grid()
        plt.legend()
        ax = plt.gca()
        df_bkg = pd.read_csv(self.bkg)
        lnd = float(df_bkg['MJD'].iloc[0] if isinstance(df_bkg['MJD'], pd.Series) else df_bkg['MJD'])
        if self.t_sbo+self.epoch < lnd:
          print('t_sbo calculated to be before lnd')
        ax.annotate(self.t_sbo+self.epoch,
            xy=(self.t_sbo+self.epoch, self.ave_BKG),
            xytext=(self.t_sbo+self.epoch + 5, self.ave_BKG + 0.5),
            arrowprops=dict(facecolor='black', shrink=0.05),
            horizontalalignment='left', verticalalignment='top',
            fontsize=12)
        plot_title = "Shock break out time from %.2f to %.2f MJD" % (self.t_sbo_l+self.epoch, self.t_sbo_r+self.epoch)
        plt.title(plot_title)
        plt.savefig(self.lc.split('.')[0]+'-fit.png')
       