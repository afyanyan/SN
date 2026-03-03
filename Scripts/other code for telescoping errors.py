#Goes in main function
supernova.plot_telescoping_errors(Xp, yp, ystd) 



#Goes under the plot function
error_scale = 10
plt.errorbar(self.time+self.epoch, self.mag, yerr=self.mag_err * error_scale, 
                fmt='r.', label='Observed data (errors ×({error_scale}))', capsize=8, alpha=1, elinewidth=3 , ecolor= 'black' , markersize=8)    
        #This adds the error bars to the points

self.plot_telescoping_errors(Xp+self.epoch, yp, ystd,  color='blue',sigma_levels=[1, 2], alphas=[0.3, 0.15],labels=['1σ GPR', '2σ GPR'])
