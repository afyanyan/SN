import numpy as np
from SNutils import SN
from SNkernels import KernelUtils as KU
from SNkernels import QuarticKernel,ExpCompositeKernel, ReducedQuarticKernel  # Import custom kernels
from sklearn.gaussian_process.kernels import WhiteKernel
import os
import pandas as pd

def main():

      # Define the directory containing light curve files
    #lc_directory = '../../O4_SN/o4_supernovae/SN_2025vzq/'
    lc_directory = r'C:\Users\conno\New folder\SN\O4_SN\o4_supernovae\SN_2024jlf'
#C:\Users\conno\New folder\SN\O4_SN\o4_supernovae\SN_2023abdg
    # Get a list of all light curve files in the directory
    # Assuming all files in this directory are light curve files with the specified format
    lcs = [os.path.join(lc_directory, f) for f in os.listdir(lc_directory) if f.endswith('.dat')] # Assuming .dat files
    mag_FRG = []
    dmag_FRG = []
    t_FRG = []

    # Loop through each light curve file to combine the data
    for f in lcs:
      df_ex = pd.read_csv(f ,sep=",", names=['time', 'mag','dmag'], skiprows=1)
      time_i = df_ex['time'].values
      mag_i = df_ex['mag'].values
      dmag_i = df_ex['dmag'].values

      mag_FRG.extend(list(mag_i))
      dmag_FRG.extend(list(dmag_i))
      t_FRG.extend(list(time_i))
      #print('1')
    combined_lc_df = pd.DataFrame({'MJD': t_FRG, 'mag': mag_FRG, 'dmag': dmag_FRG})
    combined_lc_file = 'combined_lightcurve.csv'
    combined_lc_df.to_csv(combined_lc_file, index=False)
    supernova = SN(bkg='C:/Users/conno/New folder/SN/O4_SN/o4_BKG/SN_2024_jlf_background_tns.txt', lc = combined_lc_file)
    # Load supernova data
    #supernova = SN(bkg='../../O4_SN/o4_BKG/SN_2023_abdg_background_tns.txt', lc='../../O4_SN/o4_supernovae/SN_2023abdg/sn_2023abdg_atlas_c.dat')
    #supernova = SN(bkg='sn_2023_abdg_background_tns.txt', lc='sn_2023abdg_asas_g.dat')
    # Read and plot data
    time, mag = supernova.read_data()
    supernova.plot_data()

    # Define supernova kernel
    kernel = (
            #ReducedQuarticKernel() * 
            ExpCompositeKernel()
        
            + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e1))
        )

    # Fit Gaussian Process Regression model
    Xp, yp, ystd = KU.fit(time, mag, kernel)

    # Estimate shock break out time
    supernova.shock_breakout(Xp, yp, ystd)
 
    # Plot the results
    supernova.plot_fit(Xp, yp, ystd)

if __name__ == "__main__":
    main()