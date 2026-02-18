import numpy as np
from SNutils import SN
from SNkernels import KernelUtils as KU
from SNkernels import QuarticKernel,ExpCompositeKernel, ReducedQuarticKernel  # Import custom kernels
from sklearn.gaussian_process.kernels import WhiteKernel

def main():
    # Load supernova data
    supernova = SN(bkg='../../O4_SN/o4_BKG/SN_2023_zcu_background_tns.txt', lc='../../O4_SN/o4_supernovae/SN_2023zcu/sn_2023zcu_atlas_c.dat')
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
