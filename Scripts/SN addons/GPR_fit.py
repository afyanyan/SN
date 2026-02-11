import numpy as np
from SNutils import SN
from SNkernels import KernelUtils as KU
from SNkernels import QuarticKernel,ExpCompositeKernel, ReducedQuarticKernel  # Import custom kernels
from sklearn.gaussian_process.kernels import WhiteKernel

def main():
    # Load supernova data
    supernova = SN(bkg='sn_2024qvh_goto_L_ul.dat', lc='sn_2024qvh_atlas_o.dat')
    #supernova = SN(bkg='sn_2023_abdg_background_tns.txt', lc='sn_2023abdg_asas_g.dat')
    supernova2 = SN(bkg='sn_2024qvh_goto_L_ul.dat', lc='sn_2024qvh_atlas_c.dat')
    supernova3 = SN(bkg='sn_2024qvh_goto_L_ul.dat', lc='sn_2024qvh_asas_g.dat')

    # Read and plot data
    time, mag = supernova.read_data()
    supernova.plot_data()

    time2, mag2 = supernova2.read_data()
    supernova2.plot_data()
    
    time3, mag3 = supernova3.read_data()
    supernova3.plot_data()

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
    supernova2.shock_breakout(Xp, yp, ystd)
    supernova3.shock_breakout(Xp, yp, ystd)

 
    # Plot the results
    supernova.plot_fit(Xp, yp, ystd)
    supernova.plot_telescoping_errors(Xp, yp, ystd) 

    supernova2.plot_fit(Xp, yp, ystd)
    supernova2.plot_telescoping_errors(Xp, yp, ystd)

    supernova3.plot_fit(Xp, yp, ystd)
    supernova3.plot_telescoping_errors(Xp, yp, ystd)
    


if __name__ == "__main__":
    main()
