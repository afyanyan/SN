import numpy as np
from SNutils import SN, gaussian_model
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS   
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az

def main():
    # Load supernova data
    supernova = SN(bkg='sn_2024abbv_atlas_o_ul.dat', lc='sn_2024abbv_atlas_o.dat')
    #supernova = SN(bkg='sn_2023_abdg_background_tns.txt', lc='sn_2023abdg_asas_g.dat')

    # Read and plot data
    time, mag, mag_err = supernova.read_data()
    supernova.plot_data()
    plt.close()
    #convert data to tensors
    time_t = torch.tensor(time, dtype = torch.float32)
    mag_t = torch.tensor(mag, dtype = torch.float32)
    mag_err_t = torch.tensor(mag_err, dtype = torch.float32)

    #sbo window
    t_i = supernova.t_last_nondetection - supernova.epoch
    t_f = supernova.t_first_detection - supernova.epoch

    #MCMC model

    # the mask is irrelevant for the MCMC ignore
    mask      = time_t < 10
    time_early = time_t[mask]
    mag_early  = mag_t[mask]
    err_early  = mag_err_t[mask]
    # where the mask code ends

    nuts_kernel = NUTS(gaussian_model)
    mcmc = MCMC(nuts_kernel, num_samples=1000, warmup_steps=200)
    mcmc.run(time_t, mag_t, mag_err_t, t_i, t_f)

    samples = mcmc.get_samples()

    idata = az.from_pyro(mcmc)
    print(az.summary(idata, var_names=["t_sbo", "amplitude", "sigma"]))

# Print the SBO time in MJD
    t_sbo_samples = samples["t_sbo"].numpy()
    mean = t_sbo_samples.mean() + supernova.epoch
    std  = t_sbo_samples.std()
    print(f"\nShock Breakout Time: {np.mean(t_sbo_samples) + supernova.epoch:.5f} MJD")
    print(f"Uncertainty:        ±{np.std(t_sbo_samples):.5f} days")

# Plot posterior distribution
    az.plot_posterior(idata, var_names=["t_sbo"])
    plt.show()

# Plot trace (convergence check)
    az.plot_trace(idata, var_names=["t_sbo", "amplitude", "sigma"])
    plt.show()
    

    



if __name__ == "__main__":
    main()
