import numpy as np
from SNutils import SN
from SNkernels import KernelUtils as KU
from SNkernels import ExpCompositeKernel
from sklearn.gaussian_process.kernels import WhiteKernel


def main():
    supernova = SN(
        bkg="data/sn_2024pxg_clear_ul.dat",
        lc="data/sn_2024pxg_clear.dat"
    )

    time, mag = supernova.read_data()
    supernova.last_non_detection_and_first_detection()

    kernel = ExpCompositeKernel() + WhiteKernel(
        noise_level=1e-3,
        noise_level_bounds=(1e-6, 1e1)
    )

    Xp, yp, ystd = KU.fit(time, mag, kernel)

    supernova.shock_breakout(Xp, yp, ystd)

    outpath = supernova.plot_fit(Xp, yp, ystd, outdir="outputs")
    print(f"Saved plot: {outpath}")


if __name__ == "__main__":
    main()
