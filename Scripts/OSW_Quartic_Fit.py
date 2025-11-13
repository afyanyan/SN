import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import pandas as pd
from astropy.time import Time
import os
import argparse

# Example to use:
# python OSW_Quactic_Fit.py --f-FRG "o4_supernovae/SN_2024qvh/sn_2024qvh_asas_g.dat" --f-BKG "o4_BKG/SN_2024_qvh_background_tns.txt" --f-extra "" --Ncutoff 10 --Nrising 8 --Nsamples 10000 --Peaklum-index 0 --save-path './OSW_Results/'

#==========================================
#--------define variables------------------
#==========================================
# Set up the parser and add arguments


parser = argparse.ArgumentParser(description='Estimate OSW of CCSN with Quartic fit.')
parser.add_argument('--f-FRG',help='path of the main foreground light curve data')
parser.add_argument('--f-BKG',  help="path of the background light curve data")
parser.add_argument('--f-extra', help="path of other foregroud light curve data", nargs='+',type=str, default="")
parser.add_argument('--Ncutoff',help='mainband cutoff after the first data point',type=int)
parser.add_argument('--Nrising',help='number of the rising points',type=int)
parser.add_argument('--Nsamples',help='number of the random samples within the dmag',type=int, default=10000)
parser.add_argument('--Peaklum-index',help='index of the peak luminosity',type=int)
parser.add_argument('--save-path',help='save path for the figures', default= "./OSW_Results/")

command_line_arguments = vars(parser.parse_args())

f_FRG = command_line_arguments['f_FRG']
f_BKG = command_line_arguments['f_BKG']
f_extra = command_line_arguments['f_extra']
N_cutoff = command_line_arguments['Ncutoff']
N_rising = command_line_arguments['Nrising']
N_samples = command_line_arguments['Nsamples']
Peak_lum_index = command_line_arguments['Peaklum_index']
save_path = command_line_arguments['save_path']

#==========================================

#--------------------plot settings------------------------------------------------------

plt.rcParams.update(plt.rcParamsDefault)
size=22
params = {'legend.fontsize': 'large',
          'axes.labelsize': size,
          'axes.titlesize': size,
          'xtick.labelsize': size*0.9,
          'ytick.labelsize': size*0.9,
          'legend.fontsize': 10,
          'axes.titlepad': 7,
          'font.family': 'serif',
          'font.weight': 'medium',
          'xtick.major.size': 10,
          'ytick.major.size': 10,
          'xtick.minor.size': 5,
          'ytick.minor.size': 5,
          'text.usetex':True,
          #'text.latex.preamble': r'\usepackage{color}'
          }
plt.rcParams.update(params)

#==========================================
#----------main----------------
#==========================================

def main():
    
    # get SN name and check the output path
    sn_name = f_FRG.split("/")[-2]
    print("Process SN : ", sn_name)
    
    pwd = save_path +"./"+ sn_name
    check_mkdir_floder(pwd)

    
#--------------------Load data------------------------------------------------------
    
    # Load Background data
    df_BKG = pd.read_csv(f_BKG, sep=",", names=['time', 'mag'])
    ave_BKG = np.average(df_BKG['mag'])

    print ("The average backgroud magnitude is: ", ave_BKG)


    # Load Foreground data
    df_FRG = pd.read_csv(f_FRG, sep=",", names=['time','mag','dmag'],skiprows=1)

    # ignore the data 'N cutoff from the mainband', it is days after the first data
    df_FRG = df_FRG[df_FRG['time']<N_cutoff + df_FRG['time'][0]]

    t_FRG = df_FRG['time'].values
    mag_FRG = df_FRG['mag'].values
    dmag_FRG =  df_FRG['dmag'].values
    t_disc= t_FRG[0]
    print ("Discovery time is ",t_disc)


    # load more LC data from other bands
    try:
        for f in f_extra:
            df_ex = pd.read_csv(f ,sep=",", names=['time', 'mag','dmag'], skiprows=1)
                # Make sure we don't load in any data after the peak point

            time_i = df_ex['time'].values
            mag_i = df_ex['mag'][time_i < t_FRG[Peak_lum_index]].values
            dmag_i = df_ex['dmag'][time_i < t_FRG[Peak_lum_index]].values
            time_i = time_i[time_i < t_FRG[Peak_lum_index]]

            # Add this data to our data in the primary band
            mag_FRG = list(mag_FRG) + list(mag_i)
            dmag_FRG = list(dmag_FRG) + list(dmag_i)
            t_FRG = list(t_FRG) + list(time_i)

        # order all the combined data based on time.
        mag_FRG = np.array([mag_FRG[i] for i in np.argsort(t_FRG)])
        dmag_FRG = np.array([dmag_FRG[i] for i in np.argsort(t_FRG)])
        t_FRG = np.array([t_FRG[i] for i in np.argsort(t_FRG)])
        
    except FileNotFoundError:
        print ("No extra files are used.")
    
#--------Single Quartic fit------------------------------------------------------------------

    t_norm = t_FRG - t_disc # set discovery day as 0
    X1,Y1,coeff = Quartic_fit(t_norm, mag_FRG)

    t_sbo_norm = X1[np.argmin(np.abs(Y1 - ave_BKG ))]
    mag_sbo = Y1[np.argmin(np.abs(Y1 - ave_BKG ))]

    t_sbo = t_sbo_norm + t_disc
    t_sbo_utc,t_sbo_gps = MJD2GPS(t_sbo)
    print ('-----------------------------')
    print ("From a singel quartic fit, the shock break out is {:.2f} days from the discovery date {:.2f}".format(t_sbo_norm,t_disc))
    print ("t_sbo = {:.2f}, {} , {} ".format(t_sbo,t_sbo_gps,t_sbo_utc))
    print ("Magnitude = ",mag_sbo)
    print ('-----------------------------')
    t_sbo.round(2)


    plt.close()
    fig, ax1 = plt.subplots()

    ax1.scatter (t_FRG,mag_FRG,color = "red",label="Observed light curve")
    ax1.plot(X1+t_disc,Y1,color='blue',label='Quartic fit')
    ax1.axhline(ave_BKG, color='black', linestyle='-',label="Average BKG") 
    ax1.axvline(t_disc, color='gray', linestyle='--',label="Discorvery date")
    ax1.plot (t_sbo,ave_BKG, marker='*', color="yellow",markersize=15, label="t_SBO="+str(t_sbo.round(2)))

    ax1.set_xlim(t_sbo-10,t_sbo+30) # range for magnitude
    ax1.set_xlabel("MJD")

    ax1.invert_yaxis() 
    ax1.set_ylabel("Magnitude")
    ax1.set_ylim(ave_BKG+3,mag_FRG[-1]-3)

    ax1.grid()
    ax1.legend(loc=4)

    ax2 = ax1.twiny()
    ax2.set_xlim(t_sbo_norm-10,t_sbo_norm+30)

    plt.title(sn_name + ": Single quartic fit")
    plt.savefig (pwd + "/"+ sn_name + "_Single_qf.png",dpi = 150, bbox_inches = "tight")
    plt.show()

#-------- Quartic fit with uncertainties------------------------------------------------------------------

    # Compute the slope
    # for this analysis, we are using the slope of the line joining shock breakout point to peak luminosity point
    # --ref: https://wiki.ligo.org/Main/Review_Of_OSW
    

    var = 0.65  # varience for the slope

    slope = ( mag_FRG[N_rising]-mag_sbo ) / ( t_FRG[N_rising]- t_sbo) 
    slope_l = slope*(1+var)
    slope_h = slope*(1-var)
    
    if slope_l>slope_h:
        slope_l, slope_h = slope_h, slope_l
        
    print ("Slope = ", slope)
    print ("Slope range = [", slope_l, ':',slope_h,"]")

    # adding the varience from light curve into magnitude
    Mag_r_l = mag_FRG - dmag_FRG
    Mag_r_h = mag_FRG + dmag_FRG


    # draw randoms from interval [Mag_l,Mag_h] for N_sample time
    Mags = []
    for j in range(N_samples):
        mag = np.random.uniform(Mag_r_l, Mag_r_h)
        Mags.append(mag)

    all_fits = {}
    T_sbo_norm_all =[]
    T_sbo_norm_all_std = []
    N_bad = 0
    for i, Mag_chunk in enumerate (Mags):
       # print (i)
        # Fit for each chunk                           
        X,Y,coeff = Quartic_fit(t_norm,Mag_chunk)
       # print (coeff[3])
        # Check the slope with the first order coeff (d), to make sure the coeff in within the range of slope
        if slope_l <= coeff[3] <= slope_h:
            ##print ("Fit {} is in the slop range".format(i))
            all_fits[i] = X,Y,coeff                      
            t_sbo_chunks = X[np.argmin(np.abs(Y - ave_BKG))]
            T_sbo_norm_all.append(t_sbo_chunks)
            T_sbo_sigma = np.std(T_sbo_norm_all)
            T_sbo_norm_all_std.append(T_sbo_sigma)
        else:
            N_bad +=1
    
    print ("Number of the fits beyond slope range: ", N_bad, ",percentage= ", N_bad/N_samples)

    # plot the light curve data with all fits     
    plt.close()
    fig, ax = plt.subplots()
    ax.invert_yaxis()
    ax.set_ylim(ave_BKG+5,mag_FRG[-1]-5)
    ax.set_xlim(t_sbo_norm-10,t_sbo_norm+30)

    ax.set_xlabel("t_sbo[days from discovery]")
    ax.set_ylabel("Magnitude")

    ax.set_title("All fits")
    for key in all_fits:
        ax.plot(all_fits[key][0],all_fits[key][1])
    ax.scatter (t_norm,mag_FRG,color = "red",s= 100,label="Observed light curve")
    ax.axhline(ave_BKG, color='black', linestyle='-',label="Average BKG")
    ax.grid()
    ax.legend()
    plt.title(sn_name + ": all quartic fits")
    plt.savefig (pwd + "/"+ sn_name + "_all_QF.png",dpi = 150, bbox_inches = "tight")


    # plot the all fits with confidence belt 
    all_fits_Y = [np.array(v[1]) for v in all_fits.values()]
    all_fits_Y_ave = np.mean(all_fits_Y, axis=0)
    all_fits_Y_std = np.std(all_fits_Y, axis=0)

    sigma_1p = all_fits_Y_ave + all_fits_Y_std
    sigma_1m = all_fits_Y_ave - all_fits_Y_std
    sigma_2p = all_fits_Y_ave + all_fits_Y_std*2
    sigma_2m = all_fits_Y_ave - all_fits_Y_std*2

    ave_T_sbo_norm_all = np.average(T_sbo_norm_all)
    std_T_sbo_norm_all = np.std(T_sbo_norm_all)

    ave_T_sbo_all = ave_T_sbo_norm_all+t_disc
    print ("Ave t_sbo:", ave_T_sbo_all)
    print ("Ave t_sbo [days from disc]:", ave_T_sbo_norm_all)

    fig, ax1 = plt.subplots()
    ax1.invert_yaxis()
    ax1.errorbar(t_FRG, mag_FRG, yerr=dmag_FRG, fmt='.', color='black', label='Dmag from LC')
    for i, Mag_chunk in enumerate (Mags):
        ax1.plot(t_FRG,Mag_chunk,'.',color='red')
    ax1.plot(X1+t_disc,all_fits_Y_ave, color = 'blue', label = "Average of all fits")
    ax1.fill_between(X1+t_disc, sigma_1m, sigma_1p, alpha=0.4, color='blue', label = '1 sigma')
    ax1.fill_between(X1+t_disc, sigma_2m, sigma_2p, alpha=0.2, color='blue', label = '2 sigma')
    ax1.axhline(ave_BKG, color='black', linestyle='-',label="Average BKG") 
    ax1.axvline(t_disc, color='gray', linestyle='--',label="Discorvery date")
    ax1.axvline(ave_T_sbo_all, color='cyan', linestyle='-',label="Ave_Tsbo")
    ax1.set_xlim(t_sbo-5,t_sbo+20) # range for magnitude
    ax1.set_xlabel("MJD")

    ax1.set_ylabel("Magnitude")
    ax1.set_ylim(ave_BKG+3,mag_FRG[-1]-3)

    ax1.grid()
    ax1.legend(loc=4)

    ax2 = ax1.twiny()
    ax2.set_xlim(t_sbo_norm-5,t_sbo_norm+20)# range for magnitude


    plt.title(sn_name + ": all quartic fits_belt")
    plt.savefig (pwd + "/"+ sn_name + "_all_QF_belt.png",dpi = 150, bbox_inches = "tight")
    plt.show()


    # plot hist of t_sbo 


    fig, ax = plt.subplots()
    hist_y,hist_x,_ = ax.hist(T_sbo_norm_all, bins='auto', edgecolor='blue')
    ax.vlines(ave_T_sbo_norm_all, 0, hist_y.max(), linewidth=5, color="green", label=r"$\mu=$"+str(np.round(ave_T_sbo_norm_all, 2)))
    ax.vlines([ave_T_sbo_norm_all - std_T_sbo_norm_all, ave_T_sbo_norm_all + std_T_sbo_norm_all], 0, (2/5) * hist_y.max(), color="red", label=r"$\sigma=$"+str(np.round(std_T_sbo_norm_all, 2)))
    ax.axvline(ave_T_sbo_norm_all, color='cyan', label=r"$t_{sbo}=$"+str(ave_T_sbo_all.round(2)))
    ax.set_xlabel("t_sbo [days from discovery]")
    ax.set_ylabel("counts")
    ax.set_title("t_sbo from all fits")
    ax.set_xlim(ave_T_sbo_norm_all - 2, ave_T_sbo_norm_all + 2)
    ax.grid(True, which='both')
    ax.minorticks_on()
    ax.legend()
    ax.set_title(sn_name + ":t_sbo from all fits")
    plt.savefig (pwd + "/"+ sn_name + "_tsbo_hist.png",dpi = 150, bbox_inches = "tight")
    plt.show()

    # plot hist of std of t_sbo during N)samples
    plt.figure()
    y, x, _ = plt.hist(T_sbo_norm_all_std)
    ave_std = np.average(T_sbo_norm_all_std) 
    ave_std_sig = np.std(T_sbo_norm_all_std)
    plt.vlines(ave_std,0,y.max(), linewidth = 3, color = "Green", label = r"$\mu=$"+ str(ave_std.round(2)) + " days")
    plt.vlines(ave_std+ave_std_sig,0,(2/5) * y.max(), color = "Red", label = r"$\sigma=$" + str(ave_std_sig.round(2)) + " days")
    plt.vlines(ave_std-ave_std_sig,0,(2/5) * y.max(), color = "Red")
    plt.vlines(1,1,1, color = "White") #label = str(len(shock_breakout_estimate)) + " fits, out of " + str(random_sample_number) + " tests")
    plt.xlabel("Uncertainty [days from discovery ]")
    plt.ylabel("counts")
    plt.grid(which = "both")
    plt.minorticks_on()
    plt.title(sn_name +': std of t_sbo from all fits')
    plt.xlim(ave_std-1,ave_std+1)
    plt.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.legend(fontsize = "medium")
    plt.savefig(pwd + "/"+sn_name + '_std_hist.png', dpi = 150, bbox_inches = "tight")
    plt.show()
    plt.close()  
    
    return

#==========================================================================

def Quartic_function(x, a, b, c, d, e):
    return (a*x**4 + b*x**3 + c*x**2 + d*x + e)
        
#==========================================================================
 
def Quartic_fit(x,y):

    # Prepare for the Quatic fit with FRG r band with a limited points selected
    # x: time, while the discorvery time is 0
    # y: mag

    # Return the quartic fit line

    [a, b, c, d, e], pcov = scipy.optimize.curve_fit(Quartic_function, x, y)

    X1 = np.arange(-80, x[-1], 0.1) #(-80 to the end of x)
    Y1 = a*X1**4 + b*X1**3 + c*X1**2 + d*X1 + e

    return (X1,Y1,[a, b, c, d, e])
    
#==========================================================================    

def MJD2GPS(mjd):

    utc_time = Time(mjd, format='mjd').iso
    gps = Time(mjd, format='mjd').gps

    return (utc_time,gps)
    

#==========================================================================

def check_mkdir_floder(f):
    check_folder = os.path.isdir(f)
    # If folder doesn't exist, then create it.
    if not check_folder:
        os.makedirs(f)
    else:
        pass
    return
#==========================================================================


if __name__ == "__main__":
    main()

