# Python script to plot 

import numpy as np
import matplotlib.pyplot as plt

plt.close('all')

size=22

params = {'axes.labelsize': size,
          'axes.titlesize': size,
          'xtick.labelsize': size*0.9,
          'ytick.labelsize': size*0.9,
          'legend.fontsize': size*0.9,
          'axes.titlepad': 7,
          'font.family': 'serif',
          'font.weight': 'medium',
          'xtick.major.size': 10,
          'ytick.major.size': 10,
          'xtick.minor.size': 5,
          'ytick.minor.size': 5,
          'text.usetex':True,
          }

plt.rcParams.update(params)

# Early photometry (Rosa Poggiani, March 2024)

# Object name
obj_Id='SN 2023ixf'

# t2, first detection
t2 = 60082.82611

# File name
fild = obj_Id.replace(" ","_").lower()

# AAVSO data
fil_aavso_CV = fild+'_aavso_CV.dat'
fil_aavso_B = fild+'_aavso_B.dat'
fil_aavso_V = fild+'_aavso_V.dat'
fil_aavso_R = fild+'_aavso_R.dat'
fil_aavso_I = fild+'_aavso_I.dat'

# Circulars/Telegrams/published data
fil_B = fild+'_B.dat'
fil_V = fild+'_V.dat'
fil_R = fild+'_R.dat'
fil_g = fild+'_g.dat'
fil_r = fild+'_r.dat'
fil_clear = fild+'_clear.dat'
fil_itagaki = fild+'_itagaki_clear.dat'
fil_citizen_V = fild+'_citizen_V.dat'
fil_CV = fild+'_CV.dat'
fil_ztf_g = fild+'_ztf_g.dat'
# Upper limit data 
fil_g_upper = fild+'_early_g_upper.dat'
fil_o_upper = fild+'_early_o_upper.dat'
fil_clear_upper = fild+'_early_clear_upper.dat'
fil_CV_upper = fild+'_early_CV_upper.dat'
# AAVSO photometric data
mjd_aavso_CV,mag_aavso_CV,magerr_aavso_CV=np.loadtxt(fil_aavso_CV,unpack=True,usecols=(0,1,2),delimiter=',',skiprows=1)
mjd_aavso_B,mag_aavso_B,magerr_aavso_B=np.loadtxt(fil_aavso_B,unpack=True,usecols=(0,1,2),delimiter=',',skiprows=1)
mjd_aavso_V,mag_aavso_V,magerr_aavso_V=np.loadtxt(fil_aavso_V,unpack=True,usecols=(0,1,2),delimiter=',',skiprows=1)
mjd_aavso_R,mag_aavso_R,magerr_aavso_R=np.loadtxt(fil_aavso_R,unpack=True,usecols=(0,1,2),delimiter=',',skiprows=1)
mjd_aavso_I,mag_aavso_I,magerr_aavso_I=np.loadtxt(fil_aavso_I,unpack=True,usecols=(0,1,2),delimiter=',',skiprows=1)
# Circulars/telegrams/published data
mjd_B,mag_B,magerr_B=np.loadtxt(fil_B,unpack=True,usecols=(0,1,2),delimiter=',',skiprows=1)
mjd_V,mag_V,magerr_V=np.loadtxt(fil_V,unpack=True,usecols=(0,1,2),delimiter=',',skiprows=1)
mjd_R,mag_R,magerr_R=np.loadtxt(fil_R,unpack=True,usecols=(0,1,2),delimiter=',',skiprows=1)
mjd_g,mag_g,magerr_g=np.loadtxt(fil_g,unpack=True,usecols=(0,1,2),delimiter=',',skiprows=1)
mjd_r,mag_r,magerr_r=np.loadtxt(fil_r,unpack=True,usecols=(0,1,2),delimiter=',',skiprows=1)
mjd_clear,mag_clear,magerr_clear=np.loadtxt(fil_clear,unpack=True,usecols=(0,1,2),delimiter=',',skiprows=1)
mjd_clear_itagaki,mag_clear_itagaki,magerr_clear_itagaki=np.loadtxt(fil_itagaki,unpack=True,usecols=(0,1,2),delimiter=',',skiprows=1)
mjd_citizen_V,mag_citizen_V,magerr_citizen_V=np.loadtxt(fil_citizen_V,unpack=True,usecols=(0,1,2),delimiter=',',skiprows=1)
mjd_CV,mag_CV,magerr_CV=np.loadtxt(fil_CV,unpack=True,usecols=(0,1,2),delimiter=',',skiprows=1)
# Non detectioss (N.D.)
mjd_g_upper,mag_g_upper=np.loadtxt(fil_g_upper,unpack=True,usecols=(0,1),delimiter=',',skiprows=1)
mjd_o_upper,mag_o_upper=np.loadtxt(fil_o_upper,unpack=True,usecols=(0,1),delimiter=',',skiprows=1)
mjd_clear_upper,mag_clear_upper=np.loadtxt(fil_clear_upper,unpack=True,usecols=(0,1),delimiter=',',skiprows=1)
mjd_CV_upper,mag_CV_upper=np.loadtxt(fil_CV_upper,unpack=True,usecols=(0,1),delimiter=',',skiprows=1)


# Time in days after the explosion
t_days_B = mjd_B - t2
t_days_V = mjd_V - t2
t_days_R = mjd_R - t2
t_days_CV = mjd_CV - t2
t_days_aavso_CV = mjd_aavso_CV - t2
t_days_aavso_B = mjd_aavso_B - t2
t_days_aavso_V = mjd_aavso_V - t2
t_days_aavso_R = mjd_aavso_R - t2
t_days_aavso_I = mjd_aavso_I - t2
t_days_g = mjd_g - t2
t_days_r = mjd_r - t2
t_days_clear = mjd_clear - t2
t_days_clear_itagaki = mjd_clear_itagaki - t2
t_days_citizen_V = mjd_citizen_V - t2
t_days_g_upper = mjd_g_upper - t2
t_days_o_upper = mjd_o_upper - t2
t_days_clear_upper = mjd_clear_upper - t2
t_days_CV_upper = mjd_CV_upper - t2


# Early photometry
plt.figure(figsize=(20,6))
plt.errorbar(t_days_B,mag_B,magerr_B,fmt='bo',markersize=6,ls='',label='B')
plt.errorbar(t_days_V,mag_V,magerr_V,fmt='go',markersize=6,ls='',label='V')
plt.errorbar(t_days_R,mag_R,magerr_R,fmt='ro',markersize=6,ls='',label='R')
plt.errorbar(t_days_CV,mag_CV,magerr_CV,marker='o',markerfacecolor='darkgrey',ecolor='darkgrey',markersize=6,ls='',label='CV')
plt.errorbar(t_days_aavso_CV,mag_aavso_CV,magerr_aavso_CV,marker='o',markerfacecolor='darkgrey',ecolor='darkgrey',markersize=6,ls='')
plt.errorbar(t_days_aavso_B,mag_aavso_B,magerr_aavso_B,fmt='bo',markersize=6,ls='')
plt.errorbar(t_days_aavso_V,mag_aavso_V,magerr_aavso_V,fmt='go',markersize=6,ls='')
plt.errorbar(t_days_aavso_R,mag_aavso_R,magerr_aavso_R,fmt='ro',markersize=6,ls='')
plt.errorbar(t_days_aavso_I,mag_aavso_I,magerr_aavso_I,fmt='yo',markersize=6,ls='',label='I')
plt.errorbar(t_days_g,mag_g,magerr_g,fmt='g+',markersize=6,ls='',label='g')
plt.errorbar(t_days_r,mag_r,magerr_r,fmt='r+',markersize=6,ls='',label='r')
plt.errorbar(t_days_clear,mag_clear,magerr_clear,fmt='ms',markersize=6,ls='',label='clear')
plt.plot(t_days_clear_itagaki,mag_clear_itagaki, 'ms',markersize=6)
plt.errorbar(t_days_citizen_V,mag_citizen_V,magerr_citizen_V,fmt='go',markersize=6,ls='')
plt.plot(t_days_g_upper,mag_g_upper,'gv',markersize=6,label='g, N.D.')
plt.plot(t_days_o_upper,mag_o_upper,marker='v',color='orange',linestyle='',markersize=6,label='o, N.D.')
plt.plot(t_days_clear_upper,mag_clear_upper,'mv',markersize=6,label='clear, N.D.')
plt.plot(t_days_CV_upper,mag_CV_upper,marker='v',markerfacecolor='darkgrey',markersize=6,ls='',label='CV, N.D.')
plt.xlabel(r'$t - t_2$ [days]',fontsize=size)
plt.ylabel('Magnitude',fontsize=size)
plt.xlim([-2,120])
plt.ylim([22,9])
plt.legend(ncol=2,fontsize=15,loc='best')
plt.tight_layout()
plt.savefig('sn_2023ixf_early_120days.pdf')
plt.savefig('sn_2023ixf_early_120days.png')

plt.show()
