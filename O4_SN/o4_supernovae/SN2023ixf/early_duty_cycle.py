# Python script to plot 
import numpy as np
from numpy import *
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from pylab import *
from plot_defaults import *

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

br = '#773300'
gr = '#444444'


# Color-blind friendly, based on pallet:
# https://jacksonlab.agronomy.wisc.edu/2016/05/23/15-level-colorblind-friendly-palette/
#clr1 = '#009292'
#clr2 = '#db6d00'
#clr3 = '#6db6ff'
clr1 = 'darkorchid'
clr2 = 'red'
clr3 = 'deepskyblue'
# Early photometry (Rosa Poggiani, March 2024)
# Object name
obj_Id='SN 2023ixf'
# t2, first detection
t2 = 60082.82611
# Time range
mjd_min=60081
mjd_max=60084
# File name
fild = obj_Id.replace(" ","_").lower()
# AAVSO data
fil_aavso_CV = fild+'_early_aavso_CV.dat'
# Circulars/Telegrams data
fil_B = fild+'_early_B.dat'
fil_V = fild+'_early_V.dat'
fil_R = fild+'_early_R.dat'
fil_g = fild+'_early_g.dat'
fil_r = fild+'_early_r.dat'
fil_clear = fild+'_early_clear.dat'
fil_itagaki = fild+'_early_itagaki_clear.dat'
fil_citizen_V = fild+'_early_citizen_V.dat'
fil_CV = fild+'_early_CV.dat'
fil_ztf_g = fild+'_early_ztf_g.dat'
# Upper limit data 
fil_g_upper = fild+'_early_g_upper.dat'
fil_o_upper = fild+'_early_o_upper.dat'
fil_clear_upper = fild+'_early_clear_upper.dat'
fil_CV_upper = fild+'_early_CV_upper.dat'
# AAVSO photometric data
mjd_aavso_CV,mag_aavso_CV,magerr_aavso_CV=np.loadtxt(fil_aavso_CV,unpack=True,usecols=(0,1,2),delimiter=',',skiprows=1)
# Circulars/telegrams/prepints photometric data
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

# Duty cycle (Yanyan Zheng, March 2024)
# One day in seconds
od = 86400
# Compute % of science segments contained in GPS interval
# segs is the vector containing the duration of the segments
def duty(GPSs,GPSe,segS,segE,segs):
  n = len(segs)
  time = sum(segs)
  if segS < GPSs:
    time = time - (GPSs - segS)
  if segE > GPSe:
    time = time - (segE - GPSe)
  return 100*time/(GPSe-GPSs)  
# SN2023ixf
T = 15000000
SN2023ixftimes = [1368042593, 1368474593]
winSN2023ixf = SN2023ixftimes[1] - SN2023ixftimes[0]
tToTSN2023ixf = T - SN2023ixftimes[1]
SN2023ixfH1 = genfromtxt("./DQ/O4_H1H1_SN2023ixf_BKG_736b2ec0_dq.dq2")
SN2023ixfL1 = genfromtxt("./DQ/O4_L1L1_SN2023ixf_BKG_736b2ec0_dq.dq2")
SN2023ixfH1bar = []
for i in range(len(SN2023ixfH1)):
    SN2023ixfH1bar.append((SN2023ixfH1[i,0]+tToTSN2023ixf, SN2023ixfH1[i,1]-SN2023ixfH1[i,0]))
 #   SN2023ixfH1bar.append((SN2023ixfH1[i,0], SN2023ixfH1[i,1]-SN2023ixfH1[i,0]))
SN2023ixfL1bar = []
for i in range(len(SN2023ixfL1)):
    SN2023ixfL1bar.append((SN2023ixfL1[i,0]+tToTSN2023ixf, SN2023ixfL1[i,1]-SN2023ixfL1[i,0]))
 #   SN2023ixfL1bar.append((SN2023ixfL1[i,0], SN2023ixfL1[i,1]-SN2023ixfL1[i,0]))
SN2023ixfH1dd = duty(SN2023ixftimes[0], SN2023ixftimes[1], SN2023ixfH1[0,0], SN2023ixfH1[len(SN2023ixfH1)-1,1], SN2023ixfH1[:,1]-SN2023ixfH1[:,0])
SN2023ixfH1d = "%.2f" %SN2023ixfH1dd
SN2023ixfL1dd = duty(SN2023ixftimes[0], SN2023ixftimes[1], SN2023ixfL1[0,0], SN2023ixfL1[len(SN2023ixfL1)-1,1], SN2023ixfL1[:,1]-SN2023ixfL1[:,0])
SN2023ixfL1d = "%.2f" %SN2023ixfL1dd
print(SN2023ixfH1dd)
print(winSN2023ixf)
print(sum(SN2023ixfH1[:,1]-SN2023ixfH1[:,0]))
print(sum(SN2023ixfH1[:,1]-SN2023ixfH1[:,0])/winSN2023ixf)
#print(SN2023ixfH1bar)
#print(SN2023ixfL1bar)

# Duty cycle figure with MJD as abscissa
fig, ax1 = plt.subplots(figsize=(20,6))
plt.gcf().subplots_adjust(bottom=0.15)
plt.gcf().subplots_adjust(top=0.981)
plt.gcf().subplots_adjust(left=0.05)
plt.gcf().subplots_adjust(right=0.976)
ax1 = plt.gca()
# Duty cycle as inset in figure of early photometry
ax1.errorbar(mjd_B,mag_B,magerr_B,fmt='bo',markersize=8,ls='',label='B')
ax1.errorbar(mjd_V,mag_V,magerr_V,fmt='go',markersize=8,ls='',label='V')
ax1.errorbar(mjd_R,mag_R,magerr_R,fmt='ro',markersize=8,ls='',label='R')
ax1.errorbar(mjd_CV,mag_CV,magerr_CV,marker='o',markerfacecolor='darkgrey',ecolor='darkgrey',markersize=8,ls='',label='CV')
ax1.errorbar(mjd_aavso_CV,mag_aavso_CV,magerr_aavso_CV,marker='o',markerfacecolor='darkgrey',ecolor='darkgrey',markersize=8,ls='')
ax1.errorbar(mjd_g,mag_g,magerr_g,fmt='g+',markersize=8,ls='',label='g')
ax1.errorbar(mjd_r,mag_r,magerr_r,fmt='r+',markersize=8,ls='',label='r')
ax1.errorbar(mjd_clear,mag_clear,magerr_clear,fmt='ms',markersize=8,ls='',label='clear')
ax1.plot(mjd_clear_itagaki,mag_clear_itagaki, 'ms',markersize=8)
ax1.errorbar(mjd_citizen_V,mag_citizen_V,magerr_citizen_V,fmt='go',markersize=8,ls='')
ax1.plot(mjd_g_upper,mag_g_upper,'gv',markersize=8,label='g, N.D.')
ax1.plot(mjd_o_upper,mag_o_upper,marker='v',color='orange',linestyle='',markersize=8,label='o, N.D.')
ax1.plot(mjd_clear_upper,mag_clear_upper,'mv',markersize=8,label='clear, N.D.')
ax1.plot(mjd_CV_upper,mag_CV_upper,marker='v',markerfacecolor='darkgrey',markersize=8,ls='',label='CV, N.D.')
ax1.set_xlabel('MJD',fontsize=size)
ax1.set_ylabel('Magnitude',fontsize=size)
ax1.set_xlim([60081,60084])
ax1.set_ylim([22,13])
ax1.xaxis.set_major_formatter(StrMethodFormatter('{x:.1f}')) 
ax1.axvline(x=t2,color='k',linewidth=1,linestyle='--')
ax1.annotate(r'$t_2$',(60082.85, 14),fontsize=30)
ax1.annotate('SN 2023ixf', (60081.1, 20), fontsize=22)
ax1.legend(ncol=2,fontsize=15,loc='lower right')
ax1.grid(which='major',alpha=0.7,linestyle='--')
left, bottom, width, height = [0.08, 0.55, 0.51, 0.35]
ax2 = fig.add_axes([left, bottom, width, height])
sh = 0.0
shd = 1.4
ax2.broken_barh(SN2023ixfH1bar, (2.5, 2.5), facecolors=clr2, linewidth=0)
ax2.broken_barh(SN2023ixfL1bar, (6, 2.5), facecolors=clr3, linewidth=0)
ax2.annotate('H1', (T-4.8*od, 4), fontsize=18,horizontalalignment='left', verticalalignment='top')
ax2.annotate('L1', (T-4.8*od, 7.5), fontsize=18,horizontalalignment='left', verticalalignment='top')
ax2.annotate(SN2023ixfH1d+'\%', (T-4.0*od, 4),fontsize=18,horizontalalignment='right', verticalalignment='top')
ax2.annotate(SN2023ixfL1d+'\%', (T-4.0*od, 7.5),fontsize=18,horizontalalignment='right', verticalalignment='top')
ax2.annotate('IFO', (T-4.85*od,9.3), fontsize=18, 
            horizontalalignment='left', verticalalignment='top')
ax2.annotate('Duty Factor', (T-4.55*od,9.3), fontsize=18, 
            horizontalalignment='left', verticalalignment='top')
ax2.tick_params(direction='in',axis='both')
ax2.xaxis.set_ticks_position('both')
ax2.set_yticks([])
ax2.set_xticks([T, T-1*od, T-2*od, T-3*od, T-4*od, T-5*od, T-6*od, T-7*od, T-8*od])
ax2.set_xticklabels(['$0$', '$-1$','$-2$','$-3$','$-4$','$-5$', '$-6$', '$-7$', '$-8$'],)
ax2.set_ylim(1,10)
ax2.set_xlim(T-5*od,T+0*od)
ax2.set_xlabel(r'$t - t_2$ [days]',size = 22)
xticks = ax2.get_xticks()
xlabels = ax2.get_xticklabels()
xlabels[0].set_position((T-0.5*od, 0))
#ax2.set_tick_params(left = False, bottom = False) 
#plt.savefig('sn_2023ixf_early_dutycycle_v2.png')
plt.savefig('sn_2023ixf_early_dutycycle_mjd.pdf')


# Time in days after the explosion
t_days_B = mjd_B - t2
t_days_V = mjd_V - t2
t_days_R = mjd_R - t2
t_days_CV = mjd_CV - t2
t_days_aavso_CV = mjd_aavso_CV - t2
t_days_g = mjd_g - t2
t_days_r = mjd_r - t2
t_days_clear = mjd_clear - t2
t_days_clear_itagaki = mjd_clear_itagaki - t2
t_days_citizen_V = mjd_citizen_V - t2
t_days_g_upper = mjd_g_upper - t2
t_days_o_upper = mjd_o_upper - t2
t_days_clear_upper = mjd_clear_upper - t2
t_days_CV_upper = mjd_CV_upper - t2


# Duty cycle figure with days from explosion as abscissa
fig, ax1 = plt.subplots(figsize=(20,6))
plt.gcf().subplots_adjust(bottom=0.15)
plt.gcf().subplots_adjust(top=0.981)
plt.gcf().subplots_adjust(left=0.05)
plt.gcf().subplots_adjust(right=0.976)
ax1 = plt.gca()
# Duty cycle as inset in figure of early photometry
ax1.errorbar(t_days_B,mag_B,magerr_B,fmt='bo',markersize=8,ls='',label='B')
ax1.errorbar(t_days_V,mag_V,magerr_V,fmt='go',markersize=8,ls='',label='V')
ax1.errorbar(t_days_R,mag_R,magerr_R,fmt='ro',markersize=8,ls='',label='R')
ax1.errorbar(t_days_CV,mag_CV,magerr_CV,marker='o',markerfacecolor='darkgrey',ecolor='darkgrey',markersize=8,ls='',label='CV')
ax1.errorbar(t_days_aavso_CV,mag_aavso_CV,magerr_aavso_CV,marker='o',markerfacecolor='darkgrey',ecolor='darkgrey',markersize=8,ls='')
ax1.errorbar(t_days_g,mag_g,magerr_g,fmt='g+',markersize=8,ls='',label='g')
ax1.errorbar(t_days_r,mag_r,magerr_r,fmt='r+',markersize=8,ls='',label='r')
ax1.errorbar(t_days_clear,mag_clear,magerr_clear,fmt='ms',markersize=8,ls='',label='clear')
ax1.plot(t_days_clear_itagaki,mag_clear_itagaki, 'ms',markersize=8)
ax1.errorbar(t_days_citizen_V,mag_citizen_V,magerr_citizen_V,fmt='go',markersize=8,ls='')
ax1.plot(t_days_g_upper,mag_g_upper,'gv',markersize=8,label='g, N.D.')
ax1.plot(t_days_o_upper,mag_o_upper,marker='v',color='orange',linestyle='',markersize=8,label='o, N.D.')
ax1.plot(t_days_clear_upper,mag_clear_upper,'mv',markersize=8,label='clear, N.D.')
ax1.plot(t_days_CV_upper,mag_CV_upper,marker='v',markerfacecolor='darkgrey',markersize=8,ls='',label='CV, N.D.')
ax1.set_xlabel(r'$t - t_2$ [days]',fontsize=size)
ax1.set_ylabel('Magnitude',fontsize=size)
ax1.set_xlim([-2,1.2])
ax1.set_ylim([22,13])
ax1.xaxis.set_major_formatter(StrMethodFormatter('{x:.1f}')) 
ax1.axvline(x=0,color='k',linewidth=1,linestyle='--')
ax1.annotate(r'$t_2$',(60082.85-t2, 14),fontsize=30)
ax1.annotate('SN 2023ixf', (60081.1-t2, 20), fontsize=22)
ax1.legend(ncol=2,fontsize=15,loc='lower right')
ax1.grid(which='major',alpha=0.7,linestyle='--')
left, bottom, width, height = [0.08, 0.55, 0.45, 0.35]
ax2 = fig.add_axes([left, bottom, width, height])
sh = 0.0
shd = 1.4
ax2.broken_barh(SN2023ixfH1bar, (2.5, 2.5), facecolors=clr2, linewidth=0)
ax2.broken_barh(SN2023ixfL1bar, (6, 2.5), facecolors=clr3, linewidth=0)
ax2.annotate('H1', (T-4.8*od, 4), fontsize=18,horizontalalignment='left', verticalalignment='top')
ax2.annotate('L1', (T-4.8*od, 7.5), fontsize=18,horizontalalignment='left', verticalalignment='top')
ax2.annotate(SN2023ixfH1d+'\%', (T-4.0*od, 4),fontsize=18,horizontalalignment='right', verticalalignment='top')
ax2.annotate(SN2023ixfL1d+'\%', (T-4.0*od, 7.5),fontsize=18,horizontalalignment='right', verticalalignment='top')
ax2.annotate('IFO', (T-4.85*od,9.3), fontsize=18, 
            horizontalalignment='left', verticalalignment='top')
ax2.annotate('Duty Factor', (T-4.55*od,9.3), fontsize=18, 
            horizontalalignment='left', verticalalignment='top')
ax2.tick_params(direction='in',axis='both')
ax2.xaxis.set_ticks_position('both')
ax2.set_yticks([])
ax2.set_xticks([T, T-1*od, T-2*od, T-3*od, T-4*od, T-5*od, T-6*od, T-7*od, T-8*od])
ax2.set_xticklabels(['$0$', '$-1$','$-2$','$-3$','$-4$','$-5$', '$-6$', '$-7$', '$-8$'],)
ax2.set_ylim(1,10)
ax2.set_xlim(T-5*od,T+0*od)
ax2.set_xlabel(r'$t - t_2$ [days]',size = 22)
xticks = ax2.get_xticks()
xlabels = ax2.get_xticklabels()
xlabels[0].set_position((T-0.5*od, 0))
#ax2.set_tick_params(left = False, bottom = False) 
#plt.savefig('sn_2023ixf_early_dutycycle_v2.png')
plt.savefig('sn_2023ixf_early_dutycycle_days.pdf')
plt.savefig('sn_2023ixf_early_dutycycle_days.png')

plt.show()

from astropy.time import Time

mjd = 60081
utc_time = Time(mjd, format='mjd').iso
print (utc_time)

mjd = 60084
utc_time2 = Time(mjd, format='mjd').iso
print (utc_time2)

times = ['2023-05-18T19:49:35']
t = Time(times, format= 'isot', scale='utc')
print (t.mjd)

mjd = 60082.82611
utc_time = Time(mjd, format='mjd').iso
print (utc_time)

mjd = 60082.82611
utc_time = Time(mjd, format='mjd').gps
print (utc_time)

plt.rcParams.update(matplotlib.rcParamsDefault)
