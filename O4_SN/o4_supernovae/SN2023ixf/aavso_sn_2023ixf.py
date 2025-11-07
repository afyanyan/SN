import urllib.request
import csv
import pandas as pd
import matplotlib.pyplot as plt

size = 16

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

plt.close('all')

# Downloading AAVSO data, adapted from "Eric Dose :: New Mexico Mira Project, Albuquerque" program

# Data URL
VSX_OBSERVATIONS_HEADER = 'https://www.aavso.org/vsx/index.php?view=api.delim'
VSX_DELIMITER = '@@@'  # NB: ',' fails as obsName values already have a comma.

# Object name
obj_id='SN 2023ixf'
print('Object id: ',obj_id)

# Object name with replacements suitable for WEB
obj_id_safe =obj_id.replace("+", "%2B").replace(" ", "+")
# Object identifier
parm_ident = '&ident=' + obj_id_safe
parm_delimiter = '&delimiter=' + VSX_DELIMITER
# Object URL
url = VSX_OBSERVATIONS_HEADER + parm_ident 
print('Object URL: ',url)

# Request data
byte_text = urllib.request.urlopen(url)
# Save data into variable
text = [line.decode('utf-8') for line in byte_text]
# Remove final '\r\n'
text = [rem.strip('\r\n') for rem in text]
# Split individual data    
data = [line.split('@$@') for line in text]

# Save data in file including "fainter than" data
fild = obj_id.replace(" ","_").lower()+'_aavso.csv'
with open(fild, 'w') as f:
    write = csv.writer(f,delimiter=';')
    write.writerow(data[0])
    write.writerows(data[1:])
f.close()

# Remove "fainter than" data
df = pd.read_csv(fild,sep=';')
df_magg = df[(df['fainterThan'] == 0)]
nlin=len(df_magg)
print('Total number of observations = ',nlin)

# Convert Julian Date to Modified Julian Date
df_magg['MJD']=df_magg['JD'].to_numpy()-2400000.5

# Remove discrepant observations
df_mag1 = df_magg[(df_magg['by'] != 'TVIB')]
df_mag2 = df_mag1[(df_mag1['by'] != 'PJEE')]
df_mag3 = df_mag2[(df_mag2['by'] != 'SAME')]
df_mag4 = df_mag3[(df_mag3['by'] != 'CGIA')]
df_mag5 = df_mag4[(df_mag4['by'] != 'ZJIE')]
df_mag6 = df_mag5[(df_mag5['by'] != 'GEA')]
df_mag7 = df_mag6[(df_mag6['by'] != 'MDJ')]
df_mag = df_mag7[(df_mag7['by'] != 'MAND')]

# Save data of different bands in separate files 
filn = fild.split('.')[0]
fileB = filn+'_B.dat'
fileV = filn+'_V.dat'
fileR = filn+'_R.dat'
fileI = filn+'_I.dat'
fileCV = filn+'_CV.dat'
print('Processing ',fild)

# Dataframes for photometric filters
df_B = df_mag[(df_mag['band'] == 'B')]
df_V = df_mag[(df_mag['band'] == 'V')]
df_R = df_mag[(df_mag['band'] == 'R')]
df_I = df_mag[(df_mag['band'] == 'I')]
df_CV = df_mag[(df_mag['band'] == 'CV')]

# Drop data points without photometric error
df_V.dropna(subset=['uncert'], inplace=True)
df_B.dropna(subset=['uncert'], inplace=True)
df_R.dropna(subset=['uncert'], inplace=True)
df_I.dropna(subset=['uncert'], inplace=True)
df_CV.dropna(subset=['uncert'], inplace=True)

# Save data of diffetent filters in separate files

# B band
mjdb=df_B['MJD'].to_numpy()
magb=df_B['mag'].to_numpy()
magerrb = df_B["uncert"].to_numpy()
print('Number of B observations = ',len(mjdb)) 
df_B.to_csv(fileB,sep=',',columns=['MJD','mag','uncert'],header=True,index=False)
# V band
mjdv = df_V["MJD"].to_numpy()
magv = df_V["mag"].to_numpy()
magerrv = df_V["uncert"].to_numpy()
print("Number of V observations = ", len(mjdv))
plt.figure()
df_V.to_csv(fileV, sep=",", columns=["MJD", "mag", "uncert"], header=True, index=False)
# R band
mjdr=df_R['MJD'].to_numpy()
magr=df_R['mag'].to_numpy()
magerrr = df_R["uncert"].to_numpy()
print('Number of R observations = ',len(mjdr)) 
df_R.to_csv(fileR,sep=',',columns=['MJD','mag','uncert'],header=True,index=False)
# I band
mjdi=df_I['MJD'].to_numpy()
magi=df_I['mag'].to_numpy()
print('Number of I observations = ',len(mjdi))
magerri = df_I["uncert"].to_numpy()
df_I.to_csv(fileI,sep=',',columns=['MJD','mag','uncert'],header=True,index=False)
# CV band
mjdcv=df_CV['MJD'].to_numpy()
magcv=df_CV['mag'].to_numpy()
magerrcv = df_CV["uncert"].to_numpy()
print('Number of CV observations = ',len(mjdcv))
df_CV.to_csv(fileCV,columns=['MJD','mag','uncert'],header=True,index=False)

# General plot with B, V, R, I data
plt.figure()
plt.plot(mjdb,magb,'bo',markersize=2,label='B')
plt.plot(mjdv,magv,'go',markersize=2,label='V')
plt.plot(mjdr,magr,'ro',markersize=2,label='R')
plt.plot(mjdi,magi,'yo',markersize=2,label='I')
plt.plot(mjdcv,magcv,'mo',markersize=2,label='CV')
plt.xlabel('MJD')
plt.ylabel('Magnitude')
plt.title(obj_id)
plt.ylim(plt.ylim()[::-1])
plt.legend()
plt.tight_layout()

#plt.rcParams.update(matplotlib.rcParamsDefault)

plt.show()
