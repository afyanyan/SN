import numpy as np
import pandas as pd

# Compilation of distances from NED
filin = 'm101_distance.csv'

# Read data
df = pd.read_csv(filin)

# Remove estimation with statistical method
df_allns = df.loc[df['Method'] != 'Statistical']

# Median distance
d_allns = np.median(df_allns['D(Mpc)'])
# Standard deviation
de_allns = np.std(df_allns['D(Mpc)'])
print('Median distance (Mpc): ',round(d_allns,2))
print('Standard deviation of distances (Mpc): ',round(de_allns,2))

# Distance modulus DM, distance (Mpc)
modulus = df_allns['(m-M)']
modulus_error = df_allns['err(m-M)']
distance = df_allns['D(Mpc)']