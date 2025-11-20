Early photometric data for SN 2023ixf
=====================================


The photometric data and the non detections used for Figure 1 are collected in different files according to the photometric filter and the Astronote/Astronomical Telegram/published papers where they appeared. Data file names have extension ".dat".


Time range covered by data: from MJD = 60081 to MJD = 60084



Each file includes comment lines with the reference to the data source before listing the related data.

TNSAN: Transient Name Server AstroNote

ATel: Astronomical Telegram

Reference to published papers reported in the standard format


The name of the files reporting observations start with "sn_2023ixf_early" and end with the label for the photometric filters described below.

==

Files reporting obervations have three columns separated by commas:

- Modified Julian Date (MJD)
- magnitude
- magnitude uncertainty

==

The name of the files reporting non detections start with "sn_2023ixf_early" and end with "upper_" followed by the label for the photometric filters.

Files reporting non detections have two columns:
- Modified Julian Date (MJD)
- magnitude

==

Photometric filter labels and description:

- B: Johnson B filter

- V: Johnson V filter

- R: Johnson R filter

- o: orange filter by ATLAS survey

- g: Sloan g filter

- r: Sloan r filter

- CV: clear (unfiltered observation) reduced to V sequence

- clear: clear (unfiltered observation) without additional information

==

References for data in files with observations

sn_2023ixf_early_B.dat: TNSAN

sn_2023ixf_early_V.dat: TNSAN, Nat 627 (2024) 754

sn_2023ixf_early_R.dat: TNSAN

sn_2023ixf_early_g.dat: TNSAN

sn_2023ixf_early_r.dat: TNSAN

sn_2023ixf_early_CV.dat: TNSAN, ATel

sn_2023ixf_early_clear.dat: TNSAN

sn_2023ixf_early_itagaki_clear.dat: TNSAN discovery observation

sn_2023ixf_early_atlas_o.dat: ATLAS survey

sn_2023ixf_early_ztf_g.dat: ZTF survey, single detection in time range MJD=60181-60084 in file sn_2023ixf_ztf.csv downloaded from Lasair

sn_2023ixf_early_aavso_CV.dat: AAVSO data. The script aavso_sn_2023ixf.py downloads the data from AAVSO and saves them as file sn_2023ixf_aavso.csv (file in directory extending beyond MJD=60084). The script also applies some cuts and saves data of different filters into separate files. Check initial part of sn_2023ixf_aavso_CV.dat with sn_2023ixf_early_aavso_CV.dat used to generate Figure 1

sn_2023ixf_early_citizen_V.dat: RNAAS 7 (2023) 141

==

References for data in files with non detections

sn_2023ixf_early_CV_upper.dat: TNSAN

sn_2023ixf_early_g_upper.dat: TNSAN

sn_2023ixf_early_o_upper.dat: TNSAN

sn_2023ixf_early_clear_upper.dat: TNSAN

