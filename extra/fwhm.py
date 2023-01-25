import numpy as np
import sys
from glob import glob
from astropy.io import ascii
from astropy.table import Table


peak_max = 65000
peak_min = 1000
mag_max = -6
mag_min = -22
fwhm_max = 8
fwhm_min = 0
percentile = 96
nbins = 200

file = sys.argv[1]

tab = ascii.read(file)

fwhm = tab['FWHM_IMAGE']
peak = tab['FLUX_MAX']
mag = tab['MAG_AUTO']
# flag = tab['FLAGS']

hfwhm = (fwhm > fwhm_min) & (fwhm < fwhm_max)
hpeak = (peak > peak_min) & (peak < peak_max)
hmag = (mag > mag_min) & (mag < mag_max)

hist = np.histogram(fwhm[hfwhm & hpeak & hmag], bins=nbins)
hist_criteria = np.percentile(hist[0], percentile)
idx = np.where(hist[0] > hist_criteria)[0]

fwhm_star = sum(hist[0][idx] * hist[1][idx]) / sum(hist[0][idx])

if fwhm_star < 1.75:
	filter_file = 'gauss_1.5_3x3.conv'
elif fwhm_star < 2.3:
	filter_file = 'gauss_2.0_3x3.conv'
elif fwhm_star < 2.8:
	filter_file = 'gauss_2.5_5x5.conv'
elif fwhm_star < 3.7:
	filter_file = 'gauss_3.0_7x7.conv'
elif fwhm_star < 4.7:
	filter_file = 'gauss_4.0_7x7.conv'
elif fwhm_star < 5.7:
	filter_file = 'gauss_5.0_9x9.conv'
else:
    filter_file = 'gauss_6.0_13x13.conv'
# elif fwhm_star < 7.5:
# 	filter_file = 'gauss_6.0_13x13.conv'
# elif fwhm_star < 11.5:
# 	filter_file = 'gauss_9.0_19x19.conv'
# elif fwhm_star < 15.5:
# 	filter_file = 'gauss_15.0_31x31.conv'
# else:
# 	filter_file = 'gauss_16.0_33x33.conv'


sys.exit(filter_file)


