"""

Usage:

    In data directory:
    $ python path_to_script/science_background_unevenness_removal.py filename

    In script directory:
    $ python science_background_unevenness_removal.py path_to_files/filename

This code takes in a mosaic science frame and noobj.fits (if not exist then estimated it),
find the corresponding optic_flat_asux_filtermask_600.fits, calculate the sky median value in the filter region,
subtract the offsets from the extension images that have cluster and save it as .unevenness_removed.fits.


"""
import time
from astropy.io import fits, ascii
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
# from astropy.coordinates import SkyCoord
# from astropy.table import Table
# from glob import glob
import numpy as np
# import matplotlib.pyplot as plt
#plt.ion()
from photutils import MedianBackground, SExtractorBackground, Background2D
from photutils.aperture import CircularAnnulus, aperture_photometry, CircularAperture, EllipticalAnnulus, EllipticalAperture
from astropy.stats import SigmaClip,sigma_clip,sigma_clipped_stats
from astropy.visualization import simple_norm
from astropy.wcs import WCS
# from astropy.nddata.utils import Cutout2D
# from astropy import units as u
import subprocess
import os
import sys
from pathlib import Path
from scipy.ndimage import gaussian_filter as gf

if len(sys.argv) < 2:
    print(__doc__)
    sys.exit()


script_dir = sys.path[0] + '/'
bias_dark_flats_dir = os.environ['_90PRIME_BIAS_DARK_FLATS_DIR']

# Parameters for removing objects if necessary
threshold = 3
# when comparing the skybackground and annulus background median, criteria is 
# median(annulus)- threshold * std of the mean < median(sky) + threshold * std of the mean
peak_thres = 10000
# peak_thres=45   # for clash images
large_boxratio = 18 # this*a as halfbox
small_boxratio = 12 # this*a as halfbox
large_aratio = 4.
small_aratio = 3.
max_ra = 100 # max allowed radius (semimajor axis a) in pixel
min_d2edge_thres = 5   # if cx, cy is within this distance to edge, don't remove the object

pathfname = sys.argv[1]
fname = pathfname.split('/')[-1]
path = pathfname[:-len(fname)]

# figure out cluster name

all_cluster_names = ['abell1576', \
                     'abell370', \
                     'abell611', \
                     'macs0329', \
                     'macs0717', \
                     'macs1115', \
                     'macs1149', \
                     'rxj1532', \
                     'zwicky1953']
all_cluster_names = np.array(all_cluster_names)
fnamesplit = fname.split('_')

clustername = ''
for i in range(len(fnamesplit)):
    if (fnamesplit[i] == all_cluster_names).any():
        clustername = fnamesplit[i]
if clustername == '':
    print("Cluster name not found!")
    sys.exit()


f = fits.open(pathfname)
hdr = f[0].header
obsdate = hdr['date']
filt = hdr['filter'].lower()
month = obsdate.split('-')[1]
if month == '02':
    month_dir = 'Feb'
elif month == '03':
    month_dir = 'Mar'
else:
    print("Couldn't determine the observing month of the mosaic!")
    sys.exit()
o600 = fits.open(bias_dark_flats_dir + '/optic_masks/{}/optic_flat_{}_filtermask_600.fits'.format(month_dir, filt))


# run source extractor
pathwtname = pathfname.replace('.fits','.wt.fits')
pathcatname = pathfname.replace('.fits','.remove_obj.cat')
pathnoobjfname = pathfname.replace('.fits','.noobj.fits')

if Path(pathcatname).is_file():
    print('SExtractor catalog:\t{}.remove_obj.cat already exists.'.format(fname.replace('.fits','')))
else:
    print('Running SExtractor...')
    subprocess.run(['sex', '-c', script_dir+'extra/default.sex', pathfname, '-PARAMETERS_NAME', \
                    script_dir+'/extra/param.param', '-WEIGHT_IMAGE', pathwtname, \
                    '-FILTER_NAME', script_dir+'/extra/gauss_3.0_7x7.conv', \
                    '-CATALOG_NAME', pathcatname, '-VERBOSE_TYPE', 'QUIET'])
    print('SExtractor done.')

# If mosaic, determine which location needs object removal
locs = np.array(['ne','nw','se','sw'])
extentions_dict = {'ne':[1,2,3,4], \
              'nw':[9,10,11,12], \
              'se':[5,6,7,8], \
              'sw':[13,14,15,16],}

loc = ''

for i in range(len(fnamesplit)):
    if (fnamesplit[i][:2] == locs).any():
        loc = fnamesplit[i][:2]
if loc == '':
    print("Can't find location for mosaic image!")
    sys.exit()

extensions = extentions_dict[loc]

cat = ascii.read(pathcatname)

skymeans = []
skymeds = []
skystds = []

# if noobj.fits doesn't exist, loop over extensions to remove objects, otherwise just use noobj.fits
for i in range(len(extensions)):

    # skip object removal if noobj.fits exist
    if Path(pathfname.replace('.fits','.noobj.fits')).is_file() == False:
        f0 = f[extensions[i]].data
        f1 = f0.copy()
        outmask = f1==0

        # fsky0 = fsky[extensions[i]].data

        # get the sub table
        cat1 = cat[cat['EXT_NUMBER'] == extensions[i]]
            
        x = cat1['X_IMAGE'] - 1
        y = cat1['Y_IMAGE'] - 1

        yshape, xshape = f0.shape

        # loop over each object in catalog to remove it
        for gi in range(len(x)):
            cx = x[gi].astype(int)
            cy = y[gi].astype(int)
            obj = cat1[gi]
            peak = obj['FLUX_MAX']
            min_d2edge = min(cx, cy, xshape - cx, yshape - cy)

            if f1[cy, cx] == 0:
                pass
            elif min_d2edge < min_d2edge_thres:
                pass
            else:
                dx = x[gi] - cx
                dy = y[gi] - cy
                a = obj['A_IMAGE']
                b = obj['B_IMAGE']
                theta = obj['THETA_IMAGE'] * np.pi/180
                if peak > peak_thres:
                    aratio = large_aratio
                    boxratio = large_boxratio
                else:
                    aratio = small_aratio
                    boxratio = small_boxratio
                ra = min(aratio * a, max_ra)
                rb = min(aratio * b, max_ra)
                b2aratio = rb/ra
                halfbox=min(int(boxratio*a),min_d2edge-1)
                ft = f1[cy-halfbox:cy+halfbox+1, cx-halfbox:cx+halfbox+1]
                position = (halfbox+dx, halfbox+dy)

                if peak > peak_thres:
                    ra = (ra+rb)/2
                    aa = CircularAnnulus(position,r_in=ra,r_out=ra+5)
                else:
                    aa = EllipticalAnnulus(position,a_in=ra,a_out=ra+5,b_out=(ra+5)*b2aratio,theta=theta)

                ftdat = sigma_clip(ft[ft>0],sigma=2.5,masked=False)
                med1 = np.median(ftdat)
                std1 = np.std(ftdat)
                stdmean1 = std1/np.sqrt(len(ftdat))
                thres = med1+threshold*stdmean1
                dat = aa.to_mask('center').multiply(ft)
                dat1 = sigma_clip(dat[dat>0],sigma=2.5,masked=False)
                dat1lower = np.median(dat1)-threshold*np.std(dat1)/np.sqrt(len(dat1))
                r_increment=2
                while dat1lower > thres and ra < max_ra:
                    #print(r,dat1lower,thres)
                    ra = ra+r_increment
                    rb = ra*b2aratio
                    halfbox = min(halfbox+r_increment+1,min_d2edge-1)
                    ft = f1[cy-halfbox:cy+halfbox+1,cx-halfbox:cx+halfbox+1]
                    position = (halfbox+dx,halfbox+dy)
                    if peak > peak_thres:
                        aa = CircularAnnulus(position,r_in=ra,r_out=ra+5)
                    else:
                        aa = EllipticalAnnulus(position,a_in=ra,a_out=ra+5,b_out=(ra+5)*b2aratio,theta=theta)
                    ftdat = sigma_clip(ft[ft>0],sigma=2.5,masked=False)
                    med1 = np.median(ftdat)
                    std1 = np.std(ftdat)
                    stdmean1 = std1/np.sqrt(len(ftdat))
                    thres = med1+threshold*stdmean1
                    dat = aa.to_mask('center').multiply(ft)
                    dat1 = sigma_clip(dat[dat>0],sigma=2.5,masked=False)
                    dat1lower = np.median(dat1)-threshold*np.std(dat1)/np.sqrt(len(dat1))
                #ap=EllipticalAperture(position,r*a,r*b,theta*np.pi/180)
                if peak > peak_thres:
                    ap = CircularAperture(position,ra)
                else:
                    ap = EllipticalAperture(position,ra,rb,theta)
                am = ap.to_mask(method='center')
                mask = am.to_image((2*halfbox+1,2*halfbox+1))
                mlength = len(mask[mask>0])
                #mean1,med1,std1=sigma_clipped_stats(ft[ft>0],sigma=2.5)
                ft[mask>0] = np.random.normal(med1,std1,mlength)
                # ft[mask>0] = 0.0
                f1[cy-halfbox:cy+halfbox+1,cx-halfbox:cx+halfbox+1] = ft
    else:
        print('{}.noobj.fits already exists, skip object removal'.format(fname.replace('.fits','')))
        f0 = f[extensions[i]].data
        f1 = fits.getdata(pathfname.replace('.fits','.noobj.fits'), ext=extensions[i])
        outmask = f1==0

    o600_0 = o600[extensions[i]].data
    dat = f1[o600_0>0]
    skymean1, skymed1, skystd1 = sigma_clipped_stats(dat, sigma=3.0)
    skymeans.append(skymean1)
    skymeds.append(skymean1)
    skystds.append(skystd1)

skymeans = np.array(skymeans)
skymeds = np.array(skymeds)
skystds = np.array(skystds)

offsets = skymeds - min(skymeds)
# idxi = 0
print(offsets)

for i in range(len(extensions)):
    f0 = f[extensions[i]].data
    f0[f0!=0] = f0[f0!=0] - offsets[i]
    f[extensions[i]].data = f0
    # idxi += 1

f.writeto(pathfname.replace('.fits','.unevenness_removed.fits'), overwrite=True)




