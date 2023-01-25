'''

In file folder:
    $ python path_to_script/starmask.py filename.fits (thres=8) (small_aratio=2.0) (large_aratio=4) (fits=false)

This code creates starmask.txt that contains info on each star mask's position and radius.

Requires:
    master_stack_clustername_cutout.fits
    master_stack_clustername_cutout.wt.fits
    clustername_star_sdss(panstarrs)_radec.csv

Creates:
    clustername_starmask.csv
    (filename.starmask.fits if fits=true)

'''
import numpy as np
from astropy.io import fits, ascii
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
import sys
from pathlib import Path
from astropy.nddata import Cutout2D
from astropy.convolution import convolve
from astropy.modeling.models import Gaussian2D
from photutils.psf import create_matching_kernel, TukeyWindow, TopHatWindow, SplitCosineBellWindow
from photutils.aperture import CircularAnnulus, aperture_photometry, CircularAperture, EllipticalAnnulus, EllipticalAperture
from astropy.stats import SigmaClip,sigma_clip,sigma_clipped_stats
from astropy.convolution import convolve
from glob import glob
from reproject import reproject_exact
import multiprocessing as mp
import subprocess
import pickle
from astropy.table import Table

# object removal parameters
threshold = 8
# when comparing the skybackground and annulus background median, criteria is 
# median(annulus)- threshold * std of the mean < median(sky) + threshold * std of the mean
peak_thres = 10000
# peak_thres=45   # for clash images
large_boxratio = 18 # this*a as halfbox
small_boxratio = 12 # this*a as halfbox
large_aratio = 4.
small_aratio = 2.0
max_ra = 150 # max allowed radius (semimajor axis a) in pixel
min_d2edge_thres = 5   # if cx, cy is within this distance to edge, don't remove the object

fill_val0 = -100000 # the beginning value to fill in the cutouts
make_fits = False

script_dir = sys.path[0] + '/'
if len(sys.argv) < 2:
    print(__doc__)
    sys.exit()
pathfname = sys.argv[1]
fname = pathfname.split('/')[-1]
path = pathfname[:-len(fname)]
wtname = fname.replace('.fits','.wt.fits')

if len(sys.argv) > 2:
    nargs = len(sys.argv)
    for i in range(2, nargs):
        arg0 = sys.argv[i]
        if len(arg0.split('=')) == 1:
            print('Use "=" to separate keywords and their value')
            sys.exit()
        else:
            arg_type = arg0.split('=')[0].lower()
            arg_val = arg0.split('=')[-1].lower()

            if arg_type == 'thres':
                threshold = float(arg_val)
            elif arg_type == 'small_aratio':
                small_aratio = float(arg_val)
            elif arg_type == 'large_aratio':
                large_aratio = float(arg_val)
            elif arg_type == 'fits':
                if arg_val.lower() == 'true':
                    make_fits = True
                elif arg_val.lower() == 'false':
                    make_fits = False
            else:
                print('Unrecognized arguments!')
                print(__doc__)
                sys.exit()



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

filenamesplit = fname.split('_')
clustername = ''
for i in range(len(filenamesplit)):
    if (filenamesplit[i] == all_cluster_names).any():
        clustername = filenamesplit[i]
if clustername == '':
    print("Cluster name not found!")
    sys.exit()


catname = fname.replace('.fits','.remove_obj.cat')

# If removeobjcatname is not found, run source extractor
if Path(path + catname).is_file() == False:
    print('Run SExtractor...')
    subprocess.run(['sex', '-c', script_dir+'extra/remove_obj.sex', path+fname, '-PARAMETERS_NAME', \
                    script_dir+'/extra/param.param', '-WEIGHT_IMAGE', path+wtname, \
                    '-FILTER_NAME', script_dir+'/extra/gauss_3.0_7x7.conv', \
                    '-CATALOG_NAME', path+catname, '-VERBOSE_TYPE', 'QUIET'])
    print('SExtractor done.')


cat = ascii.read(path + catname)
coords = SkyCoord(cat['ALPHA_J2000'], cat['DELTA_J2000'], unit='deg')

# Find the stars catalog from either SDSS or panstarrs
if Path(path + clustername+'_star_sdss_radec.csv').is_file():
    stars = ascii.read(path + clustername+'_star_sdss_radec.csv')
elif Path(path + clustername+'_star_panstarrs_radec.csv'):
    stars = ascii.read(path + clustername+'_star_panstarrs_radec.csv')
else:
    print('Cannot find stars catalog!')
    sys.exit()
coord_stars = SkyCoord(stars['ra'], stars['dec'], unit='deg')

idx, d2d, d3d = coords.match_to_catalog_sky(coord_stars)
h = d2d.arcsec < 1.0

cat1 = cat[h]

f = fits.open(pathfname)
f0 = f[0].data
f0p = f0.copy()
hdr = f[0].header

# remove objects
print('Remove stars...')
xi = cat1['X_IMAGE'] - 1
yi = cat1['Y_IMAGE'] - 1
rai = cat1['ALPHA_J2000']
deci = cat1['DELTA_J2000']

positions_x = []
positions_y = []
positions_ra = []
positions_dec = []
radius = []

yshape, xshape = f0.shape
for si in range(len(xi)):
    cx = xi[si].astype(int)
    cy = yi[si].astype(int)
    obj = cat1[si]
    peak = obj['FLUX_MAX']
    min_d2edge = min(cx, cy, xshape - cx, yshape - cy)
    if f0[cx, cy] == 0:
        pass
    elif min_d2edge < min_d2edge_thres:
        pass
    else:
        dx = xi[si] - cx
        dy = yi[si] - cy
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
        ft = f0[cy-halfbox:cy+halfbox+1, cx-halfbox:cx+halfbox+1]
        position = (halfbox+dx, halfbox+dy)

        # if peak > peak_thres:
        #     ra = (ra+rb)/2
        #     aa = CircularAnnulus(position,r_in=ra,r_out=ra+5)
        # else:
        #     aa = EllipticalAnnulus(position,a_in=ra,a_out=ra+5,b_out=(ra+5)*b2aratio,theta=theta)
        aa = CircularAnnulus(position, r_in=ra, r_out=ra+5)

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
            ft = f0[cy-halfbox:cy+halfbox+1,cx-halfbox:cx+halfbox+1]
            position = (halfbox+dx,halfbox+dy)
            # if peak > peak_thres:
            aa = CircularAnnulus(position,r_in=ra,r_out=ra+5)
            # else:
                # aa = EllipticalAnnulus(position,a_in=ra,a_out=ra+5,b_out=(ra+5)*b2aratio,theta=theta)
            ftdat = sigma_clip(ft[ft>0],sigma=2.5,masked=False)
            med1 = np.median(ftdat)
            std1 = np.std(ftdat)
            stdmean1 = std1/np.sqrt(len(ftdat))
            thres = med1+threshold*stdmean1
            dat = aa.to_mask('center').multiply(ft)
            dat1 = sigma_clip(dat[dat>0],sigma=2.5,masked=False)
            dat1lower = np.median(dat1)-threshold*np.std(dat1)/np.sqrt(len(dat1))
        #ap=EllipticalAperture(position,r*a,r*b,theta*np.pi/180)
        # if peak > peak_thres:

        # convert xi, yi to ra/dec

        positions_x.append(xi[si])
        positions_y.append(yi[si])
        positions_ra.append(rai[si])
        positions_dec.append(deci[si])

        radius.append(ra)
        if make_fits == True:
            ap = CircularAperture(position,ra)
            am = ap.to_mask(method='center')
            mask = am.to_image((2*halfbox+1,2*halfbox+1))
            ft[mask>0] = fill_val0 - si
            f0p[cy-halfbox:cy+halfbox+1,cx-halfbox:cx+halfbox+1] = ft

        # else:
            # ap = EllipticalAperture(position,ra,rb,theta)
        # mlength = len(mask[mask>0])
        # mean1,med1,std1=sigma_clipped_stats(ft[ft>0],sigma=2.5)
        # ft[mask>0] = np.random.normal(med1,std1,mlength)

# f0p[outmask] = 0
if make_fits == True:
    hduI = fits.PrimaryHDU()
    hduI.data = f0p
    hdr['smsval'] = (fill_val0, 'Star Mask Starting filled Value')
    hdr['smeval'] = (fill_val0-si, 'Star Mask Ending filled Value')

    hduI.header = hdr
    hduI.writeto(pathfname.replace('.fits','.starmask.fits'), overwrite=True)

positions_x = np.array(positions_x)
positions_y = np.array(positions_y)
positions_ra = np.array(positions_ra)
positions_dec = np.array(positions_dec)

radius = np.array(radius)
# finaldata = [positions_x, positions_y, positions_ra, positions_dec, radius]

finaltab = Table()
finaltab.add_columns([positions_x,positions_y,positions_ra,positions_dec,radius], names=['x','y','ra','dec','radius'])
finaltab.write(path+clustername+'_starmask.csv',format='ascii.ecsv',overwrite=True)

# with open(path+clustername+'_starmask.pkl', 'wb') as p:
    # pickle.dump(finaldata, p)
