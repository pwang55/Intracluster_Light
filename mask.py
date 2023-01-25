"""

Usage:
    In script folder:
    $ python mask.py path_to_file/filename.fits (sig=3) (amt=5) (detect_thresh=2.0) (backsize=16) 
                                                (backfiltersize=1) (deblend_nthresh=64) (deblend_mincont=0.001) 
                                                (cleanparam=1.0) (ratio=3) (sexfilter=N)

    In data folder:
    $ python path_to_script/mask.py filename.fits (sig=3) (amt=5) (detect_thresh=2.0) 
                                                  (backsize=16) (backfiltersize=1) (deblend_nthresh=64) (deblend_mincont=0.001) 
                                                  (cleanparam=1.0) (ratio=3) (sexfilter=N)


This code takes input fits file, create an unsharp image and feed it to SExtractor to detect objects.
Then the code use the created catalog to mask all objects except bcg(s).

Unsharp image creates with equation:
    unsharp = original + amount * (original - gf(original, sigma))

Requires:
    filename.fits
    filename.wt.fits
Creates:
    filename.unsharp.fits
    filename.aper.fits
    filename.iclmask.cat
    filename.iclmask.fits

"""
import numpy as np
from astropy.io import fits, ascii
import subprocess
from scipy.ndimage import gaussian_filter as gf
import sys
from astropy.coordinates import SkyCoord
from photutils import EllipticalAperture

# default parameters
sig = 3
amt = 5

detect_thresh = 1.5
backsize = '16'
backfiltersize = '1'
deblend_nthresh = '64'
deblend_mincont = '0.001'
cleanparam = '1.0'
sexfilter = True
# filtername = 'gauss_6.0_13x13.conv'
filtername = 'gauss_4.0_7x7.conv'

ratio = 3   # ration of aperture to a, b determined by sextractor; the aperture to mask object will be ratio * a, ratio * b for semi-major and semi-minor axis

if len(sys.argv) < 2:
    print(__doc__)
    sys.exit()


pathfname = sys.argv[1]
fname = pathfname.split('/')[-1]
path = pathfname[:-len(fname)]

script_dir = sys.path[0] + '/'

args_dict = {}



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

            if arg_type == 'sig' or arg_type == 'sigma':
                sig = float(arg_val)
            elif arg_type == 'amt' or arg_type == 'amount':
                amt = float(arg_val)
            elif arg_type == 'detect_thresh':
                detect_thresh = arg_val
            elif arg_type == 'backsize':
                backsize = arg_val
            elif arg_type == 'backfiltersize':
                backfiltersize = arg_val
            elif arg_type == 'deblend_nthresh':
                deblend_nthresh = arg_val
            elif arg_type == 'deblend_mincont':
                deblend_mincont = arg_val
            # elif arg_type == 'sexfilter':
                # sexfilter = arg_val
            elif arg_type == 'cleanparam':
                cleanparam = arg_val
            elif arg_type == 'ratio':
                ratio = float(arg_val)
            elif arg_type == 'sexfilter':
                if arg_val == 'Y' or arg_val.lower() == 'true':
                    sexfilter = True
                elif arg_val == 'N' or arg_val.lower() == 'false':
                    sexfilter = False
            else:
                print('Unrecognized arguments!')
                print(__doc__)
                sys.exit()




print('Unsharp amount = {}'.format(amt))
print('Unsharp gaussian sigma = {}'.format(sig))
print('')
print('DETECT_THRESH = {}'.format(detect_thresh))
print('BACK_SIZE = {}'.format(backsize))
print('BACK_FILTERSIZE = {}'.format(backfiltersize))
print('DEBLEND_NTHRESH = {}'.format(deblend_nthresh))
print('DEBLEND_MINCONT = {}'.format(deblend_mincont))
print('CLEAN_PARAM = {}'.format(cleanparam))
print('SExtractor FILTER = {}'.format(sexfilter))
if sexfilter:
    print('\t Filter = {}'.format(filtername))
print('')
print('ratio = {}'.format(ratio))
print('')


# get cluster name
filenamesplit = fname.split('.')[0].split('_')
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

clustername = ''
# print(filenamesplit)
for i in range(len(filenamesplit)):
    if (filenamesplit[i] == all_cluster_names).any():
        clustername = filenamesplit[i]
if clustername == '':
    print("Cluster name not found!")
    sys.exit()

# get cluster bcg location from table for trimming
tab = ascii.read(script_dir + 'extra/bcg_table.ecsv')
tab = tab[tab['cluster'] == clustername]
bcg_ra = tab['ra']
bcg_dec = tab['dec']
cbcg = SkyCoord(bcg_ra, bcg_dec, unit='deg')


fname0 = fname.replace('.fits','')
wname = fname0 + '.wt.fits'
catname = fname0 + '.iclmask.cat'
unsharpname = fname0 + '.unsharp.fits'
checkimagename = fname0 + '.aper.fits'
maskname = fname0 + '.iclmask.fits'

f = fits.open(pathfname)
f0 = f[0].data
fhdr = f[0].header
f1 = f0 + amt * (f0 - gf(f0, sig))

hduI = fits.PrimaryHDU()
hduI.data = f1
hduI.header = fhdr
hduI.writeto(path + unsharpname, overwrite=True)


print('Running SExtractor to create catalog...')
if not sexfilter:
    subprocess.run(['sex', '-c', script_dir+'extra/icl_mask.sex', path+unsharpname, '-PARAMETERS_NAME', script_dir+'extra/icl_mask.param', \
        '-DEBLEND_NTHRESH', deblend_nthresh, '-DEBLEND_MINCONT', deblend_mincont, \
        '-WEIGHT_IMAGE', path+wname, '-BACK_SIZE', backsize, '-BACK_FILTERSIZE', backfiltersize, \
        '-CHECKIMAGE_NAME', path+checkimagename, '-CATALOG_NAME', path+catname, '-CLEAN_PARAM', cleanparam, '-VERBOSE_TYPE', 'QUIET'])
else:
    subprocess.run(['sex', '-c', script_dir+'extra/icl_mask.sex', path+unsharpname, '-PARAMETERS_NAME', script_dir+'extra/icl_mask.param', \
        '-DEBLEND_NTHRESH', deblend_nthresh, '-DEBLEND_MINCONT', deblend_mincont, '-FILTER', 'Y', '-FILTER_NAME', script_dir+'extra/'+filtername,\
        '-WEIGHT_IMAGE', path+wname, '-BACK_SIZE', backsize, '-BACK_FILTERSIZE', backfiltersize, \
        '-CHECKIMAGE_NAME', path+checkimagename, '-CATALOG_NAME', path+catname, '-CLEAN_PARAM', cleanparam, '-VERBOSE_TYPE', 'QUIET'])    


cat = ascii.read(path + catname)

ra = cat['ALPHA_J2000']
dec = cat['DELTA_J2000']
obja = cat['A_IMAGE']
objb = cat['B_IMAGE']
objtheta = cat['THETA_IMAGE'] * np.pi/180
objx = cat['X_IMAGE']-1
objy = cat['Y_IMAGE']-1
nobj = len(cat)

coords = SkyCoord(ra, dec, unit='deg')

idx, d2d, d3d = cbcg.match_to_catalog_sky(coords)

f2 = f0.copy()
print('Masking objects...')
for i in range(nobj):
    if (i != idx).all():
        a0 = obja[i]
        b0 = objb[i]
        x0 = objx[i]
        y0 = objy[i]
        theta0 = objtheta[i]

        ap = EllipticalAperture((x0, y0), a=ratio*a0, b=ratio*b0, theta=theta0)
        iap = ap.to_mask('center').to_image(f2.shape)
        f2[iap>0] = np.nan

hduI = fits.PrimaryHDU()
hduI.data = f2
hduI.header = fhdr
fhdr['s_sig'] = (sig, 'Sharpen sigma')
fhdr['s_amt'] = (amt, 'Sharpen amount')
fhdr['det_th'] = (detect_thresh, 'Detect threshold')
fhdr['bsize'] = (backsize, 'Back size for detecting objects')
fhdr['fsize'] = (backfiltersize, 'Back filter size for detecting objects')
fhdr['db_nth'] = (deblend_nthresh, 'Deblending nthresh')
fhdr['db_mct'] = (deblend_mincont, 'Deblending min contrast')
fhdr['clean'] = (cleanparam, 'Cleaning parameter')
fhdr['sexfilt'] = (sexfilter, 'SExtractor filter')
fhdr['ratio'] = (ratio, 'Aperture ratio')
hduI.writeto(path+maskname, overwrite=True)

