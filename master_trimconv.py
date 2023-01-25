"""

Usage:

    In script folder:
    $ python master_trimconv.py path_to_file/master_stack_clustername_cutout.fits (sigma=2.5) (alpha=0.5) (removestars=true) (halfbox=600)

    In file folder:
    $ python path_to_script/master_trimconv.py master_stack_clustername_cutout.fits (sigma=2.5) (alpha=0.5) (removestars=true) (halfbox=600)

Requires:
    master_stack_clustername_cutout.fits and .wt.fits
    clustername_starmask.csv
Creates:
    master_stack_clustername_cutout.trim.fits and .trim.wt.fits
    master_stack_clustername_cutout.trimconv.fits and .trimconv.wt.fits

This code takes master cutout file as input, make a cutout, match psf to target gaussian.

If argument "removestars" is true, this code will read clustername_starmask.csv to mask & fill stars,
then make the updated .trimconv.fits.


"""
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
import subprocess

cutouthalfbox = 600
gaussian_sig = 2.5  # default target gaussian sigma
alpha = 0.5  # default Tukey window alpha

# distance outside star aperture to calculate nearby median & std
dsr = 5
removestars = True

script_dir = sys.path[0] + '/'

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
            if arg_type == 'sigma':
                gaussian_sig = float(arg_val)
            elif arg_type == 'alpha':
                alpha = float(arg_val)
            elif arg_type == 'removestars':
                if arg_val.lower() == 'true':
                    removestars = True
                elif arg_val.lower() == 'false':
                    removestars = False
            elif arg_type == 'halfbox':
                cutouthalfbox = int(arg_val)
            else:
                print('Unrecognized arguments!')
                print(__doc__)
                sys.exit()

if len(sys.argv) < 2:
    print(__doc__)
    sys.exit()

print('gaussian sigma = {}'.format(gaussian_sig))
print('Tukey Window alpha = {}'.format(alpha))
print('Halfbox = {}'.format(cutouthalfbox))
print('\n')

pathfname = sys.argv[1]
master = pathfname.split('/')[-1]
path = pathfname[:-len(master)]

if master.split('_')[0] != 'master':
    print('Input file not master stack')
    sys.exit()

fnamesplit = master.split('_')

clustername = fnamesplit[2]

# get cluster bcg location from table for trimming
tab = ascii.read(script_dir + 'extra/bcg_table.ecsv')
tab = tab[tab['cluster'] == clustername]
bcg_ra = np.mean(tab['ra'])
bcg_dec = np.mean(tab['dec'])
cbcg = SkyCoord(bcg_ra, bcg_dec, unit='deg')

# # Find the stars catalog from either SDSS or panstarrs
# if Path(path + clustername+'_star_sdss_radec.csv').is_file():
#     stars = ascii.read(path + clustername+'_star_sdss_radec.csv')
# elif Path(path + clustername+'_star_panstarrs_radec.csv'):
#     stars = ascii.read(path + clustername+'_star_panstarrs_radec.csv')
# else:
#     print('Cannot find stars catalog!')
#     sys.exit()
# coord_stars = SkyCoord(stars['ra'], stars['dec'], unit='deg')


if removestars == True:
    # find starmask.csv
    if Path(path+clustername+'_starmask.csv').is_file() == True:
        # with open(path+clustername+'_starmask.pkl', 'rb') as pk:
        #     starmasks = pickle.load(pk)
        starmasks = ascii.read(path+clustername+'_starmask.csv')
        # starmasks_x = starmasks['x']
        # starmasks_y = starmasks['y']
        starmasks_ra = starmasks['ra']
        starmasks_dec = starmasks['dec']
        starmasks_r = starmasks['radius']
    else:
        print(clustername+'_starmask.csv not available!')
        sys.exit()




masterwt = master.replace('.fits','.wt.fits')
masterpsfname = master.replace('.fits','_psf.psf')

home = str(Path.home())

# if masterpsfname is not available, call make_psf.sh
if Path(path + masterpsfname).is_file() == False:
    print('Call make_psf.sh...')
    subprocess.run([home+'/sources/90Prime/ICL/make_psf.sh', path+master])


m = fits.open(path + master)
m0 = m[0].data
m0p = m0.copy()
outmask = m0p == 0
mhdr = m[0].header
mwcs = WCS(mhdr)
mwt = fits.open(path + masterwt)
mwt0 = mwt[0].data
mpsf = fits.open(path + masterpsfname)
mpsf0 = mpsf[1].data[0][0][0]
psfgrid = mpsf0.shape[0]
window = TukeyWindow(alpha=alpha)

# create target gaussian
y, x = np.mgrid[0:psfgrid, 0:psfgrid]
gm1 = Gaussian2D(100, (psfgrid-1)/2, (psfgrid)/2, gaussian_sig, gaussian_sig)
gauss = gm1(x, y)
gauss /= gauss.sum()


finalhdr = mhdr.copy()  # copy a master stack hdr for later modification and used as header for all trimmed images

m0 = m0p

print('Making a cutout of {}'.format(master))
mcutout = Cutout2D(m0p, cbcg, 2*cutouthalfbox+1, wcs=mwcs)
mcutoutwt = Cutout2D(mwt0, cbcg, 2*cutouthalfbox+1, wcs=mwcs)

crpix1 = mcutout.wcs.wcs.crpix[0]
crpix2 = mcutout.wcs.wcs.crpix[1]
naxis1 = 2*cutouthalfbox + 1
naxis2 = naxis1

finalhdr['crpix1'] = crpix1
finalhdr['crpix2'] = crpix2
finalhdr['naxis1'] = naxis1
finalhdr['naxis2'] = naxis2

mcutout0 = mcutout.data
mcutoutwt0 = mcutoutwt.data


print('Match cutout to target gaussian...')

kernel = create_matching_kernel(mpsf0, gauss, window)
finalhdr['tgsig'] = (gaussian_sig, 'Matched target gaussian sigma')
finalhdr['win_a'] = (alpha, 'Tukey window alpha')

# save original psf, kernel and convolved psf to a psfs.fits file
hduP = fits.PrimaryHDU()
hduP.data = np.zeros(mpsf0.shape)
hduP.header['D_TYPE'] = ('zero','Data type for psf information')
hduP.header['tgsig'] = (gaussian_sig, 'Matched target gaussian sigma')
hduP.header['win_a'] = (alpha, 'Tukey window alpha')
hduP1 = fits.ImageHDU()
hduP1.data = mpsf0
hduP1.header['D_TYPE'] = ('original','Data type for psf information')
hduP2 = fits.ImageHDU()
hduP2.data = kernel
hduP2.header['D_TYPE'] = ('kernel','Data type for psf information')
hduP3 = fits.ImageHDU()
hduP3.data = convolve(mpsf0, kernel)
hduP3.header['D_TYPE'] = ('convolved','Data type for psf information')
hduPA = fits.HDUList([hduP, hduP1, hduP2, hduP3])
hduPA.writeto(path + master.replace('.fits','.psfs.fits'), overwrite=True)

mcutout0p = mcutout0.copy()

# mask stars and fill value with nearby annulus
print('Removing and filling stars with mask...')
fwcs = WCS(finalhdr)
xbcg, ybcg = cbcg.to_pixel(fwcs)
starmasks_x, starmasks_y = SkyCoord(starmasks_ra, starmasks_dec, unit='deg').to_pixel(fwcs)
hbcg = (np.abs(starmasks_x - xbcg) < cutouthalfbox) & (np.abs(starmasks_y - ybcg) < cutouthalfbox)
starmasks_x = starmasks_x[hbcg]
starmasks_y = starmasks_y[hbcg]
starmasks_ra = starmasks_ra[hbcg]
starmasks_dec = starmasks_dec[hbcg]
starmasks_r = starmasks_r[hbcg]
for j in range(len(starmasks_x)):
    # px = starmasks_x[j]
    # py = starmasks_y[j]
    coord1 = SkyCoord(starmasks_ra[j], starmasks_dec[j], unit='deg')
    px, py = coord1.to_pixel(fwcs)
    sr = starmasks_r[j]
    ap = CircularAperture((px, py), r=sr)
    aa = CircularAnnulus((px, py), r_in=sr, r_out=sr+dsr)
    m_ap = ap.to_mask('center')
    m_aa = aa.to_mask('center')
    i_ap = m_ap.to_image(mcutout0p.shape)
    i_aa = m_aa.to_image(mcutout0p.shape)
    dat = mcutout0p[i_aa>0]
    #dat1 = sigma_clip(dat, masked=False)
    #med1 = np.median(dat1)
    #std1 = np.std(dat1)
    mean1, med1, std1 = sigma_clipped_stats(dat)
    mcutout0p[i_ap>0] = np.random.normal(med1, std1, len(i_ap[i_ap>0]))



hduI = fits.PrimaryHDU()
hduI.data = mcutout0p
hduI.header = finalhdr
hduI.writeto(path + master.replace('.fits','.trim.fits'), overwrite=True)

hduW = fits.PrimaryHDU()
hduW.data = mcutoutwt0
hduW.header = finalhdr
hduW.writeto(path + master.replace('.fits','.trim.wt.fits'), overwrite=True)

mcutout1 = convolve(mcutout0p, kernel)
mcutoutwt1 = convolve(mcutoutwt0, kernel)

hduI = fits.PrimaryHDU()
hduI.data = mcutout1
hduI.header = finalhdr
hduI.writeto(path + master.replace('.fits','.trimconv.fits'), overwrite=True)

hduW = fits.PrimaryHDU()
hduW.data = mcutoutwt1
hduW.header = finalhdr
hduW.writeto(path + master.replace('.fits','.trimconv.wt.fits'), overwrite=True)


