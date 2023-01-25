"""

This code is outdated.

Usage:

    In script folder:
    $ python model_gradient_match_psf.py path_to_files/trimlist.txt (sigma=3.0) (alpha=0.5) (model=true)
    or 
    $ python model_gradient_match_psf.py path_to_files/filename.trim.fits (sigma=3.0) (alpha=0.5) (model=true)
    
    In data folder:
    $ python path_to_script/model_gradient_match_psf.py trimlist.txt (sigma=3.0) (alpha=0.5) (model=true)
    or 
    $ python path_to_script/model_gradient_match_psf.py filename.trim.fits (sigma=3.0) (alpha=0.5) (model=true)

This code takes .trim.fits (images that are trimmed and reprojected), use .trim.noobj.fits to estimate a background gradient model,
subtract from image and then match psf to a target gaussian.

If setting model=false, this code will not try to fit a model, instead will read it directly from .trim.bkgblock.fits if it is available.

Requires:
    .trim.fits
    .trim.wt.fits
    .trim.noobj.fits
    _psf.psf

Creates:
    .trim.convolved.fits
    .trim.convolved.wt.fits
    .trim.bkgblock.fits

.trim.bkgblock.fits contains 4 extensions:
0   Primary HDU with just zeros
1   fitted model
2   noobj - model
3   z0 - model (z0: smoothed background from photutils)

"""
import numpy as np
from astropy.io import fits, ascii
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.modeling import models, fitting
from astropy.stats import SigmaClip, sigma_clip, sigma_clipped_stats
from photutils import create_matching_kernel, TukeyWindow, TopHatWindow, SplitCosineBellWindow
from astropy.convolution import convolve
from astropy.modeling.models import Gaussian2D
from photutils import MedianBackground, Background2D
from photutils import CircularAperture
import sys
from pathlib import Path
from glob import glob

# default values for optional inputs
gradient_deg = 1    # order of the background gradient model, 1 -> linear, 2 -> quadratic
gaussian_sig = 3.0  # default target gaussian sigma
alpha = 0.5  # default Tukey window alpha
model_background = True
nb = 5  # number of iterations for estimating background from noobj
bsize = 25
fsize = 5
bkg_estimator = MedianBackground()

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
            elif arg_type == 'model':
                if arg_val == 'true':
                    model_background = True
                elif arg_val == 'false':
                    model_background = False

            else:
                print('Unrecognized arguments!')
                print(__doc__)
                sys.exit()

if len(sys.argv) < 2:
    print(__doc__)
    sys.exit()

print('gaussian sigma = {}'.format(gaussian_sig))
print('Tukey Window alpha = {}'.format(alpha))
print('\n')

psffile = glob('*_psf.psf')[0]
psf = fits.open(psffile)[1].data[0][0][0]
psfgrid = psf.shape[0]
window = TukeyWindow(alpha=alpha)

# create target gaussian
y, x = np.mgrid[0:psfgrid, 0:psfgrid]
gm1 = Gaussian2D(100, (psfgrid-1)/2, (psfgrid)/2, gaussian_sig, gaussian_sig)
gauss = gm1(x, y)
gauss /= gauss.sum()

pathfname = sys.argv[1]
fname = pathfname.split('/')[-1]
path = pathfname[:-len(fname)]

files = []
# determine if input is a single file or a list of files
# in both case, files only contains the file names, not paths
fname_extention = fname.split('.')[-1]
if fname_extention == 'fits':
    files = [fname]
elif fname_extention == 'txt':
    with open(pathfname) as f:
        for l in f:
            files.append(l.strip())


for i in range(len(files)):
    filename = files[i]
    # make sure filename is .trim.fits
    filenamesplit = filename.split('.')
    if filenamesplit[-1] != 'fits' and filenamesplit[-2] != 'trim':
        print('Input files need to be .trim.fits!')
        sys.exit()

    print('File:\t' + filename)

    f = fits.open(path + filename)
    p = fits.open(path + filename.replace('.trim.fits','_psf.psf'))
    n = fits.open(path + filename.replace('.fits','.noobj.fits'))
    w = fits.open(path + filename.replace('.fits','.wt.fits'))
    f0 = f[0].data
    p0 = p[1].data[0][0][0]
    fhdr = f[0].header
    n0 = n[0].data
    w0 = w[0].data
    whdr = w[0].header

    if model_background == True:
        # first estimate smooth background from n0
        z0 = n0.copy()
        for j in range(nb):
            bkg = Background2D(z0, (bsize, bsize), filter_size=(fsize, fsize), sigma_clip=SigmaClip(sigma=3.), bkg_estimator=bkg_estimator)
            z0 = bkg.background

        fit_p = fitting.LevMarLSQFitter()
        p_init = models.Polynomial2D(degree=gradient_deg)

        xp = np.arange(len(f0))
        yp = np.arange(len(f0))
        xp, yp = np.meshgrid(xp, yp)

        print('Fitting model to background, deg={}'.format(gradient_deg))
        model = fit_p(p_init, xp, yp, z0)(xp, yp)
        # model = np.zeros(n0.shape)
        print('Done.')
    
    elif model_background == False:
        if Path(path + filename.replace('.fits','.bkgblock.fits')).is_file() == False:
            print('model=false but .bkgblock.fits not found!')
            sys.exit()
        else:
            bkgblock = fits.open(path + filename.replace('.fits','.bkgblock.fits'))
            model = bkgblock[1].data

    #modelmin = np.min(model)
#    model = model - modelmin
    f0 = f0 - model

    # match psf
    kernel = create_matching_kernel(p0, gauss, window)
    fhdr['tgsig'] = (gaussian_sig, 'Matched target gaussian sigma')
    whdr['tgsig'] = (gaussian_sig, 'Matched target gaussian sigma')
    fhdr['win_a'] = (alpha, 'Tukey window alpha')
    whdr['win_a'] = (alpha, 'Tukey window alpha')

    # save original psf, kernel and convolved psf to a psfs.fits file
    hduP = fits.PrimaryHDU()
    hduP.data = np.zeros(p0.shape)
    hduP.header['D_TYPE'] = ('zero','Data type for psf information')
    hduP1 = fits.ImageHDU()
    hduP1.data = p0
    hduP1.header['D_TYPE'] = ('original','Data type for psf information')
    hduP2 = fits.ImageHDU()
    hduP2.data = kernel
    hduP2.header['D_TYPE'] = ('kernel','Data type for psf information')
    hduP3 = fits.ImageHDU()
    hduP3.data = convolve(p0, kernel)
    hduP3.header['D_TYPE'] = ('convolved','Data type for psf information')
    hduPA = fits.HDUList([hduP, hduP1, hduP2, hduP3])
    hduPA.writeto(path + filename.replace('.fits','.psfs.fits'))

    f1 = convolve(f0, kernel)
    w1 = convolve(w0, kernel)
    hduI = fits.PrimaryHDU()
    hduI.data = f1
    hduI.header = fhdr
    hduI.writeto(path + filename.replace('.fits','.convolved.fits'),overwrite=True)

    hduW = fits.PrimaryHDU()
    hduW.data = w1
    hduW.header = whdr
    hduW.writeto(path + filename.replace('.fits','.convolved.wt.fits'),overwrite=True)

    if model_background == True:
        # save all background related info into an image block
        hduB = fits.PrimaryHDU()
        hduB.data = np.zeros(f1.shape)
        # hduB.header = fhdr
        hduB.header['D_TYPE'] = ('zero','Data type for estimatinb background')
        # hduB.header['EXTEND'] = True

        hduB1 = fits.ImageHDU()
        hduB1.data = model
        # hduB1.header = fhdr
        hduB1.header['D_TYPE'] = ('model','Data type for estimatinb background')
        # hduB1.header['XTENSION'] = ('IMAGE', 'IMAGE extension')

        hduB2 = fits.ImageHDU()
        hduB2.data = n0 - model
        # hduB2.header = fhdr
        hduB2.header['D_TYPE'] = ('noobj - model','Data type for estimatinb background')
        # hduB2.header['XTENSION'] = ('IMAGE', 'IMAGE extension')

        hduB3 = fits.ImageHDU()
        hduB3.data = z0 - model
        # hduB3.header = fhdr
        hduB3.header['D_TYPE'] = ('z0 - model','Data type for estimatinb background')
        # hduB3.header['XTENSION'] = ('IMAGE', 'IMAGE extension')

        hduA = fits.HDUList([hduB, hduB1, hduB2, hduB3])
        hduA.writeto(path + filename.replace('.fits','.bkgblock.fits'), overwrite=True)

    print('\n')

