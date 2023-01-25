"""

No longer used.

Usage:

    In script folder:
    $ python bkgblock.py path_to_file/filelist.txt
    or
    $ python bkgblock.py path_to_file/filename.fits

    In data folder:
    $ python path_to_script/bkgblock.py filelist.txt
    or
    $ python path_to_script/bkgblock.py filename.fits

Requires:
    master_stack_clustername_cutout.trimconv.fits (or master_stack_clustername_cutout.fits)

This code takes input file, use master_stack_clustername_cutout.trimconv.fits header as target, 
reproject the filename.noobj.fits to master trimconv, then fit the model background, save it as filename.bkgblock.fits
If filename.noobj.fits is not availabe, this code will call remove_object.py.


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
from photutils import MedianBackground, Background2D
from photutils import CircularAperture
import sys
from pathlib import Path
from astropy.nddata import Cutout2D
import subprocess
from photutils import create_matching_kernel, TukeyWindow, TopHatWindow, SplitCosineBellWindow
from astropy.convolution import convolve
from astropy.modeling.models import Gaussian2D
from reproject import reproject_exact
from glob import glob

# parameters
halfbox = 600
gradient_deg = 1    # order of the background gradient model, 1 -> linear, 2 -> quadratic
model_background = True
nb = 5  # number of iterations for estimating background from noobj
bsize = 25
fsize = 5
bkg_estimator = MedianBackground()

script_dir = sys.path[0] + '/'

if len(sys.argv) < 2:
    print(__doc__)
    sys.exit()

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

nf = len(files)
# get cluster name
filenamesplit = files[0].split('_')
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
for i in range(len(filenamesplit)):
    if (filenamesplit[i] == all_cluster_names).any():
        clustername = filenamesplit[i]
if clustername == '':
    print("Cluster name not found!")
    sys.exit()

# get cluster bcg location from table for trimming
tab = ascii.read(script_dir + 'extra/bcg_table.ecsv')
tab = tab[tab['cluster'] == clustername]
bcg_ra = np.mean(tab['ra'])
bcg_dec = np.mean(tab['dec'])
cbcg = SkyCoord(bcg_ra, bcg_dec, unit='deg')


master = 'master_stack_{}_cutout.fits'.format(clustername)
mastertrimconv = master.replace('.fits','.trimconv.fits')
# masterwt = master.replace('.fits','.wt.fits')
# masterpsfname = master.replace('.fits','_psf.psf')

# First search for mastertrimconv
if Path(path + mastertrimconv).is_file() == True:
    m = fits.open(path + mastertrimconv)
    mhdr = m[0].header
    finalhdr = mhdr.copy()
elif Path(path + master).is_file() == True:
    m = fits.open(path + master)
    m0 = m[0].data
    mhdr = m[0].header
    finalhdr = mhdr.copy()
    mwcs = WCS(mhdr)
    mcutout = Cutout2D(m0, cbcg, 2*halfbox+1, wcs=mwcs)

    crpix1 = mcutout.wcs.wcs.crpix[0]
    crpix2 = mcutout.wcs.wcs.crpix[1]
    naxis1 = 2*halfbox + 1
    naxis2 = naxis1

    finalhdr['crpix1'] = crpix1
    finalhdr['crpix2'] = crpix2
    finalhdr['naxis1'] = naxis1
    finalhdr['naxis2'] = naxis2
else:
    print("Can't find either master cutout or master cutout trim.conv!")
    sys.exit()

# for each input file, reproject the noobj file to master header (finalhdr), then calculate the background model
for i in range(nf):
    filename = files[i]
    noobjfilename = filename.replace('.fits','.noobj.fits')
    print('\nFile: {}'.format(filename))

    if Path(path + noobjfilename).is_file() == False:
        home = str(Path.home())
        print('Run remove_object.py...')
        subprocess.run(['python', home+'/sources/90Prime/ICL/remove_objects.py', path+filename])

    # if the file is a master cutout, skip
    if filename.split('_')[0] == 'master':        
        pass
    else:
        n = fits.open(path + noobjfilename)
        # n0 = n[0].data
        print('Reproject {} to master cutout trim header...'.format(noobjfilename))
        noobj_reproj = reproject_exact(n, finalhdr, parallel=False, return_footprint=False)
        print('Reprojection done.')

        z0 = noobj_reproj.copy()
        for j in range(nb):
            bkg = Background2D(z0, (bsize, bsize), filter_size=(fsize, fsize), sigma_clip=SigmaClip(sigma=3.), bkg_estimator=bkg_estimator)
            z0 = bkg.background

        fit_p = fitting.LevMarLSQFitter()
        p_init = models.Polynomial2D(degree=gradient_deg)

        xp = np.arange(2*halfbox+1)
        yp = np.arange(2*halfbox+1)
        xp, yp = np.meshgrid(xp, yp)

        print('Fitting model to background, deg={}'.format(gradient_deg))
        model = fit_p(p_init, xp, yp, z0)(xp, yp)
        # model = np.zeros(n0.shape)
        print('Done.')

        # save all background related info into an image block
        hduB = fits.PrimaryHDU()
        hduB.data = np.zeros(noobj_reproj.shape)
        # hduB.header = fhdr
        hduB.header['D_TYPE'] = ('zero','Data type for estimatinb background')
        # hduB.header['EXTEND'] = True

        hduB1 = fits.ImageHDU()
        hduB1.data = model
        # hduB1.header = fhdr
        hduB1.header['D_TYPE'] = ('model','Data type for estimatinb background')
        # hduB1.header['XTENSION'] = ('IMAGE', 'IMAGE extension')

        hduB2 = fits.ImageHDU()
        hduB2.data = noobj_reproj - model
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

