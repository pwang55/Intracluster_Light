"""

This code is outdated.

Usage:

    In script folder:
    $ python trim_for_icl.py path_to_file/filelist.txt
    or 
    $ python trim_for_icl.py path_to_file/filename.fits

    In data folder:
    $ path_to_script/trim_for_icl.py filelist.txt
    or
    $ path_to_script/trim_for_icl.py filename.fits
    
Requires:
master_stack_clustername_cutout.fits, wt.fits, _psf.psf
(or master_stack_clustername_cutout.trim.fits)
.wt.fits
.noobj.fits

This code takes a list of filenames or a single filename, find the master stack image or the master stack trim image,
reproject the image, noobj and wt image to the master stack trim image.

This code is intended to use on images before psf matching, but any image will do 
(it only change whatever the filename's extension .fits to .trim.fits, .noobj.fits to .trim.noobj.fits, .wt.fits to .trim.wt.fits)

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

# parameters
halfbox = 600
# default values for optional inputs

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

# Find master_stack fits files
master = 'master_stack_{}_cutout.fits'.format(clustername)
if Path(path + master).is_file() == False:
    print(master + ' cannot be found!')
    sys.exit()

masterwt = master.replace('.fits','.wt.fits')
m = fits.open(path + master)
m0 = m[0].data
mhdr = m[0].header
mwcs = WCS(mhdr)
finalhdr = mhdr.copy()  # copy a master stack hdr for later modification and used as header for all trimmed images

mwt = fits.open(path + masterwt)
mwt0 = mwt[0].data
masterpsfname = master.replace('.fits','_psf.psf')

mnoobj = fits.open(path + master.replace('.fits','.noobj.fits'))
mnoobj0 = mnoobj[0].data
mnoobjhdr = mnoobj[0].header
# mpsf = fits.open(path + masterpsfname)
# mpsf0 = mpsf[1].data[0][0][0]

mastertrimname = master.replace('.fits','.trim.fits')
# Make Cutout2D of master stack
if Path(path + mastertrimname).is_file() == False or Path(path + mastertrimname.replace('.wt.fits','.trim.wt.fits')).is_file() == False:
    print('Making a cutout of {}'.format(master))
    mcutout = Cutout2D(m0, cbcg, 2*halfbox+1, wcs=mwcs)
    mcutoutwt = Cutout2D(mwt0, cbcg, 2*halfbox+1, wcs=mwcs)
    mcutoutnoobj = Cutout2D(mnoobj0, cbcg, 2*halfbox+1, wcs=mwcs)

    crpix1 = mcutout.wcs.wcs.crpix[0]
    crpix2 = mcutout.wcs.wcs.crpix[1]
    naxis1 = 2*halfbox + 1
    naxis2 = naxis1

    finalhdr['crpix1'] = crpix1
    finalhdr['crpix2'] = crpix2
    finalhdr['naxis1'] = naxis1
    finalhdr['naxis2'] = naxis2

    hduM = fits.PrimaryHDU()
    hduM.data = mcutout.data
    hduM.header = finalhdr
    hduM.writeto(path + mastertrimname, overwrite=True)

    hduMwt = fits.PrimaryHDU()
    hduMwt.data = mcutoutwt.data
    hduMwt.header = finalhdr
    hduMwt.writeto(path + mastertrimname.replace('.fits','.wt.fits'), overwrite=True)

    hduMnoobj = fits.PrimaryHDU()
    hduMnoobj.data = mcutoutnoobj.data
    hduMnoobj.header = finalhdr
    hduMnoobj.writeto(path + mastertrimname.replace('.fits','.noobj.fits'), overwrite=True)

    mcutout0 = mcutout.data
    mcutoutwt0 = mcutoutwt.data


else:
    print(mastertrimname + 'already exists.')
    mcutout = fits.open(path + mastertrimname)
    mcutout0 = mcutout[0].data
    mcutoutwt0 = fits.getdata(path + mastertrimname.replace('.fits','.wt.fits'))
    finalhdr = mcutout[0].header


for i in range(nf):
    filename = files[i]
    noobjfilename = filename.replace('.fits','.noobj.fits')
    wtfilename = filename.replace('.fits','.wt.fits')

    # if the file is a master cutout, skip
    if filename.split('_')[0] == 'master':        
        pass
    else:

        # psfname = filename.replace('.fits','_psf.psf')
        f = fits.open(path + filename)
        n = fits.open(path + noobjfilename)
        wt = fits.open(path + wtfilename)
        # psf = fits.open(path + psfname)

        print('Reproject image to master cutout...')
        f_reproj = reproject_exact(f, finalhdr, parallel=False, return_footprint=False)
        print('Reproject noobj image to master cutout...')
        noobj_reproj = reproject_exact(n, finalhdr, parallel=False, return_footprint=False)
        print('Reproject wt image to master cutout...')
        wt_reproj = reproject_exact(wt, finalhdr, parallel=False, return_footprint=False)

        hduI = fits.PrimaryHDU()
        hduI.data = f_reproj
        hduI.header = finalhdr
        hduI.writeto(path + filename.replace('.fits','.trim.fits'), overwrite=True)

        hduI = fits.PrimaryHDU()
        hduI.data = noobj_reproj
        hduI.header = finalhdr
        hduI.writeto(path + noobjfilename.replace('.noobj.fits','.trim.noobj.fits'), overwrite=True)

        hduI = fits.PrimaryHDU()
        hduI.data = wt_reproj
        hduI.header = finalhdr
        hduI.writeto(path + wtfilename.replace('.wt.fits','.trim.wt.fits'), overwrite=True)

