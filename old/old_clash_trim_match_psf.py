"""

Usage:

    In script folder:
    $ python trim_for_icl.py path_to_file/clashlist.txt (path/master_stack_clustername_cutout.trim.convolved.fits)
    or
    $ python trim_for_icl.py path_to_file/clash_subaru_abell611_rc.fits (path/master_stack_clustername_cutout.trim.convolved.fits)

    In data folder:
    $ path_to_script/trim_for_icl.py clashlist.txt (path/master_stack_clustername_cutout.trim.convolved.fits)
    or
    $ path_to_script/trim_for_icl.py clash_subaru_abell611_rc.fits (path/master_stack_clustername_cutout.trim.convolved.fits)
    
Requires:
master_stack_clustername_cutout.trim.convolved.fits or path to it as second argument

This code takes clash image, reproject it to the master trim image and match psf to a target gaussian.
Creates .trim.convolved.fits and .trim.convolved.wt.fits

Note:
1. If the path and filename to the master trim convolved fits file is not provided as second argument, the script looks for the file in the current data folder.
2. If _psf.psf does not exist, the script will call clash_make_psf.sh to create it.


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
# default values for psf matching
# gaussian_sig = 3.0  # default target gaussian sigma
# alpha = 0.5  # default Tukey window alpha

# print('gaussian sigma = {}'.format(gaussian_sig))
# print('Tukey Window alpha = {}'.format(alpha))
# print('\n')

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

if len(sys.argv) == 3:
    pathmaster = sys.argv[2]
    if Path(pathmaster).is_file() == True:
        mcutout = fits.open(pathmaster)
        mcutout0 = mcutout[0].data
        finalhdr = mcutout[0].header
        gaussian_sig = finalhdr['TGSIG']
        alpha = finalhdr['WIN_A']
    else:
        print('{}\nnot found!'.format(pathmaster))
        sys.exit()
elif len(sys.argv) == 2:
    mastertrimname = 'master_stack_{}_cutout.trim.convolved.fits'.format(clustername)
    if Path(path + mastertrimname).is_file() == False:
        print('{} not found!'.format(mastertrimname))
        sys.exit()
    else:
        mcutout = fits.open(path + mastertrimname)
        mcutout0 = mcutout[0].data
        finalhdr = mcutout[0].header        
        gaussian_sig = finalhdr['TGSIG']
        alpha = finalhdr['WIN_A']


for i in range(nf):
    filename = files[i]
    wtfilename = filename.replace('.fits','.wt.fits')

    # psfname = filename.replace('.fits','_psf.psf')
    f = fits.open(path + filename)
    wt = fits.open(path + wtfilename)
    # fhdr = f[0].header
    # whdr = wt[0].header
    f0 = f[0].data
    w0 = wt[0].data

    print('Reproject image to master cutout...')
    f_reproj = reproject_exact(f, finalhdr, parallel=False, return_footprint=False)
    print('Reproject wt image to master cutout...')
    wt_reproj = reproject_exact(wt, finalhdr, parallel=False, return_footprint=False)

    # TESTING TEMP
    hduI = fits.PrimaryHDU()
    hduI.data = f_reproj
    hduI.header = finalhdr
    hduI.writeto(path + filename.replace('.fits','.trim.fits'), overwrite=True)

    hduI = fits.PrimaryHDU()
    hduI.data = wt_reproj
    hduI.header = finalhdr
    hduI.writeto(path + wtfilename.replace('.wt.fits','.trim.wt.fits'), overwrite=True)

    psffilename = filename.replace('.fits','_psf.psf')
    # If psffilename does not exist, create one by calling clash_make_psf.sh:
    if Path(path + psffilename).is_file() == False:
        print("{} doesn't exist, calling clash_make_psf.sh...".format(psffilename))
        subprocess.run(['/Users/brianwang76/sources/90Prime/ICL/clash_make_psf.sh', path+filename])
    psf = fits.open(path + psffilename)[1].data[0][0][0]
    psfgrid = psf.shape[0]
    window = TukeyWindow(alpha=alpha)

    # create target gaussian
    y, x = np.mgrid[0:psfgrid, 0:psfgrid]
    gm1 = Gaussian2D(100, (psfgrid-1)/2, (psfgrid)/2, gaussian_sig, gaussian_sig)
    gauss = gm1(x, y)
    gauss /= gauss.sum()

    hdr = finalhdr.copy()

    kernel = create_matching_kernel(psf, gauss, window)
    # hdr['tgsig'] = (gaussian_sig, 'Matched target gaussian sigma')
    # hdr['win_a'] = (alpha, 'Tukey window alpha')

    hduP = fits.PrimaryHDU()
    hduP.data = np.zeros(psf.shape)
    hduP.header['D_TYPE'] = ('zero','Data type for psf information')
    hduP1 = fits.ImageHDU()
    hduP1.data = psf
    hduP1.header['D_TYPE'] = ('original','Data type for psf information')
    hduP2 = fits.ImageHDU()
    hduP2.data = kernel
    hduP2.header['D_TYPE'] = ('kernel','Data type for psf information')
    hduP3 = fits.ImageHDU()
    hduP3.data = convolve(psf, kernel)
    hduP3.header['D_TYPE'] = ('convolved','Data type for psf information')
    hduPA = fits.HDUList([hduP, hduP1, hduP2, hduP3])
    hduPA.writeto(path + filename.replace('.fits','.psfs.fits'), overwrite=True)

    f1 = convolve(f_reproj, kernel)
    w1 = convolve(wt_reproj, kernel)
    hduI = fits.PrimaryHDU()
    hduI.data = f1
    hduI.header = hdr
    hduI.writeto(path + filename.replace('.fits','.trim.convolved.fits'),overwrite=True)

    hduW = fits.PrimaryHDU()
    hduW.data = w1
    hduW.header = hdr
    hduW.writeto(path + filename.replace('.fits','.trim.convolved.wt.fits'),overwrite=True)

