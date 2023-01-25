"""

Usage:

    In script folder:
    $ python trimconv.py path_to_file/filelist.txt (sigma=2.5) (alpha=0.5) (conv=false) (sexbackground=false) (removestars=false) (bsize=128) (fsize=3)
    or
    $ python trimconv.py path_to_file/filename.fits (sigma=2.5) (alpha=0.5) (conv=false) (sexbackground=false) (removestars=false) (bsize=128) (fsize=3)

    In data folder:
    $ python path_to_script/trimconv.py filelist.txt (sigma=2.5) (alpha=0.5) (conv=false) (sexbackground=false) (removestars=false) (bsize=128) (fsize=3)
    or
    $ python path_to_script/trimconv.py filename.fits (sigma=2.5) (alpha=0.5) (conv=false) (sexbackground=false) (removestars=false) (bsize=128) (fsize=3)

Requires:
    master_stack_clustername_cutout.trimconv.fits (or master_stack_clustername_cutout.fits)
    clustername_starmask.csv
    filename.fits
    filename.wt.fits
        (filename_psf.psf) will be created if not available

Creates:
    filename.trim.fits
    filename.trim.wt.fits
If conv=true, then creates
    filename.trimconv.fits
    filename.trimconv.wt.fits
    filename.psfs.fits

This code is mainly used for trimming and reprojecting cutout stack images. By default convolving and removing stars, removing backgrounds are all false.

If master_stack_clustername_cutout.trimconv.fits is available, the code reads sigma and alpha from header and overrides the optional input arguments even if provided.
If sexbackground=true, code will use photutils.Background2D to estimate a background and subtract from the image before convolving with kernel.
sigma, alpha are used only if conv=true.
Arguments "bsize" and "fsize" are only used when sexbackground=true.

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
from photutils.background import Background2D, MedianBackground, SExtractorBackground
from astropy.stats import SigmaClip, sigma_clip, sigma_clipped_stats
from astropy.convolution import convolve
from glob import glob
from reproject import reproject_exact
import multiprocessing as mp
import subprocess
import pickle


mp.set_start_method('fork')

halfbox = 600
gaussian_sig = 2.5  # default target gaussian sigma
alpha = 0.5  # default Tukey window alpha
window = TukeyWindow(alpha=alpha)
parallelTF = True
conv = False
cutoutbox = 1401    # temporary cutout before reprojection
# parameters for background estimation and subtraction
sexbackground = False
removestars = False
bsize = 80
fsize = 3


dsr = 5 # distance outside star aperture to calculate nearby median & std
nrmsfac = 20    # number of times to calculate the factor between original rms and kernel convolved rms
npool = 5   # number of pools for mp.Pool

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
            elif arg_type == 'conv':
                if arg_val.lower() == 'true':
                    conv = True
                elif arg_val.lower() == 'false':
                    conv = False
            elif arg_type == 'sexbackground':
                if arg_val.lower() == 'true':
                    sexbackground = True
                elif arg_val.lower() == 'false':
                    sexbackground = False
            elif arg_type == 'removestars':
                if arg_val.lower() == 'true':
                    removestars = True
                elif arg_val.lower() == 'false':
                    removestars = False
            elif arg_type == 'bsize':
                bsize = float(arg_val)
            elif arg_type == 'fsize':
                fsize = float(arg_val)
            else:
                print('Unrecognized arguments!')
                print(__doc__)
                sys.exit()

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
# print(nf)
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

master = 'master_stack_{}_cutout.fits'.format(clustername)
mastertrimconv = master.replace('.fits','.trimconv.fits')


# First search for mastertrimconv for the finalhdr
if Path(path + mastertrimconv).is_file():
    m = fits.open(path + mastertrimconv)
    mhdr = m[0].header
    gaussian_sig = mhdr['tgsig']
    alpha = mhdr['win_a']
    finalhdr = mhdr.copy()
    halfbox = int((finalhdr['naxis1']-1)/2)
    cutoutbox = halfbox*2 + 200
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
    finalhdr['tgsig'] = (gaussian_sig, 'Matched target gaussian sigma')
    finalhdr['win_a'] = (alpha, 'Tukey window alpha')

else:
    print("Can't find either master cutout or master cutout trim.conv!")
    sys.exit()

print('Halfbox = {}'.format(halfbox))
if conv:
    print('gaussian sigma = {}'.format(gaussian_sig))
    print('Tukey Window alpha = {}'.format(alpha))
if sexbackground:
    print('bsize = {}, fsize = {}'.format(bsize, fsize))
    

# psffile = glob('*_psf.psf')[0]
# psf = fits.open(psffile)[1].data[0][0][0]
# psfgrid = psf.shape[0]
psfgrid = 25
# create target gaussian
y, x = np.mgrid[0:psfgrid, 0:psfgrid]
gm1 = Gaussian2D(100, (psfgrid-1)/2, (psfgrid)/2, gaussian_sig, gaussian_sig)
gauss = gm1(x, y)
gauss /= gauss.sum()

# define function for figuring out the factor between original rms and kernel convolved rms
def func_rmsfac(x, kernel):
    np.random.seed(x)
    im0 = np.random.normal(0, 1, (2*halfbox+1,2*halfbox+1))
    im1 = convolve(im0, kernel)
    return(np.std(im0)/np.std(im1))



for i in range(nf):

    filename = files[i]
    wtname = filename.replace('.fits','.wt.fits')
    psfname = filename.replace('.fits','_psf.psf')
    # bkgblockname = filename.replace('.fits','.bkgblock.fits')
    removeobjcatname = filename.replace('.fits','.remove_obj.cat')


    print('\nFIle: ' + filename)
    home = str(Path.home())

    # # If bkgblock is not available, call bkgblock.py to create one
    # if Path(path + bkgblockname).is_file() == False and sexbackground == False:
    #     print('\tCall bkgblock.py...')
    #     subprocess.run(['python', home+'/sources/90Prime/ICL/bkgblock.py', path+filename])

    # If _psf.psf not available, call make_psf.sh
    if not Path(path + psfname).is_file() and conv:
        print('\tCall make_psf.sh...')
        subprocess.run([home+'/sources/90Prime/ICL/make_psf.sh', path+filename])
        psf = fits.open(path + psfname)
        psf0 = psf[1].data[0][0][0]

    f = fits.open(path + filename)
    w = fits.open(path + wtname)

    f0 = f[0].data
    fhdr = f[0].header
    f0p = f0.copy()
    outmask = f0p == 0
    w0 = w[0].data
    whdr = w[0].header

    f1p = f0.copy()
    f1p[f1p==0] = np.nan
    print('Estinating Background...')
    bkg = Background2D(f1p, bsize, filter_size=fsize, sigma_clip=SigmaClip(3.), bkg_estimator=SExtractorBackground())
    bkg1 = bkg.background
    f0p[outmask] = bkg1[outmask]

    if sexbackground:
        f0p = f0p - bkg1
        # f0p[outmask] = 0.0


    if conv == True:
        print('Creating psf matching kernel...')
        kernel = create_matching_kernel(psf0, gauss, window)

        # Figure out the factor between original rms and convolved rms with this kernel
        print('Calculating RMS factor...')
        dat_rmsfac = [[np.random.randint(nrmsfac), kernel] for si in range(nrmsfac)] # the input argument to the function for mp.Pool
        with mp.Pool(npool) as pl:
            res = pl.starmap(func_rmsfac, dat_rmsfac)
        rmsfac = np.mean(res)
        print('RMS factor: {}'.format(rmsfac))

        print('Saving psf and kernel info into .psfs.fits')
        # save original psf, kernel and convolved psf to a psfs.fits file
        hduP = fits.PrimaryHDU()
        hduP.data = np.zeros(psf0.shape)
        hduP.header['D_TYPE'] = ('zero','Data type for psf information')
        hduP.header['tgsig'] = (gaussian_sig, 'Matched target gaussian sigma')
        hduP.header['win_a'] = (alpha, 'Tukey window alpha')
        hduP1 = fits.ImageHDU()
        hduP1.data = psf0
        hduP1.header['D_TYPE'] = ('original','Data type for psf information')
        hduP2 = fits.ImageHDU()
        hduP2.data = kernel
        hduP2.header['D_TYPE'] = ('kernel','Data type for psf information')
        hduP2.header['rmsfac'] = (rmsfac, 'RMS factor from kernel convolving')
        hduP3 = fits.ImageHDU()
        hduP3.data = convolve(psf0, kernel)
        hduP3.header['D_TYPE'] = ('convolved','Data type for psf information')
        hduPA = fits.HDUList([hduP, hduP1, hduP2, hduP3])
        hduPA.writeto(path + filename.replace('.fits','.psfs.fits'), overwrite=True)


    if removestars:
        # mask stars and fill value with nearby annulus
        print('Removing and filling stars with mask...')
        fwcs = WCS(fhdr)
        xbcg, ybcg = cbcg.to_pixel(fwcs)
        starmasks_x, starmasks_y = SkyCoord(starmasks_ra, starmasks_dec, unit='deg').to_pixel(fwcs)
        hbcg = (np.abs(starmasks_x - xbcg) < halfbox) & (np.abs(starmasks_y - ybcg) < halfbox)
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
            i_ap = m_ap.to_image(f0.shape)
            i_aa = m_aa.to_image(f0.shape)
            dat = f0p[i_aa>0]
            #dat1 = sigma_clip(dat, masked=False)
            #med1 = np.median(dat1)
            #std1 = np.std(dat1)
            mean1, med1, std1 = sigma_clipped_stats(dat)
            f0p[i_ap>0] = np.random.normal(med1, std1, len(i_ap[i_ap>0]))

    if conv == True:
        print('Convolve image with kernel...')
        f1 = convolve(f0p, kernel)
        w1 = convolve(w0, kernel)
    else:
        print('No convolution')
        f1 = f0p
        w1 = w0


    fwcs = WCS(fhdr)
    # wwcs = WCS(whdr)
    # make a smaller cutout first for the reprojection
    cf = Cutout2D(f1, cbcg, cutoutbox, wcs=fwcs)
    cwt = Cutout2D(w1, cbcg, cutoutbox, wcs=fwcs)

    fcrpix1 = cf.wcs.wcs.crpix[0]
    fcrpix2 = cf.wcs.wcs.crpix[1]
    fnaxis1 = cutoutbox
    fnaxis2 = cutoutbox

    wtcrpix1 = cwt.wcs.wcs.crpix[0]
    wtcrpix2 = cwt.wcs.wcs.crpix[1]
    wtnaxis1 = cutoutbox
    wtnaxis2 = cutoutbox

    fhdr['crpix1'] = fcrpix1
    fhdr['crpix2'] = fcrpix2
    fhdr['naxis1'] = fnaxis1
    fhdr['naxis2'] = fnaxis2

    whdr['crpix1'] = wtcrpix1
    whdr['crpix2'] = wtcrpix2
    whdr['naxis1'] = wtnaxis1
    whdr['naxis2'] = wtnaxis2

    # create hdu objects for convolved data with the original header, so that they can be reprojected to master trimconv
    hduI = fits.PrimaryHDU()
    hduI.data = cf.data
    hduI.header = fhdr

    hduW = fits.PrimaryHDU()
    hduW.data = cwt.data
    hduW.header = whdr


    print('Reproject image to master cutout...')
    f_reproj = reproject_exact(hduI, finalhdr, parallel=parallelTF, return_footprint=False)
    # print('Done')
    print('Reproject wt image to master cutout...')
    w_reproj = reproject_exact(hduW, finalhdr, parallel=parallelTF, return_footprint=False)
    # print('Done')

    # put original gain back to finalhdr
    finalhdr['GAIN'] = fhdr['GAIN']

    hduI2 = fits.PrimaryHDU()
    if not sexbackground:
        hduI2.data = f_reproj# - model
    else:
        hduI2.data = f_reproj
    hduI2.header = finalhdr
    if conv:
        hduI2.header['rmsfac'] = (rmsfac, 'RMS factor from kernel convolving')
    

    hduW2 = fits.PrimaryHDU()
    hduW2.data = w_reproj
    hduW2.header = finalhdr
    if conv:
        hduW2.header['rmsfac'] = (rmsfac, 'RMS factor from kernel convolving')


    if conv:
        hduI2.writeto(path + filename.replace('.fits','.trimconv.fits'), overwrite=True)
        hduW2.writeto(path + filename.replace('.fits','.trimconv.wt.fits'), overwrite=True)
    elif not conv:
        hduI2.writeto(path + filename.replace('.fits','.trim.fits'), overwrite=True)
        hduW2.writeto(path + filename.replace('.fits','.trim.wt.fits'), overwrite=True) 


