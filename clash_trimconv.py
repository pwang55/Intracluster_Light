"""

Usage:

    In script folder:
    $ python clash_trimconv.py path_to_file/filelist.txt (master_stack_clustername_cutout.trimconv.fits)
    or
    $ python clash_trimconv.py path_to_file/clashimage.fits (master_stack_clustername_cutout.trimconv.fits)

    In data folder:
    $ python path_to_script/clash_trimconv.py filelist.txt (master_stack_clustername_cutout.trimconv.fits)
    or
    $ python path_to_script/clash_trimconv.py clashimage.fits (master_stack_clustername_cutout.trimconv.fits)

Requires:
    clashimage.fits
    clashimage.wt.fits
    master_stack_clustername_cutout.trimconv.fits
    clustername_starmask.csv
Creates:
    clashimage.trim.fits
    clashimage.trim.wt.fits
    clashimage.trim_psf.psf
    clashimage.trim.psfs.fits
    clashimage.trimconv.fits
    clashimage.trimconv.wt.fits

This code reads header from master_stack_clustername_cutout.trimconv.fits, reproject the clash or HST image to it, save as .trim.fits
then called clash_make_psf.sh to create .trim_psf.psf for psf matching, remove stars with clustername_starmask.csv,
convolved the reprojected image with kernel and save as .trimconv.fits.

The master_stack_clustername_cutout.trimconv.fits argument is optional if the file is already in the same folder.


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
from photutils.psf import create_matching_kernel, TukeyWindow, TopHatWindow, SplitCosineBellWindow
from photutils.aperture import CircularAnnulus, aperture_photometry, CircularAperture, EllipticalAnnulus, EllipticalAperture
from astropy.stats import SigmaClip,sigma_clip,sigma_clipped_stats
from astropy.convolution import convolve
from astropy.modeling.models import Gaussian2D
from reproject import reproject_exact
import multiprocessing as mp
from glob import glob
import pickle


mp.set_start_method('fork')

halfbox = 600
clashcutoutbox = 3200

# distance outside star aperture to calculate nearby median & std
dsr = 5
# make the star radius this times larger than the values used in narrowband images, 
# since clash is much deeper and stars extended outwards more
star_r_factor = 2. 
nrmsfac = 20    # number of times to calculate the factor between original rms and kernel convolved rms
npool = 5   # number of pools for mp.Pool

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


# find starmask.csv
if Path(path+clustername+'_starmask.csv').is_file() == True:
    # with open(path+clustername+'_starmask.pkl', 'rb') as pk:
    #     starmasks = pickle.load(pk)
    starmasks = ascii.read(path+clustername+'_starmask.csv')
    # starmasks_x = starmasks['x']
    # starmasks_y = starmasks['y']
    starmasks_ra = starmasks['ra']
    starmasks_dec = starmasks['dec']
    starmasks_r = starmasks['radius'] * star_r_factor
else:
    print(clustername+'_starmask.csv not available!')
    sys.exit()


if len(sys.argv) == 3:
    pathmaster = sys.argv[2]
    if Path(pathmaster).is_file() == True:
        mcutout = fits.open(pathmaster)
        mcutout0 = mcutout[0].data
        finalhdr = mcutout[0].header
        gaussian_sig = finalhdr['TGSIG']
        alpha = finalhdr['WIN_A']
        halfbox = int((finalhdr['naxis1']-1)/2)
        clashcutoutbox = 2.25 * 2 * halfbox + 400
    else:
        print('{}\nnot found!'.format(pathmaster))
        sys.exit()
elif len(sys.argv) == 2:
    mastertrimconv = 'master_stack_{}_cutout.trimconv.fits'.format(clustername)
    if Path(path + mastertrimconv).is_file() == False:
        print('{} not found!'.format(mastertrimconv))
        sys.exit()
    else:
        mcutout = fits.open(path + mastertrimconv)
        mcutout0 = mcutout[0].data
        finalhdr = mcutout[0].header        
        gaussian_sig = finalhdr['TGSIG']
        alpha = finalhdr['WIN_A']
        halfbox = int((finalhdr['naxis1']-1)/2)
        clashcutoutbox = 2.25 * 2 * halfbox + 400

# define function for figuring out the factor between original rms and kernel convolved rms
def func_rmsfac(x, kernel):
    np.random.seed(x)
    im0 = np.random.normal(0, 1, (2*halfbox+1,2*halfbox+1))
    im1 = convolve(im0, kernel)
    return(np.std(im0)/np.std(im1))

print('Halfbox = {}'.format(halfbox))

for i in range(nf):
    filename = files[i]
    wtfilename = filename.replace('.fits','.wt.fits')
    print('\nFile: ' + filename)

    trimname = filename.replace('.fits','.trim.fits')
    pixscale = np.abs(fits.getheader(path + filename)['CD2_2'])
    gain = fits.getheader(path + filename)['GAIN']
    finalhdr['GAIN'] = gain

    if Path(path + trimname).is_file() == False:

        f = fits.open(path + filename)
        wt = fits.open(path + wtfilename)
        f0 = f[0].data
        w0 = wt[0].data
        fhdr = f[0].header
        wthdr = wt[0].header
        # gain = fhdr['GAIN']
        fwcs = WCS(fhdr)
        wtwcs = WCS(wthdr)

        # if clash, make a smaller cutout first for the reprojection
        if filename.split('_')[0] == 'clash' and filename.split('_')[1] != 'hst':
            cf = Cutout2D(f0, cbcg, clashcutoutbox, wcs=fwcs)
            cwt = Cutout2D(w0, cbcg, clashcutoutbox, wcs=wtwcs)

            fcrpix1 = cf.wcs.wcs.crpix[0]
            fcrpix2 = cf.wcs.wcs.crpix[1]
            fnaxis1 = clashcutoutbox
            fnaxis2 = clashcutoutbox

            wtcrpix1 = cwt.wcs.wcs.crpix[0]
            wtcrpix2 = cwt.wcs.wcs.crpix[1]
            wtnaxis1 = clashcutoutbox
            wtnaxis2 = clashcutoutbox

            fhdr['crpix1'] = fcrpix1
            fhdr['crpix2'] = fcrpix2
            fhdr['naxis1'] = fnaxis1
            fhdr['naxis2'] = fnaxis2

            wthdr['crpix1'] = wtcrpix1
            wthdr['crpix2'] = wtcrpix2
            wthdr['naxis1'] = wtnaxis1
            wthdr['naxis2'] = wtnaxis2

        else:
            cf = f0
            cwt = w0

        hduCF = fits.PrimaryHDU()
        hduCF.data = cf.data
        hduCF.header = fhdr

        hduCW = fits.PrimaryHDU()
        hduCW.data = cwt.data
        hduCW.header = wthdr

        print('Reproject image to master cutout...')
        f_reproj = reproject_exact(hduCF, finalhdr, parallel=True, return_footprint=False)
        print('Reproject wt image to master cutout...')
        wt_reproj = reproject_exact(hduCW, finalhdr, parallel=True, return_footprint=False)

        hduI = fits.PrimaryHDU()
        hduI.data = f_reproj
        hduI.header = finalhdr
        hduI.writeto(path + filename.replace('.fits','.trim.fits'), overwrite=True)

        hduI = fits.PrimaryHDU()
        hduI.data = wt_reproj
        hduI.header = finalhdr
        hduI.writeto(path + wtfilename.replace('.wt.fits','.trim.wt.fits'), overwrite=True)

        f.close()
        wt.close()
        del f0, w0

    else:
        print(trimname + ' already exists.')
        f_reproj = fits.getdata(path + trimname)
        wt_reproj = fits.getdata(path + trimname.replace('.fits','.wt.fits'))

    # call clash_make_psf.sh to create psf profile for the trim & reprojected image
    if filename.split('_')[0] == 'clash' and filename.split('_')[1] != 'hst':
        psffilename = trimname.replace('.fits','_psf.psf')
        if Path(path + psffilename).is_file() == False:
            print("{} doesn't exist, calling clash_make_psf.sh...".format(psffilename))
            subprocess.run(['/Users/brianwang76/sources/90Prime/ICL/clash_make_psf.sh', path+trimname])
        psf = fits.open(path + psffilename)[1].data[0][0][0]
        psfgrid = psf.shape[0]

    elif filename.split('_')[0] == 'clash' and filename.split('_')[1] == 'hst':
        filtername = filename.split('.')[0].split('_')[-1]
        psffilename = glob(path+'*{}*psf*fits'.format(filtername))[0]
        print('Using PSF file: {}'.format(psffilename))
        psf = fits.open(psffilename)[0].data
        psfgrid = psf.shape[0]


    f_reprojp = f_reproj.copy()
    # outmask = f_reprojp == 0


    window = TukeyWindow(alpha=alpha)
    # create target gaussian
    y, x = np.mgrid[0:psfgrid, 0:psfgrid]
    gm1 = Gaussian2D(100, (psfgrid-1)/2, (psfgrid)/2, gaussian_sig, gaussian_sig)
    gauss = gm1(x, y)
    gauss /= gauss.sum()

    hdr = finalhdr.copy()
    hdr['pixscl'] = (pixscale, 'CLASH original pixscale')
    # hdr['GAIN'] = gain

    kernel = create_matching_kernel(psf, gauss, window)

    # Figure out the factor between original rms and convolved rms with this kernel
    print('Calculating RMS factor...')
    dat_rmsfac = [[np.random.randint(nrmsfac), kernel] for si in range(nrmsfac)] # the input argument to the function for mp.Pool
    with mp.Pool(npool) as pl:
        res = pl.starmap(func_rmsfac, dat_rmsfac)
    rmsfac = np.mean(res)
    print('RMS factor: {}'.format(rmsfac))


    hduP = fits.PrimaryHDU()
    hduP.data = np.zeros(psf.shape)
    hduP.header['D_TYPE'] = ('zero','Data type for psf information')
    hduP.header['tgsig'] = (gaussian_sig, 'Matched target gaussian sigma')
    hduP.header['win_a'] = (alpha, 'Tukey window alpha')
    hduP1 = fits.ImageHDU()
    hduP1.data = psf
    hduP1.header['D_TYPE'] = ('original','Data type for psf information')
    hduP2 = fits.ImageHDU()
    hduP2.data = kernel
    hduP2.header['D_TYPE'] = ('kernel','Data type for psf information')
    hduP2.header['rmsfac'] = (rmsfac, 'RMS factor from kernel convolving')
    hduP3 = fits.ImageHDU()
    hduP3.data = convolve(psf, kernel)
    hduP3.header['D_TYPE'] = ('convolved','Data type for psf information')
    hduPA = fits.HDUList([hduP, hduP1, hduP2, hduP3])
    hduPA.writeto(path + trimname.replace('.fits','.psfs.fits'), overwrite=True)

    # mask stars and fill value with nearby annulus
    print('Removing and filling stars with mask...')
    fwcs = WCS(finalhdr)
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
        i_ap = m_ap.to_image(f_reproj.shape)
        i_aa = m_aa.to_image(f_reproj.shape)
        dat = f_reprojp[i_aa>0]
        #dat1 = sigma_clip(dat, masked=False)
        #med1 = np.median(dat1)
        #std1 = np.std(dat1)
        mean1, med1, std1 = sigma_clipped_stats(dat)
        f_reprojp[i_ap>0] = np.random.normal(med1, std1, len(i_ap[i_ap>0]))



    print('Convolving image with kernel...')
    f1 = convolve(f_reprojp, kernel)
    # f1 = convolve(f_reproj, kernel)
    w1 = convolve(wt_reproj, kernel)
    hduI = fits.PrimaryHDU()
    hduI.data = f1
    hduI.header = hdr
    hduI.header['rmsfac'] = (rmsfac, 'RMS factor from kernel convolving')
    hduI.writeto(path + filename.replace('.fits','.trimconv.fits'),overwrite=True)

    hduW = fits.PrimaryHDU()
    hduW.data = w1
    hduW.header = hdr
    hduW.header['rmsfac'] = (rmsfac, 'RMS factor from kernel convolving')
    hduW.writeto(path + filename.replace('.fits','.trimconv.wt.fits'),overwrite=True)


