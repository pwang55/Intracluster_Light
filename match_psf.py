"""

Usage:

    In python script folder:
    $ python match_psf.py path_to_files/filelist.txt (sigma=3.0) (alpha=0.5) (onlyplot=true)

    In data folder:
    $ python path_to_script/match_psf.py filelist.txt (sigma=3.0) (alpha=0.5) (onlyplot=true)

filelist.txt should contain files with file name cutout_stack_clustername_1ne.fits and .wt.fits, not the one with .deghost.fits.
If images are .deghost.fits, then then should be moved to a separate folder and renamed .fits.
sigma and alpha are optional keywords.

This code matches PSF of images in filelist.txt to a target gaussian (default sigma=3.0) with a window function (default Tukey(alpha=0.5)).
A header keyword "TGSIG" will be added to the header of output fits files.
If keyword "onlyplot" is set to True (default), then this code doesn't output any files and instead only shows matched PSF results. Useful for testing different target gaussian.

Required:
    .fits
    _psf.psf
    .wt.fits
Creates:
    .convolved.fits
    .convolved.wt.fits

"""
import numpy as np
from photutils.psf import create_matching_kernel, TukeyWindow, TopHatWindow, SplitCosineBellWindow
import sys
from astropy.convolution import convolve
from astropy.modeling.models import Gaussian2D
import pathlib
from glob import glob
from astropy.io import fits
import matplotlib.pyplot as plt

# default values for optional inputs
gaussian_sig = 2.5  # default target gaussian sigma
alpha = 0.5  # default Tukey window alpha
plot_only = True

rows=[3,3,3,3,2,2,2,2,1,1,1,1,0,0,0,0]
cols=[0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3]

path = ''
script_dir = pathlib.Path(__file__).parent.absolute()
working_dir = pathlib.Path().absolute()


if len(sys.argv) == 1:
    print(__doc__)
    sys.exit()

path_file = sys.argv[1]
filelistname = path_file.split('/')[-1]
path = path_file[:-len(filelistname)]

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
            elif arg_type == 'onlyplot':
                if arg_val == 'true':
                    plot_only = True
                elif arg_val == 'false':
                    plot_only = False
            else:
                print('Unrecognized arguments!')
                print(__doc__)
                sys.exit()



print('gaussian sigma = {}'.format(gaussian_sig))
print('Tukey Window alpha = {}'.format(alpha))
print('Plot only = {}\n'.format(plot_only))

psffile = glob('*_psf.psf')[0]
psf = fits.open(psffile)[1].data[0][0][0]
psfgrid = psf.shape[0]
window = TukeyWindow(alpha=alpha)

# create target gaussian
y, x = np.mgrid[0:psfgrid, 0:psfgrid]
gm1 = Gaussian2D(100, (psfgrid-1)/2, (psfgrid)/2, gaussian_sig, gaussian_sig)
gauss = gm1(x, y)
gauss /= gauss.sum()

psfs = []
files0 = glob(path + 'cutout_stack_*fits')
month_Dict = {'abell1576': 'FebMar', 'abell370': 'Jan', 'abell611': 'FebMar', 'macs0329': 'Jan', 'macs0717': \
                'FebMar', 'macs1115': 'Jan', 'macs1149': 'Jan', 'rxj1532': 'FebMar', 'zwicky1953': 'FebMar'}

# clustername = files0[0].split('/')[-1].split('_')[2]
# month = month_Dict[clustername]


# if month == 'FebMar':
#     filters = ['1sw', '3nw', '1se', '2nw', '3ne', '2ne', '3sw', '2sw', '3se', '2se', '4nw', '4ne', '4sw', '4se', '1ne', '1nw']
# elif month == 'Jan':
#     filters = []

#nf = len(filters)

filenames = []
# for i in range(len(filters)):
#     if len(glob(path + 'cutout_stack_*{}.fits'.format(filters[i]))) > 0:
#         file0=glob(path + 'cutout_stack_*{}.fits'.format(filters[i]))[0]
#         filenames.append(file0)
with open(path_file) as ff:
    for l in ff:
        filenames.append(l.strip())
nf = len(filenames)

clustername = filenames[0].split('/')[-1].split('_')[2]
month = month_Dict[clustername]

print('Files to match psf:')
for i in range(len(filenames)):
    print(filenames[i])
print('\n')

psfnames = [filenames[i].replace('.fits','_psf.psf') for i in range(nf)]
wtnames = [filenames[i].replace('.fits','.wt.fits') for i in range(nf)]

# if onlyplot set to true, then only make psf plots
if plot_only == True:
    fig0, axs0 = plt.subplots(4,4, sharex=True, sharey=True, figsize=(12,12))
    fig1, axs1 = plt.subplots(4,4, sharex=True, sharey=True, figsize=(12,12))
    fig2, axs2 = plt.subplots(1,2, sharex=True, sharey=True, figsize=(6,3))

for i in range(nf):
    filename = filenames[i]
    psfname = psfnames[i]
    wtname = wtnames[i]
    print('Matching: {}'.format(filename))

    f = fits.open(path + filename)
    f0 = f[0].data
    hdr = f[0].header
    psf = fits.open(path + psfname)
    psf0 = psf[1].data[0][0][0]
    w = fits.open(path + wtname)
    w0 = w[0].data
    whdr = w[0].header
    f.close()
    w.close()
    psf.close()

    kernel = create_matching_kernel(psf0, gauss, window)
    hdr['tgsig'] = (gaussian_sig, 'Matched target gaussian sigma')
    whdr['tgsig'] = (gaussian_sig, 'Matched target gaussian sigma')
    hdr['win_a'] = (alpha, 'Tukey window alpha')
    whdr['win_a'] = (alpha, 'Tukey window alpha')

    if plot_only == False:
        f1 = convolve(f0, kernel)
        w1 = convolve(w0, kernel)
        hduI = fits.PrimaryHDU()
        hduI.data = f1
        hduI.header = hdr
        hduI.writeto(path + filename.replace('.fits','.convolved.fits'),overwrite=True)

        hduW = fits.PrimaryHDU()
        hduW.data = w1
        hduW.header = whdr
        hduW.writeto(path + wtname.replace('.wt.fits','.convolved.wt.fits'),overwrite=True)

    else:
        if filename.split('_')[0] == 'master':
            filtname = 'master'
            ax0 = axs2[0]
            ax1 = axs2[1]
        else:
            # filtname = filename.replace('.fits','')[-3:]
            filenamesplits = filename.split('_')
            if filenamesplits[1] == 'stack':
                filtname = filenamesplits[-1][:3]
            elif filenamesplits[1] == 'science':
                filtname = filenamesplits[3][-1] + filenamesplits[4][:2]
            ax0 = axs0[rows[i], cols[i]]
            ax1 = axs1[rows[i], cols[i]]

        psf1 = convolve(psf0, kernel)

        ax0.imshow(psf0)
        ax1.imshow(psf1)
        ax0.set_title(filtname)
        ax1.set_title(filtname)

if plot_only == True:
    fig0.suptitle('Original PSF')
    fig1.suptitle('Target Gaussian sigma={}'.format(gaussian_sig))
    fig0.tight_layout()
    fig1.tight_layout()
    plt.show()



