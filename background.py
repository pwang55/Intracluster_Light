'''

Usage:
    $ python path_to_script/background.py trimlist.txt (bsize=55) (fsize=5) (sigma=3) (combine=false)
    or
    $ python path_to_script/background.py filename.trim.fits (bsize=55) (fsize=5) (sigma=3) (combine=false)

Takes .trim.fits as input, 
creates a background file with Photutils.background.Background2D, takes .psfs.fits to convolve with kernel.
Also output median, mean and std images of all backgrounds if input is a txt file.

'''
import numpy as np
from photutils.background import Background2D, MedianBackground, SExtractorBackground
from astropy.stats import SigmaClip, sigma_clip, sigma_clipped_stats
import sys
from astropy.io import fits
from astropy.convolution import convolve


bsize = 55
fsize = 5
sigma = 3
# bkg_estimator = MedianBackground()
bkg_estimator = SExtractorBackground()
combine = False


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

            if arg_type == 'bsize':
                bsize = float(arg_val)
            elif arg_type == 'fsize':
                fsize = float(arg_val)
            elif arg_type == 'sigma':
                sigma = float(arg_val)
            elif arg_type == 'combine':
                if arg_val.lower() == 'true':
                    combine = True
                elif arg_val.lower() == 'false':
                    combine = False
            else:
                print('Unrecognized arguments!')
                print(__doc__)
                sys.exit()

if len(sys.argv) < 2:
    print(__doc__)
    sys.exit()

print('bsize = {}, fsize = {}'.format(bsize, fsize))

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

f0t = fits.getdata(files[0])
naxis = f0t.shape[0]
f00 = np.zeros((len(files), naxis, naxis))

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


for i in range(len(files)):
    file = files[i]
    psfname = file.replace('.fits','.psfs.fits')
    print('Estimating background for:\t' + file)

    f = fits.open(path + file)
    f0 = f[0].data
    hdr = f[0].header
    outmask = f0 == 0
    f0[outmask] = np.nan
    bkg = Background2D(f0, bsize, filter_size=fsize, sigma_clip=SigmaClip(sigma), bkg_estimator=bkg_estimator)
    z = bkg.background

    kernel = fits.open(path + psfname)[2].data
    z1 = convolve(z, kernel)

    hduI = fits.PrimaryHDU()
    hduI.data = z1
    hduI.header = hdr
    hduI.header['bsize'] = (bsize, 'Background BACK_SIZE')
    hduI.header['fsize'] = (fsize, 'Background BACK_FILTERSIZE')
    hduI.header['bkgest'] = (str(bkg_estimator).split('(')[0][1:], 'bkg_estimator for photutils')
    hduI.writeto(path + file.replace('.trim.fits', '.trimconv.bkg.fits'), overwrite=True)

    f00[i] = z1

# If input file is a text file containing multiple files, combine them as median, mean and std
if fname_extention == 'txt' and combine:

    finalfilename = file.split('.')[0][:-1-len(file.split('.')[0].split('_')[-1])]  # get rid of the filter name
    finalfilename_suffix = file[len(file.split('.')[0]):]   # any suffix after main filename, for example .trimconv.fits

    med00 = np.median(f00, axis=0)
    mean00 = np.mean(f00, axis=0)
    std00 = np.std(f00, axis=0)

    hduI = fits.PrimaryHDU()
    hduI.data = med00
    hduI.header = hdr
    hduI.header['bsize'] = (bsize, 'Background BACK_SIZE')
    hduI.header['fsize'] = (fsize, 'Background BACK_FILTERSIZE')
    hduI.header['bkgest'] = (str(bkg_estimator).split('(')[0][1:], 'bkg_estimator for photutils')

    hduI.writeto(path + finalfilename + finalfilename_suffix.replace('.fits','.medbkg.fits'), overwrite=True)

    hduI.data = mean00
    hduI.writeto(path + finalfilename + finalfilename_suffix.replace('.fits','.meanbkg.fits'), overwrite=True)

    hduI.data = std00
    hduI.writeto(path + finalfilename + finalfilename_suffix.replace('.fits','.stdbkg.fits'), overwrite=True)


