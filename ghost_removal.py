"""

Usage:

    In python script directory:
    $ python ghost_removal.py path_to_files/filelist.txt

    In data directory:
    $ python path_to_script/ghost_removal.py filelist.txt

This code takes files in filelist.txt, read the ghost location files in script/extra and use .noobj.fits to calculate ghost, 
then remove it from the image. Also calculate the secondary ghost if applicable. 
The code assumes a simple radially symmetric circular ghost pattern.

Requires:
    .fits
    .wt.fits
    .noobj.fits
Creates:
    .ghost.fits
    .deghost.fits
    .deghost.wt.fits

Noted that if in the ghost lotacion table a row is -1 it means the ghost cannot be calculated. 
Even if the filename is listed in the filelist.txt, it will be ignored and there will be no output from that file.

"""
import numpy as np
from astropy.io import fits, ascii
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.wcs import WCS
from astropy import units as u
from photutils import CircularAnnulus
import pathlib
import sys
from glob import glob
from astropy.stats import sigma_clip
import pickle
import subprocess

dr = 10 # radius increments in pixels
additional_nr = 6   # extra few data points of annulus pass the radius read from table
additional_nr2 = 6  # extra few data points of annulus pass the radius for secondary ghost
start_r = 0.1   # initial r + start_r since CircularAnnulus can't start from zero

# path to files
path = ''
if len(sys.argv) != 2:
    print(__doc__)
    sys.exit()
else:
    path_file = sys.argv[1]
    filelistname = path_file.split('/')[-1]
    path = path_file[:-len(filelistname)]

files = []
with open(path_file) as ff:
    for l in ff:
        files.append(l.strip()) # the filenames in this files list doesn't contain full path, only each file name

clustername = files[0].split('_')[2]

# Read the table for all ghost locations and size
script_dir = pathlib.Path(__file__).parent.absolute().as_posix()
tab = ascii.read(script_dir+'/extra/{}_ghost_locations.ecsv'.format(clustername))
tab2_filepath = script_dir+'/extra/{}_ghost2_locations.ecsv'.format(clustername)
if pathlib.Path(tab2_filepath).is_file():
    tab2 = ascii.read(tab2_filepath)


ntab = len(tab)
ntab2 = len(tab2)
filters = tab['filt']
filters2 = tab2['filt']
# files = []
# files = filenames

# for i in range(len(filters)):
#     if len(glob(path + 'cutout_stack_*{}.noobj.fits'.format(filters[i]))) > 0:
#         file0=glob(path + 'cutout_stack_*{}.noobj.fits'.format(filters[i]))[0]
#         files.append(file0)
#     if len(file0) == 0:
#         print(".noobj.fits files don't exist! Existing...")
#         sys.exit()
actual_filters = [files[i].split('_')[-1][:3] for i in range(len(files))]   # the filters that are in filelist.txt
actual_filters = np.array(actual_filters)

allfilters = []
allprofs = []
allstds = []
allnpixs = []
allxprofs = []

# Removing the main ghost
# for i in range(ntab):
for k in range(len(actual_filters)):
    fname = files[k]
    noobjname = fname.replace('.fits','.noobj.fits')
    filt = actual_filters[k]
    i = np.where(tab['filt'] == filt)[0][0]

    ra = tab[i]['ra']
    dec = tab[i]['dec']
    radius = tab[i]['radius']

    # Only work on files if their radius > 0, e.g. not nw fields
    if radius > 0:
        # fname = files[i]
        print('Calculating ghost image for: \t{}'.format(fname))
        f = fits.open(path + noobjname)
        f0 = f[0].data
        hdr = f[0].header
        wcs = WCS(hdr)

        coord = SkyCoord(ra, dec, unit='deg')
        position = coord.to_pixel(wcs, origin=0)

        nr = int(radius/dr) + additional_nr

        gprof = []
        gstd = []
        gnpix = []
        xprof = []

        fg = np.zeros(f0.shape)

        for j in range(nr):

            r1 = start_r + j * dr
            r2 = r1 + dr
            aa = CircularAnnulus(position, r1, r2)
            aam = aa.to_mask('center')
            iaam = aam.to_image(fg.shape)

            dat = aam.multiply(f0)
            dat1 = sigma_clip(dat[dat>0], sigma=2.5, masked=False)
            med1 = np.median(dat1)
            std1 = np.std(dat1)
            gprof.append(med1)
            gstd.append(std1)
            gnpix.append(len(dat1))
            xprof.append((r1+r2)/2)
            fg[iaam>0] = med1
        
        gprof = np.array(gprof)
        gstd = np.array(gstd)
        gnpix = np.array(gnpix)
        xprof = np.array(xprof)
        fg[fg!=0] -= np.min(gprof[-additional_nr-5:]) # find minimum of the last few data points to avoid finding minimum at the center of the ghost
        allprofs.append(gprof)
        allstds.append(gstd)
        allnpixs.append(gnpix)
        allxprofs.append(xprof)
        allfilters.append(filters[i])

        # write ghost to image
        hduI = fits.PrimaryHDU()
        hduI.data = fg
        hduI.header = hdr
        hduI.writeto(path + fname.replace('.fits', '.ghost.fits'), overwrite=True)

        # write ghost subtracted image to new image file
        img = fits.open(path + fname)
        img0 = img[0].data
        hdr0 = img[0].header
        hduI = fits.PrimaryHDU()
        hduI.data = img0 - fg
        hduI.header = hdr0
        hduI.writeto(path + fname.replace('.fits', '.deghost.fits'), overwrite=True)

        imgwt = fits.open(path + fname.replace('.fits','.wt.fits'))
        imgwt0 = imgwt[0].data
        hdrwt0 = imgwt[0].header
        hduI = fits.PrimaryHDU()
        hduI.data = imgwt0  # TEMP need to figure out how ghost affects weight image
        hduI.header = hdrwt0
        hduI.writeto(path + fname.replace('.fits', '.deghost.wt.fits'), overwrite=True) 

        img.close()
        imgwt.close()
        f.close()
    else:
        print("Ignore: \t{}".format(fname))

all_stuffs = {}
all_stuffs['profiles'] = allprofs
all_stuffs['stds'] = allstds
all_stuffs['npixs'] = allnpixs
all_stuffs['xpts'] = allxprofs
all_stuffs['filters'] = allfilters

all_stuffs_both = {}
all_stuffs_both['1st'] = all_stuffs



# Removing secondary ghost for 4 images in tab2, if tab2 exists
if pathlib.Path(tab2_filepath).is_file():
    print('Also calculating and removing secondary ghosts...\n')

    allfilters2 = []
    allprofs2 = []
    allstds2 = []
    allnpixs2 = []
    allxprofs2 = []

    filters3 = []
    noobjfiles = []
    ghostfiles = []
    deghostfiles = []

    for i in range(len(filters2)):
        if (filters2[i]==actual_filters).any():
            # if len(glob(path + 'cutout_stack_*{}.noobj.fits'.format(filters[i]))) > 0:
            noobjfile = glob(path + 'cutout_stack_*{}.noobj.fits'.format(filters2[i]))[0]
            ghostfile = noobjfile.replace('.noobj.fits', '.ghost.fits')
            deghostfile = ghostfile.replace('.ghost.fits', '.deghost.fits')
            noobjfiles.append(noobjfile)
            ghostfiles.append(ghostfile)
            deghostfiles.append(deghostfile)
            filters3.append(filters2[i])

    if len(noobjfiles) > 0:
        for i in range(len(noobjfiles)):
            print('Removing secondary ghost from: \tcutout_stack_{}_{}.fits'.format(clustername, filters3[i]))
            noobjfile = noobjfiles[i]
            ghostfile = ghostfiles[i]
            deghostfile = deghostfiles[i]
            noobj = fits.open(noobjfile)
            ghost = fits.open(ghostfile)
            deghost = fits.open(deghostfile)
            hdr = deghost[0].header
            noobj0 = noobj[0].data
            ghost0 = ghost[0].data
            deghost0 = deghost[0].data
            noobj.close()
            ghost.close()
            deghost.close()

            ra = tab2[i]['ra']
            dec = tab2[i]['dec']
            radius = tab2[i]['radius']
            wcs = WCS(hdr)
            coord = SkyCoord(ra, dec, unit='deg')
            position = coord.to_pixel(wcs, origin=0)

            nr = int(radius/dr) + additional_nr2

            gprof2 = []
            gstd2 = []
            gnpix2 = []
            xprof2 = []

            fg2 = np.zeros(noobj0.shape)
            dat0 = noobj0 - ghost0    # remove primary ghost so the only thing left is secondary ghost

            for j in range(nr):

                r1 = start_r + j * dr
                r2 = r1 + dr
                aa = CircularAnnulus(position, r1, r2)
                aam = aa.to_mask('center')
                iaam = aam.to_image(fg2.shape)

                dat = aam.multiply(dat0)
                dat1 = sigma_clip(dat[dat>0], sigma=2.5, masked=False)
                med1 = np.median(dat1)
                std1 = np.std(dat1)
                gprof2.append(med1)
                gstd2.append(std1)
                gnpix2.append(len(dat1))
                xprof2.append((r1+r2)/2)
                fg2[iaam>0] = med1
            
            gprof2 = np.array(gprof2)
            gstd2 = np.array(gstd2)
            gnpix2 = np.array(gnpix2)
            xprof2 = np.array(xprof2)
            fg2[fg2!=0] -= np.min(gprof2[-additional_nr-5:]) # find minimum of the last few data points to avoid finding minimum at the center of the ghost
            allprofs2.append(gprof2)
            allstds2.append(gstd2)
            allnpixs2.append(gnpix2)
            allxprofs2.append(xprof2)
            allfilters2.append(filters3[i])

            # add 2nd ghosts to the existed ghost image
            hduI = fits.PrimaryHDU()
            hduI.data = ghost0 + fg2
            hduI.header = hdr
            hduI.writeto(ghostfile, overwrite=True)

            # subtract 2nd ghost from deghost image
            hduI = fits.PrimaryHDU()
            hduI.data = deghost0 - fg2
            hduI.header = hdr
            hduI.writeto(deghostfile, overwrite=True) 

            # write weight image TEMP 
            # hduI = fits.PrimaryHDU()
            # hduI.data = deghost0 - fg2
            # hduI.header = hdr
            # hduI.writeto(deghostfile, overwrite=True) 

        all_stuffs2 = {}
        all_stuffs2['profiles'] = allprofs2
        all_stuffs2['stds'] = allstds2
        all_stuffs2['npixs'] = allnpixs2
        all_stuffs2['xpts'] = allxprofs2
        all_stuffs2['filters'] = allfilters2

        all_stuffs_both['2nd'] = all_stuffs2

# save profiles to pickle files
with open(path+'{}_ghost_profiles.pkl'.format(clustername), 'wb') as p1:
    pickle.dump(all_stuffs_both, p1)


