import numpy as np
from astropy.io import fits
import sys


if len(sys.argv) < 2:
    print(__doc__)
    sys.exit()


pathfname = sys.argv[1]
fname = pathfname.split('/')[-1]
path = pathfname[:-len(fname)]

script_dir = sys.path[0] + '/'

filelist = []
print('Weight combine:')
with open(pathfname) as f:
    for l in f:
        print('\t'+l.strip())
        filelist.append(l.strip())
nf = len(filelist)
nx, ny = fits.getdata(path + filelist[0]).shape

fdats = np.zeros((nf, nx, ny), dtype=float)
wdats = np.zeros((nf, nx, ny), dtype=float)

# get cluster name
filenamesplit = filelist[0].split('.')[0].split('_')
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
# print(filenamesplit)
for i in range(len(filenamesplit)):
    if (filenamesplit[i] == all_cluster_names).any():
        clustername = filenamesplit[i]
if clustername == '':
    print("Cluster name not found!")
    sys.exit()



for i in range(len(filelist)):

    fname1 = filelist[i]
    wname1 = fname1.replace('.fits','.wt.fits')

    f = fits.getdata(path + fname1)
    w = fits.getdata(path + wname1)

    fdats[i] = f
    wdats[i] = w

fdats2 = fdats * wdats  # images x weight
wdats2 = np.sum(wdats, axis=0)  # sum of weight
fdats3 = np.sum(fdats2, axis=0) # sum of images x weight

fdat = fdats3 / wdats2  # weighted average
wdat = wdats2   # in weighted average, the final weight map is just sum of weight, according to SWarp manual

hdr = fits.getheader(path + filelist[0])

hduI = fits.PrimaryHDU()
hduI.data = fdat
hduI.header = hdr
hduI.writeto(path + 'master_clash_{}.trimconv.fits'.format(clustername))

hduW = fits.PrimaryHDU()
hduW.data = wdat
hduW.header = hdr
hduW.writeto(path + 'master_clash_{}.trimconv.wt.fits'.format(clustername))


