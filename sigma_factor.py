'''

Just an outdated test code. Implemented in trimconv.py and clash_trimconv.py already.

Usage:
    $ python path_to_script/sigma_factor.py psfslist.txt

Requires:
- psfslist.txt (containing all .psfs.fits names)

Creates:
- clustername_sigmafactor.ecsv

'''
import numpy as np
import multiprocessing as mp
from scipy.ndimage import convolve
import sys
from astropy.io import fits, ascii
from astropy.table import Table

mp.set_start_method('fork')

npool = 5   
ndat = 20   # number of times the factor is evaluated


if len(sys.argv) < 2:
    print(__doc__)
    sys.exit()

pathfname = sys.argv[1]
fname = pathfname.split('/')[-1]
path = pathfname[:-len(fname)]

files = []
with open(pathfname) as f:
    for l in f:
        files.append(l.strip())

filenamesplit = files[0].split('.')[0].split('_')
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


def sigmaunc(x, kernel):
    np.random.seed(x)
    im0 = np.random.normal(0,1,(1201,1201))
    im1 = convolve(im0, kernel)
    return(np.std(im0)/np.std(im1))

p0s = []
for i in range(len(files)):
    p = fits.open(files[i])
    p0 = p[2].data
    p0s.append(p0)



for i in range(len(p0s)):
    p0 = p0s[i]
    dat = [[np.random.randint(ndat), p0] for j in range(ndat)]

