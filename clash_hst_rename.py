"""

Usage:

    In data folder:
    $ python path_to_script/clash_hst_rename.py filename.fits (test)
    or
    $ python path_to_script/clash_hst_rename.py filelist.txt (test)

This code renames CLASH/HST images to my own convention. 
Add argument "test" in the end to see the results printed without actually renaming files.


"""
import numpy as np
import sys
import os
from pathlib import Path
from glob import glob
import subprocess

if len(sys.argv) == 1:
    print(__doc__)
    sys.exit()

pathfname = sys.argv[1]
fname = pathfname.split('/')[-1]
path = pathfname[:-len(fname)]

testing = False
if len(sys.argv) == 3:
    if sys.argv[2] == 'test':
        testing = True

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

# all possible filters from clash and hst data TEMP might need to add hst filters later too
filters = ['b','rc','ic','ip','v','z','u','ks', \
            'f435w', 'f475w', 'f555w', 'f606w', 'f625w', 'f775w', 'f814w', 'f850lp', \
            'f105w', 'f110w', 'f125w', 'f126n', 'f140w', 'f160w']
filters = np.array(filters)

# all possible cluster names in clash files TEMP
clusternames = ['a611','macs0717','macs1115','macs1149','rxj1532']
clusternames = np.array(clusternames)

# Some keywords in the HFF filename that will be removed before renaming
hff_kwds2remove = ['-60mas', '-65mas', '-hffpar', '-selfcal', '-bkgdcor']

for i in range(len(files)):

    filename = files[i]
    filename_original = filename
    # first get rid of -60mas or -65mas or -hffpar or -selfcal or -bkgdcor
    for j in range(len(hff_kwds2remove)):
        filename = filename.replace(hff_kwds2remove[j], '')
    filenamesplits = filename.split('_')

    clustername = ''
    filter0 = ''
    for j in range(len(filenamesplits)):
        if (filenamesplits[j] == clusternames).any():
            clustername = filenamesplits[j]
        if (filenamesplits[j] == filters).any():
            filter0 = filenamesplits[j]
            filter_idx = j
    if clustername == '':
        print('Clustername not found!')
        sys.exit()
    if filter0 == '':
        print('Filter name not found!')
        sys.exit()

    final_name = ''
    for k in range(1, filter_idx):
        final_name += filenamesplits[k] + '_'
    final_name += filter0

    # print(final_name)
    filename_end = filename.replace('-','_').split('_')[-1]
    if filename_end == 'weight.fits' or filename_end == 'wht.fits':
        final_name += '.wt.fits'
    else:
        final_name += '.fits'

    if clustername == 'a611':
        # clustername = 'abell611'
        final_name = final_name.replace('a611','abell611')
    if testing == False:
        subprocess.run(['mv', path+filename_original, path+final_name])
    else:
        print(filename + '\t' + final_name)
