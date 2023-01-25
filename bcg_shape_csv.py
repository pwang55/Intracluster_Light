'''

This code is outdate

Usage:
    $ python path_to_script/bcg_shape_csv.py filename.cat (output=true)

This code reads either iclmask.cat created by mask.py, or remove_obj.cat created by remove_object.py,
or any catalog file that has: 
    ALPHA_J2000, DELTA_J2000
    X_IMAGE, Y_IMAGE, A_IMAGE, B_IMAGE, THETA_IMAGE

find the clustername and bcg location, 
save the bcg morpholofical info in a table.
Important: this catalog has to be created from a trim fits file so that X and Y are correct.

If output=false, the info is only printed on screen but will not overwrite the existing file.

Creates:
    bcg_shape_clustername.csv

'''
import numpy as np
from astropy.io import ascii
from astropy.coordinates import SkyCoord
import sys
from astropy.table import Table
from pathlib import Path

if len(sys.argv) < 2:
    print(__doc__)
    sys.exit()

pathfname = sys.argv[1]
fname = pathfname.split('/')[-1]
path = pathfname[:-len(fname)]
script_dir = sys.path[0] + '/'

output = True


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

            if arg_type == 'output':
                if arg_val.lower() == 'true':
                    output = True
                elif arg_val.lower() == 'false':
                    output = False
            else:
                print('Unrecognized arguments!')
                print(__doc__)
                sys.exit()


tbcg = ascii.read('/Users/brianwang76/sources/90Prime/ICL/extra/bcg_table.ecsv')

pathcatname = sys.argv[1]
cat = ascii.read(pathcatname)

catnamesplit = pathcatname.split('/')[-1].split('.')[0].split('_')
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
for i in range(len(catnamesplit)):
    if (catnamesplit[i] == all_cluster_names).any():
        clustername = catnamesplit[i]
if clustername == '':
    print("Cluster name not found!")
    sys.exit()


tbcg = tbcg[tbcg['cluster'] == clustername]
cbcg = SkyCoord(tbcg['ra'], tbcg['dec'], unit='deg')

coord = SkyCoord(cat['ALPHA_J2000'], cat['DELTA_J2000'], unit='deg')
idx, d2d, d3d = cbcg.match_to_catalog_sky(coord)
idx = idx[0]

cat1 = cat[idx]
a = cat1['A_IMAGE']
b = cat1['B_IMAGE']
x = cat1['X_IMAGE'] - 1
y = cat1['Y_IMAGE'] - 1
theta = cat1['THETA_IMAGE']

if output == True:
    print('X:{}\tY:{}\tA:{}\tB:{}\ttheta:{}'.format(x,y,a,b,theta))
    tab = Table()
    tab.add_columns([[x], [y], [a], [b], [theta]], names=['X','Y','A','B','theta'])
    tab.write(path+'bcg_shape_{}.csv'.format(clustername), format='ascii.ecsv', overwrite=True)
else:
    print('X:{}\tY:{}\tA:{}\tB:{}\ttheta:{}'.format(x,y,a,b,theta))
    if Path(path+'bcg_shape_{}.csv'.format(clustername)).is_file():
        tab = ascii.read(path+'bcg_shape_{}.csv'.format(clustername))
        print('\nCurrent value in the available file:')
        print('X:{}\tY:{}\tA:{}\tB:{}\ttheta:{}'.format(tab['X'][0],tab['Y'][0],tab['A'][0],tab['B'][0],tab['theta'][0]))

