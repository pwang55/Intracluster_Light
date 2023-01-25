'''

'''
import numpy as np
from astropy.io import fits, ascii
from astropy.coordinates import SkyCoord
import subprocess
from glob import glob
import sys
from astropy.table import Table
from pathlib import Path

bsize = '32'
fsize = '1'
sexfilter = False
gaussfilter = 'gauss_4.0_7x7.conv'
aper = True
output = False

script_dir = sys.path[0] + '/'

pathfname = sys.argv[1]
fname = pathfname.split('/')[-1]
path = pathfname[:-len(fname)]

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
                    print('No output, just displaying results...')
                    output = False
            elif arg_type.lower() == 'bsize':
                bsize = arg_val
            elif arg_type.lower() == 'fsize':
                fsize = arg_val
            elif arg_type.lower() == 'sexfilter':
                if arg_val.lower() == 'true':
                    sexfilter = True
                elif arg_val.lower() == 'false':
                    sexfilter = False
            # elif arg_type.lower() == 'aper':
            #     if arg_val.lower() == 'true':
            #         aper = True
            #     elif arg_val.lower() == 'false':
            #         aper = False
            else:
                print('Unrecognized arguments!')
                print(__doc__)
                sys.exit()

print('')
print('BACK_SIZE = \t{}'.format(bsize))
print('BACK_FILTERSIZE = \t{}'.format(fsize))
print('FILTER = \t{}'.format(sexfilter))
if sexfilter:
    print('FILTER_NAME = {}'.format(gaussfilter))
print('')



tbcg = ascii.read('/Users/brianwang76/sources/90Prime/ICL/extra/bcg_table.ecsv')


# figure out cluster name

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
fnamesplit = fname.split('_')

clustername = ''
for i in range(len(fnamesplit)):
    if (fnamesplit[i] == all_cluster_names).any():
        clustername = fnamesplit[i]
if clustername == '':
    print("Cluster name not found!")
    sys.exit()

f = fits.open(pathfname)
pathwtname = pathfname.replace('.fits','.wt.fits')
pathcatname = pathfname.replace('.fits','.bcg.cat')
pathapername = pathfname.replace('.fits','.aper.fits')

print('Running SExtractor...')
if sexfilter:
    subprocess.run(['sex', '-c', script_dir+'extra/remove_obj.sex', pathfname, '-PARAMETERS_NAME', \
                    script_dir+'/extra/param.param', '-WEIGHT_IMAGE', pathwtname, \
                    '-BACK_SIZE', bsize, '-BACK_FILTERSIZE', fsize, '-FILTER_NAME', script_dir+'/extra/'+gaussfilter, \
                    '-CHECKIMAGE_TYPE', 'APERTURES', '-CHECKIMAGE_NAME', pathapername, '-CATALOG_NAME', pathcatname, '-VERBOSE_TYPE', 'QUIET'])
else:
    subprocess.run(['sex', '-c', script_dir+'extra/remove_obj.sex', pathfname, '-PARAMETERS_NAME', \
                    script_dir+'/extra/param.param', '-WEIGHT_IMAGE', pathwtname, \
                    '-BACK_SIZE', bsize, '-BACK_FILTERSIZE', fsize, '-FILTER', 'N', \
                    '-CHECKIMAGE_TYPE', 'APERTURES', '-CHECKIMAGE_NAME', pathapername, '-CATALOG_NAME', pathcatname, '-VERBOSE_TYPE', 'QUIET'])    


cat = ascii.read(pathcatname)

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



