'''

Usage:
    $ python path_to_script/other_galaxies.py filelist.txt master_stack_clustername_cutout.trimconv.iclmask.cat other_catalog (ratio=3)

Requires:
- filelist.txt containing names of all trimconv.fits
- *trimconv.fits, both narrow-band and clash
- master_stack_clustername_cutout.trimconv.iclmask.cat
- [other_catalog]: can be one of the following:
    clustername_all_zspecs_sdss(panstarrs).zall
    clustername_gal_sdss_radec.csv
    clustername_gal_panstarrs_radec.csv
    hlsp_clash_subaru_suprimecam_clustername_cat.txt




'''
import numpy as np
from astropy.io import fits, ascii
from photutils import EllipticalAnnulus, EllipticalAperture
from astropy.stats import sigma_clip, sigma_clipped_stats, SigmaClip
import sys
from astropy.table import Table, vstack
from pathlib import Path
from glob import glob
from dust_extinction.parameter_averages import F99, F04, F19
from astropy import units as u
from astropy.coordinates import SkyCoord
from photutils.aperture import ApertureStats

script_dir = sys.path[0] + '/'
extinction_model = F19
sigclip = SigmaClip(sigma=3.0, maxiters=10)

aper_ratio = 3  # bcg_aper_ratio * a and b as the aperture size to measure the bcg flux
aper_bkg_ratio = 5
aper_bkg_db = 5
apply_extinction = True
rmsfactor = True
bkgfromfits = True
method = 'sum'  # med or sum
subtract_bkg = False

if len(sys.argv) > 4:
    nargs = len(sys.argv)
    for i in range(3, nargs):
        arg0 = sys.argv[i]
        if len(arg0.split('=')) == 1:
            print('Use "=" to separate keywords and their value')
            sys.exit()
        else:
            arg_type = arg0.split('=')[0].lower()
            arg_val = arg0.split('=')[-1].lower()
            if arg_type == 'rmsfactor':
                if arg_val.lower() == 'true':
                    rmsfactor = True
                elif arg_val.lower() == 'false':
                    rmsfactor = False
            elif arg_type == 'ratio':
                aper_ratio = float(arg_val)
            #     if not (method == 'med' or method == 'sum'):
            #         print('Incorrect method keyword! method = sum or med')
            #         sys.exit()
            else:
                print('Unrecognized arguments!')
                print(__doc__)
                sys.exit()


if len(sys.argv) < 4:
    print(__doc__)
    sys.exit()

pathfname = sys.argv[1]
fname = pathfname.split('/')[-1]
path = pathfname[:-len(fname)]

aperfname = sys.argv[2]
apers_tab = ascii.read(aperfname)
# pts = len(apers_tab)

zallname = sys.argv[3]
zall = ascii.read(zallname)


maskfname = aperfname.replace('.cat','.fits')
hdr = fits.getheader(maskfname)
pixscale = np.abs(hdr['cd1_2']) * 3600
naxis = hdr['naxis1']





filters = ['1sw', '3nw', '1se', '2nw', '3ne', '2ne', '3sw', '2sw', '3se', '2se', '4nw', '4ne', '4sw', '4se', '1ne', '1nw', 
           'u', 'b', 'v', 'rc', 'ip', 'ic', 'z', 'ks']
nfilters = ['1sw', '3nw', '1se', '2nw', '3ne', '2ne', '3sw', '2sw', '3se', '2se', '4nw', '4ne', '4sw', '4se', '1ne', '1nw']
clashfilters = ['u', 'ks', 'b', 'ic', 'ip', 'rc', 'v', 'z']

lam_dict = {'1sw': 5100, '3nw': 5200, '1se': 5320, '2nw': 5400,
            '3ne': 5500, '2ne': 5600, '3sw': 5680, '2sw': 5800, 
            '3se': 5890, '2se': 6000, '4nw': 6100, '4ne': 6200, 
            '4sw': 6300, '4se': 6400, '1ne': 6500, '1nw': 6600,
            'u': 3817.2, 'ks': 21574, 'b': 4448, 'ic': 7960.5,
            'ip': 7671.2, 'rc': 6505.4, 'v': 5470.2, 'z': 9028.2}

filt_dict = {'1sw': 'F350', '3nw': 'F351', '1se': 'F352', '2nw': 'F353', 
             '3ne': 'F354', '2ne': 'F355', '3sw': 'F356', '2sw': 'F357',
             '3se': 'F358', '2se': 'F359', '4nw': 'F360', '4ne': 'F361',
             '4sw': 'F362', '4se': 'F363', '1ne': 'F364', '1nw': 'F365',
             'u': 'F113', 'ks': 'F222', 'b': 'F78', 'ic': 'F283',
             'ip': 'F82', 'rc': 'F285', 'v': 'F79', 'z': 'F83'}

inv_filt_dict = {v: k for k, v in filt_dict.items()}

eazy2lam_dict = {'F350': 5100, 'F351': 5200, 'F352': 5320, 'F353': 5400,
                 'F354': 5500, 'F355': 5600, 'F356': 5680, 'F357': 5800,
                 'F358': 5890, 'F359': 6000, 'F360': 6100, 'F361': 6200,
                 'F362': 6300, 'F363': 6400, 'F364': 6500, 'F365': 6600,
                 'F113': 3817.2, 'F222': 21574, 'F78': 4448, 'F283': 7960.5,
                 'F82': 7671.2, 'F285': 6505.4, 'F79': 5470.2, 'F83': 9028.2}

z_spec_dict = {'abell611': 0.2873, 'macs0717': -1, 'rxj1532': 0.362, 'macs1115': 0.352}
z_phot_dict = {'abell611': 0.3, 'macs0717': 0.395, 'rxj1532': 0.409, 'macs1115': 0.334}

Ebv = {'abell611': 0.0486, 'rxj1532': 0.0257, 'macs1115': 0.0335}

files0 = []

with open(pathfname) as f:
    for l in f:
        files0.append(l.strip())

filenamesplit = files0[0].split('.')[0].split('_')
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
ctab = ascii.read(script_dir + 'extra/bcg_table.ecsv')
ctab = ctab[ctab['cluster'] == clustername]
bcg_ra = ctab['ra']
bcg_dec = ctab['dec']
cbcg = SkyCoord(bcg_ra, bcg_dec, unit='deg')



# Figure out matching catalog type
if zallname.split('.')[-1] == 'zall':
    if zallname.split('/')[-1].split('.')[0].split('_')[-1] == 'sdss':
        othertype = 'szall'
        raname = 'ra'
        decname = 'dec'
    elif zallname.split('/')[-1].split('.')[0].split('_')[-1] == 'panstarrs':
        othertype = 'pzall'
        raname = 'ra'
        decname = 'dec'
    
elif zallname.split('/')[-1].split('_')[1] == 'clash':
    othertype = 'clash'
    raname = 'RA'
    decname = 'Dec'
elif zallname.split('/')[-1] == clustername + '_gal_sdss_radec.csv':
    othertype = 'sdss'
    raname = 'ra'
    decname = 'dec'
elif zallname.split('/')[-1] == clustername + '_gal_panstarrs_radec.csv':
    othertype = 'panstarrs'
    raname = 'raMean'
    decname = 'decMean'
else:
    print('Wrong matching catalog file!')
    sys.exit()



zps_n_fname = glob(path + 'b_values_*_noext_{}.ecsv'.format(clustername))[0]
zps_clash_fname = glob(path + 'zeropoint_clash_{}.csv'.format(clustername))[0]
zps_n_tab = ascii.read(zps_n_fname)
zps_clash_tab = ascii.read(zps_clash_fname)
zps_n_tab.rename_columns(['name','b'], ['filter','zeropoint'])
zps_n_tab['zeropoint'] = zps_n_tab['zeropoint'] * -1    # invert the sign so that all zeropoints are positive

zps_tab = vstack([zps_n_tab['filter','zeropoint'],zps_clash_tab['filter','zeropoint']])

# z_spec = z_spec_dict[clustername]
# z_phot = z_phot_dict[clustername]


files = []
# sort files
for fi in range(len(filters)):
    for j in range(len(files0)):
        filt = files0[j].split('.')[0].split('_')[-1]
        if filters[fi] == filt:
            files.append(files0[j])


my_filters = [] # actual available filters in the files
my_eazynames = []
my_lams = []
my_cols = []
for i in range(len(files)):
    file1 = files[i]
    filtname = file1.split('.')[0].split('_')[-1]
    my_filters.append(filtname)
    my_eazynames.append(filt_dict[filtname])
    my_lams.append(lam_dict[filtname])
    my_cols.append(filt_dict[filtname])
    my_cols.append(filt_dict[filtname].replace('F','E'))
    # my_cols.append(filt_dict[filtname].replace('F','ext_'))    # extinction columns


my_cols.insert(0, 'id')
my_cols.insert(0, 'dec')
my_cols.insert(0, 'ra')
if othertype == 'zall':
    my_cols.append('z_phot')
    my_cols.append('z_spec')
    my_cols.append('nfilts_narrowband')
# my_cols.append('b1')
# my_cols.append('b2')

my_filters = np.array(my_filters)
my_eazynames = np.array(my_eazynames)
my_lams = np.array(my_lams)
my_cols = np.array(my_cols)

# calculate extinction
fac = np.log(10)/2.5
extinction_r = extinction_model(Rv=3.1)(my_lams * u.AA)
extinction_dmag = extinction_r * 3.1 * Ebv[clustername]
extinction_flux_factors = (1 + fac * extinction_dmag)

tab = Table(np.full((len(apers_tab), len(my_cols)), -99.0), names=my_cols)
tab.meta['pixscale'] = pixscale

tab['id'] = apers_tab['NUMBER']
tab['ra'] = apers_tab['ALPHA_J2000']
tab['dec'] = apers_tab['DELTA_J2000']


# match to existing catalogs so that the fitting won't take 4 days
tra = tab['ra']
tdec = tab['dec']
zra = zall[raname]
zdec = zall[decname]
# if othertype == 'zall':
#     zra = zall['ra']
#     zdec = zall['dec']
# elif othertype == 'clash':
#     zra = zall['RA']
#     zdec = zall['Dec']    

tcoord = SkyCoord(tra, tdec, unit='deg')
zcoord = SkyCoord(zra, zdec, unit='deg')

# first remove bcg from tab
idx0, d2d0, d3d0 = cbcg.match_to_catalog_sky(tcoord)
tab.remove_row(int(idx0))
apers_tab.remove_row(int(idx0))

# create the coordinates again
tra = tab['ra']
tdec = tab['dec']
tcoord = SkyCoord(tra, tdec, unit='deg')


idx, d2d, d3d = tcoord.match_to_catalog_sky(zcoord)
h = d2d.arcsec<1
idx1 = idx[h]
tab1 = tab[h]
zall1 = zall[idx[h]]

if othertype == 'szall' or othertype == 'pzall':
    tab1['z_spec'] = zall1['z_spec']
    tab1['z_phot'] = zall1['z_m2']
    tab1['nfilts_narrowband'] = zall1['nfilts_narrowband']

x_s = apers_tab['X_IMAGE'] - 1
y_s = apers_tab['Y_IMAGE'] - 1
a_s = apers_tab['A_IMAGE'] * aper_ratio
b_s = apers_tab['B_IMAGE'] * aper_ratio
a_bkg = apers_tab['A_IMAGE'] * aper_bkg_ratio
b_bkg = apers_tab['B_IMAGE'] * aper_bkg_ratio

theta_s = apers_tab['THETA_IMAGE'] * np.pi/180


x_s = x_s[h]
y_s = y_s[h]
a_s = a_s[h]
b_s = b_s[h]
a_bkg = a_bkg[h]
b_bkg = b_bkg[h]
theta_s = theta_s[h]
pts = len(tab1)

for i in range(len(files)):

    fname1 = files[i]
    f = fits.open(path + fname1)
    f0 = f[0].data
    w0 = fits.getdata(path + fname1.replace('.fits','.wt.fits'))
    rms0 = 1/w0
    fhdr = f[0].header
    gain = fhdr['GAIN']
    if rmsfactor:
        error_factor = fhdr['rmsfac']
    else:
        error_factor = 1.0
    f1 = f0.copy()
    # f1[mask] = np.nan
    filt = my_filters[i]
    eazyfiltname = my_eazynames[i]
    eazyerrname = eazyfiltname.replace('F','E')
    extfiltname = eazyfiltname.replace('F','ext_')

    zeropoint = zps_tab[zps_tab['filter']==filt]['zeropoint'][0]

    # adjust the zeropoint if it is clash
    if fname1.split('_')[0] == 'clash':
        clash_pixscale = fhdr['pixscl']
        my_pixscale = np.abs(fhdr['cd1_2'])
        zeropoint_r = (clash_pixscale/my_pixscale)**2
        zeropoint = zeropoint + 2.5*np.log10(zeropoint_r)
    # alternatively if it is not clash, subtract the background if provided
    elif bkgfromfits:
        bkgname1 = fname1.replace('.fits','.bkg.fits')
        bkg00 = fits.open(bkgname1)
        bkg0 = bkg00[0].data
        bhdr = bkg00[0].header
        f1 = f1 - bkg0
        bsize = bhdr['bsize']
        fsize = bhdr['fsize']
        print('{}: Background subtracted, bsize = {}, fsize = {}'.format(fname1, bsize, fsize))

    flux_factor = 10**(-0.4 * (zeropoint - 23.9))   # factor to multiply to flux from image so that they are in mJy
    if apply_extinction == True:
        extinction_flux_factor = extinction_flux_factors[i]
    else:
        extinction_flux_factor = 1.0

    f2 = f1/gain
    # loop over each object to calculate total fluxes in their apertures
    for j in range(pts):
        xj = x_s[j]
        yj = y_s[j]
        aj = a_s[j]
        bj = b_s[j]
        abj = a_bkg[j]
        bbj = b_bkg[j]
        thetaj = theta_s[j]

        # print(xj,yj,aj,bj,abj,bbj,thetaj)

        aperj = EllipticalAperture((xj,yj), a=aj, b=bj, theta=thetaj)
        # maperj = aperj.to_mask('center')
        # iaperj = maperj.to_image((naxis, naxis))
        # hj = iaperj > 0
        # dat_aperj = f1[hj]
        # flux_aperj = np.sum(dat_aperj)
        stats_aperj = ApertureStats(f1, aperj)
        flux_aperj = stats_aperj.sum

        # background level around
        if subtract_bkg == True:
            b_in = bbj
            b_out = bbj + aper_bkg_db
            a_in = abj
            a_out = b_out * (aj/bj)
            bkg_aperj = EllipticalAnnulus((xj,yj), a_in=a_in, a_out=a_out, b_in=b_in, b_out=b_out, theta=thetaj)
            # mbkg_aperj = bkg_aperj.to_mask('center')
            # ibkg_aperj = mbkg_aperj.to_image((naxis, naxis))
            # hbj = ibkg_aperj > 0
            # dat_bkg_aperj = f1[hbj]
            # bmean1, bmed1, bstd1 = sigma_clipped_stats(dat_bkg_aperj, sigma=3)
            stats_bkg_aperj = ApertureStats(f1, bkg_aperj, sigma_clip=sigclip)
            bmed1 = stats_bkg_aperj.median

            flux_aperj = flux_aperj - bmed1

        # f2 = f1/gain
        # dat_aperj_div_g = f2[hj]
        # dat_aperj_w = w0[hj]
        stats_aperj_div_g = ApertureStats(f2, aperj)
        stats_rms_aperj = ApertureStats(rms0, aperj)
        # err_aperj = np.sqrt(np.sum(1/dat_aperj_w + dat_aperj_div_g))
        err_aperj = np.sqrt(stats_rms_aperj.sum + stats_aperj_div_g.sum)

        tab1[j][eazyfiltname] = flux_aperj * flux_factor * extinction_flux_factor
        tab1[j][eazyerrname] = err_aperj * flux_factor * extinction_flux_factor

        if np.isnan(flux_aperj):
            tab1[j][eazyfiltname] = -99.0
        if np.isnan(err_aperj):
            tab1[j][eazyerrname] = -99.0




outputname = clustername + '_other_gals_{}_fast.cat'.format(othertype)
ascii.write(tab1, output=path+outputname, format='commented_header', delimiter='\t', overwrite=True)


