'''

Usage:
    $ python path_to_script/radial_profile.py filelist.txt filename.iclmask.fits (bkgfromfits=true) (extinction=true) (method=med/sum) (rmsfactor=true)

Compute radial profiles from all fits files in filelist.txt and sort them, make a ecsv data table.
The table can be process through icl_utility.py functions to become FAST compatible file format.

Requires:
- All fits and wt.fits needed for measuring ICL, and filelist.txt contain names of all fits
- master_stack_clustername_cutout.iclmask.fits
- b_values_sdss(panstarrs)_noext_clustername.ecsv
- zeropoint_clash_clustername.csv

If bkgfromfits=true, code will look for filename.bkg.fits to subtract for each of the image
If meanbkg=meanbkg.fits is provided, code will use this file as a constant background to subtract from all images.
If rmsfactor=false, code will not use rms factor from header.

When method=med, the code calculate sigma clipped median in each annulus and calculate std of the mean as error. Default
When method=sum, the code calculate total sum without sigma clipping, calculate the error the same way as SExtractor and the BCG,
but will average over total number of pixels.

Creates:
- clustername_icls_data.ecsv


'''
import numpy as np
from astropy.io import fits, ascii
from photutils import EllipticalAnnulus, EllipticalAperture
from astropy.stats import sigma_clip, sigma_clipped_stats
import sys
from astropy.table import Table, vstack
from pathlib import Path
from glob import glob
from dust_extinction.parameter_averages import F99, F04, F19
from astropy import units as u


dr = 6  # the first annulus's dr
r0 = 4  # minimum starting semi-minor axis
# r_fac = 1.0 # the factor each time dr multiplies
# r_dr = 1    # the increment each time dr adds
# drs2 = [6,6,6,6,10,10,15,15,15,15]  # b
# drs2 = [8,8,10,10,12,10,10,15,15,15]
# drs2 = [8,8,8,8,8,12,12,15,15,18,18]   # best so far

# TESTING test increasing radius
r_drs2 = 1.3
drs2 = [dr*r_drs2**i for i in range(7)]
n_clipping = 2
sigma = 2.5 # sigma for sigma_clipping in the annulus
# pts = 10    # number of data points in the profile


pts = len(drs2)
extinction_model = F19
bcg_aper_ratio = 3  # bcg_aper_ratio * a and b as the aperture size to measure the bcg flux
apply_extinction = True
rmsfactor = True
method = 'med'  # med or sum
bkgfromfits = True
meanbkg = ''

script_dir = sys.path[0] + '/'


if len(sys.argv) > 3:
    nargs = len(sys.argv)
    for i in range(3, nargs):
        arg0 = sys.argv[i]
        if len(arg0.split('=')) == 1:
            print('Use "=" to separate keywords and their value')
            sys.exit()
        else:
            arg_type = arg0.split('=')[0].lower()
            arg_val = arg0.split('=')[-1].lower()
            if arg_type == 'bkgfromfits':
                if arg_val.lower() == 'true':
                    bkgfromfits = True
                elif arg_val.lower() == 'false':
                    bkgfromfits = False
            elif arg_type == 'rmsfactor':
                if arg_val.lower() == 'true':
                    rmsfactor = True
                elif arg_val.lower() == 'false':
                    rmsfactor = False
            # elif arg_type == 'meanbkg':
            #     meanbkg = arg_val.lower()
            #     bkg = fits.open(meanbkg)
            #     bkg0 = bkg[0].data
            #     bkg.close()
            elif arg_type == 'method':
                method = arg_val
                if not (method == 'med' or method == 'sum'):
                    print('Incorrect method keyword! method = sum or med')
                    sys.exit()
            elif arg_type == 'extinction':
                if arg_val.lower() == 'true':
                    apply_extinction = True
                elif arg_val.lower() == 'false':
                    apply_extinction = False
            else:
                print('Unrecognized arguments!')
                print(__doc__)
                sys.exit()


if len(sys.argv) < 3:
    print(__doc__)
    sys.exit()

pathfname = sys.argv[1]
fname = pathfname.split('/')[-1]
path = pathfname[:-len(fname)]

maskfname = sys.argv[2]
fmask = fits.getdata(maskfname)
mask = np.isnan(fmask)
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


zps_n_fname = glob(path + 'b_values_*_noext_{}.ecsv'.format(clustername))[0]
zps_clash_fname = glob(path + 'zeropoint_clash_{}.csv'.format(clustername))[0]
zps_n_tab = ascii.read(zps_n_fname)
zps_clash_tab = ascii.read(zps_clash_fname)
zps_n_tab.rename_columns(['name','b'], ['filter','zeropoint'])
zps_n_tab['zeropoint'] = zps_n_tab['zeropoint'] * -1    # invert the sign so that all zeropoints are positive

zps_tab = vstack([zps_n_tab['filter','zeropoint'],zps_clash_tab['filter','zeropoint']])

z_spec = z_spec_dict[clustername]
z_phot = z_phot_dict[clustername]

if Path(path + 'bcg_shape_{}.csv'.format(clustername)).is_file:
    bcg_shape = ascii.read(path + 'bcg_shape_{}.csv'.format(clustername))
    bcg_a = bcg_shape['A'][0]
    bcg_b = bcg_shape['B'][0]
    bcg_theta = bcg_shape['theta'][0] * np.pi/180
    bcg_x = bcg_shape['X'][0]
    bcg_y = bcg_shape['Y'][0]
    bcg_ratio = bcg_a/bcg_b
    position = (bcg_x, bcg_y)


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
    my_cols.append(filt_dict[filtname].replace('F','ext_'))    # extinction columns


my_cols.insert(0, 'id')
my_cols.append('z_phot')
my_cols.append('z_spec')
my_cols.append('b1')
my_cols.append('b2')

my_filters = np.array(my_filters)
my_eazynames = np.array(my_eazynames)
my_lams = np.array(my_lams)
my_cols = np.array(my_cols)

# calculate extinction
fac = np.log(10)/2.5
extinction_r = extinction_model(Rv=3.1)(my_lams * u.AA)
extinction_dmag = extinction_r * 3.1 * Ebv[clustername]
extinction_flux_factors = (1 + fac * extinction_dmag)

# construct data table
tab = Table(np.full((pts+1, len(my_cols)), -99.0), names=my_cols)
tab.meta['r0'] = r0
tab.meta['dr'] = dr
tab.meta['A'] = bcg_a
tab.meta['B'] = bcg_b
tab.meta['theta'] = bcg_theta
# tab.meta['bcg_aper_ratio'] = bcg_aper_ratio
tab.meta['pixscale'] = pixscale
# tab.meta['r_fac'] = r_fac
tab.meta['xbcg'] = bcg_x
tab.meta['ybcg'] = bcg_y


# calculate radii first
# drs = [dr*r_fac**i for i in range(pts)]
# drs.insert(0,0)
drs2.insert(0,0)
# r1s = [r0 + np.sum(drs[:i+1]) for i in range(pts)]
# r2s = [r0 + np.sum(drs[:i+2]) for i in range(pts)]
r1s = [r0 + np.sum(drs2[:i+1]) for i in range(pts)]
r2s = [r0 + np.sum(drs2[:i+2]) for i in range(pts)]

for i in range(len(files)):
    fname1 = files[i]
    # f0 = fits.getdata(path + fname1)
    # if bkgfromfits:
    #     bkgname1 = fname1.replace('.fits','.bkg.fits')
    #     bkg1 = fits.open(bkgname1)
    # elif len(meanbkg) > 0:
    #     bkg1 = bkg0
    f = fits.open(path + fname1)
    f0 = f[0].data
    w0 = fits.getdata(path + fname1.replace('.fits','.wt.fits'))
    fhdr = f[0].header
    gain = fhdr['GAIN']
    if rmsfactor:
        error_factor = fhdr['rmsfac']
    else:
        error_factor = 1.0
    f1 = f0.copy()
    f1[mask] = np.nan
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
    elif len(meanbkg) > 0:
        f1 = f1 - bkg0
        print('Mean Background subtracted')

    flux_factor = 10**(-0.4 * (zeropoint - 23.9))   # factor to multiply to flux from image so that they are in mJy
    if apply_extinction == True:
        extinction_flux_factor = extinction_flux_factors[i]
    else:
        extinction_flux_factor = 1.0

    # get the bcg's total flux
    # bcg_aper_a = bcg_a * bcg_aper_ratio
    # bcg_aper_b = bcg_b * bcg_aper_ratio
    bcg_aper_b = r0
    bcg_aper_a = r0 * bcg_ratio
    bcg_aper = EllipticalAperture(position, a=bcg_aper_a, b=bcg_aper_b, theta=bcg_theta)
    mbcg_aper = bcg_aper.to_mask('center')
    ibcg_aper = mbcg_aper.to_image((naxis, naxis))
    hbcg = (ibcg_aper>0) & (~np.isnan(f1))
    dat_bcg_flux = f1[hbcg]
    bcg_flux = np.sum(dat_bcg_flux)
    f2 = f1 / gain
    dat_bcg_div_g = f2[hbcg]
    dat_bcg_w = w0[hbcg]
    bcg_err = np.sqrt(np.sum(1/dat_bcg_w + dat_bcg_div_g))
    
    tab[0][eazyfiltname] = bcg_flux * flux_factor
    tab[0][eazyerrname] = bcg_err * flux_factor
    tab[0][extfiltname] = extinction_flux_factor
    tab[0]['id'] = 0
    tab[0]['z_phot'] = z_phot
    tab[0]['z_spec'] = z_spec
    tab[0]['b1'] = 0.0
    tab[0]['b2'] = bcg_aper_b


    # fill the tables starting row 1 since row 0 is the bcg's total flux
    for rj in range(pts):
        # b1 = r0 + rj * dr
        # b2 = b1 + dr
        b1 = r1s[rj]
        b2 = r2s[rj]
        a1 = b1 * bcg_ratio
        a2 = b2 * bcg_ratio
        # position = (bcg_x, bcg_y)
        annulus = EllipticalAnnulus(position, a_in = a1, a_out=a2, b_in=b1, b_out=b2, theta=bcg_theta)
        mask_annulus = annulus.to_mask('center')
        image_annulus = mask_annulus.to_image((naxis, naxis))
        dat0 = f1[image_annulus>0]

        if method == 'med':
            dat00 = dat0[~np.isnan(dat0)]
            for ci in range(n_clipping):
                # dat00 = dat0[~np.isnan(dat0)]
                # dat1 = sigma_clip(dat0[~np.isnan(dat0)], sigma=sigma, masked=False)
                dat11 = sigma_clip(dat00, sigma=sigma, masked=False)
                dat00 = dat11
            dat1 = dat00

            med1 = np.median(dat1)
            std1 = np.std(dat1)
            len1 = len(dat1)

            med_flux = med1 * flux_factor # * extinction_flux_factor
            surface_brightness = med_flux / pixscale**2
            error = std1 * flux_factor / np.sqrt(len1) * error_factor / pixscale**2

            dat_final = surface_brightness

        elif method == 'sum':
            hdat = (image_annulus>0) & (~np.isnan(f1))
            dat1 = f1[hdat]
            npix_dat1 = len(dat1)
            # npix_dat1 = 1
            dat_sum = np.sum(dat1)
            dat_div_g = f2[hdat]
            dat_w = w0[hdat]
            error = np.sqrt(np.sum(1/dat_w + dat_div_g))/npix_dat1 * flux_factor
            dat_final = dat_sum/npix_dat1 * flux_factor

        tab[rj+1][eazyfiltname] = dat_final
        tab[rj+1][eazyerrname] = error
        tab[rj+1][extfiltname] = extinction_flux_factor
        tab[rj+1]['b1'] = b1
        tab[rj+1]['b2'] = b2


        # fill in id, seems a dumb way to do it
        if i == 0:
            tab[rj+1]['id'] = rj+1
            tab[rj+1]['z_phot'] = z_phot
            tab[rj+1]['z_spec'] = z_spec

tab['id'] = tab['id'].astype(int)


outputname = clustername + '_icls_data.ecsv'
ascii.write(tab, output=path+outputname, format='ecsv', overwrite=True)


