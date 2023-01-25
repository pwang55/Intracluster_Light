import numpy as np
from scipy.optimize import curve_fit
from astropy.io import fits, ascii
import sys
from astropy.table import Table
from scipy.integrate import simpson
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from glob import glob
# plt.ion()

spectype = 'flux'   # flux or model

script_dir = sys.path[0]
filterpath = '/Users/brianwang76/sources/90Prime/Filters/'

filters = ['1sw', '3nw', '1se', '2nw', '3ne', '2ne', '3sw', '2sw', '3se', '2se', '4nw', '4ne', '4sw', '4se', '1ne', '1nw', 
           'u', 'b', 'v', 'rc', 'ip', 'ic', 'z', 'ks', 
           'sdss_u', 'sdss_g', 'sdss_r', 'sdss_i', 'sdss_z',
           'panstarrs_g', 'panstarrs_r', 'panstarrs_i', 'panstarrs_z', 'panstarrs_y']
# nfilters = ['1sw', '3nw', '1se', '2nw', '3ne', '2ne', '3sw', '2sw', '3se', '2se', '4nw', '4ne', '4sw', '4se', '1ne', '1nw']
# clashfilters = ['u', 'ks', 'b', 'ic', 'ip', 'rc', 'v', 'z']

lam_dict = {'1sw': 5100, '3nw': 5200, '1se': 5320, '2nw': 5400,
            '3ne': 5500, '2ne': 5600, '3sw': 5680, '2sw': 5800, 
            '3se': 5890, '2se': 6000, '4nw': 6100, '4ne': 6200, 
            '4sw': 6300, '4se': 6400, '1ne': 6500, '1nw': 6600,
            'u': 3817.2, 'ks': 21574, 'b': 4448, 'ic': 7960.5,
            'ip': 7671.2, 'rc': 6505.4, 'v': 5470.2, 'z': 9028.2,
            'sdss_u': 3543, 'sdss_g': 4770, 'sdss_r': 6231, 'sdss_i': 7625, 'sdss_z': 9134,
            'panstarrs_g': 4849.11, 'panstarrs_r': 6201.19, 'panstarrs_i': 7534.96, 'panstarrs_z': 8674.18, 'panstarrs_y': 9627.77}

filt_dict = {'1sw': 'F350', '3nw': 'F351', '1se': 'F352', '2nw': 'F353', 
             '3ne': 'F354', '2ne': 'F355', '3sw': 'F356', '2sw': 'F357',
             '3se': 'F358', '2se': 'F359', '4nw': 'F360', '4ne': 'F361',
             '4sw': 'F362', '4se': 'F363', '1ne': 'F364', '1nw': 'F365',
             'u': 'F113', 'ks': 'F222', 'b': 'F78', 'ic': 'F283',
             'ip': 'F82', 'rc': 'F285', 'v': 'F79', 'z': 'F83',
             'sdss_u': 'F156','sdss_g': 'F157','sdss_r': 'F158','sdss_i': 'F159','sdss_z': 'F160',
             'panstarrs_g': 'F334', 'panstarrs_r': 'F335', 'panstarrs_i': 'F336', 'panstarrs_z': 'F337', 'panstarrs_y': 'F338'}

inv_filt_dict = {v: k for k, v in filt_dict.items()}

eazy2lam_dict = {'F350': 5100, 'F351': 5200, 'F352': 5320, 'F353': 5400,
                 'F354': 5500, 'F355': 5600, 'F356': 5680, 'F357': 5800,
                 'F358': 5890, 'F359': 6000, 'F360': 6100, 'F361': 6200,
                 'F362': 6300, 'F363': 6400, 'F364': 6500, 'F365': 6600,
                 'F113': 3817.2, 'F222': 21574, 'F78': 4448, 'F283': 7960.5,
                 'F82': 7671.2, 'F285': 6505.4, 'F79': 5470.2, 'F83': 9028.2,
                 'F156': 3543, 'F157': 4770, 'F158': 6231, 'F159': 7625, 'F160': 9134,
                 'F334': 4849.11, 'F335': 6201.19, 'F336': 7534.96, 'F337': 8674.18, 'F338': 9627.77}

lam2eazy_dict = {v: k for k, v in eazy2lam_dict.items()}


allfilts = [filt_dict[filters[i]] for i in range(len(filters))]

# Read FILTER.RES.latest.info anc convert info into index and number of lines in each filter
f = open(filterpath + 'FILTER.RES.latest.info')
lines = []
fidx = []
fnl = []
for l in f:
    lines.append(l.strip().split())
for x in range(len(lines)):
    fidx.append(int(lines[x][0]))
    fnl.append(int(lines[x][1]))
fidx = np.array(fidx)
fnl = np.array(fnl)
f.close()

# Now read the FILTER.RES.latest in order to get the filter curves of filters that are in filt_table[hfilt], save them into a dictionary filt_profiles
f = open(filterpath + 'FILTER.RES.latest')
lines = []
filt_profiles = {}
for l in f:
    lines.append(l.strip().split())
# for nth filter counting from 0, do: for x in range(sum(fnl[:n1])+fidx[n], sum(fnl[:n+1])+fidx[n]): filt.append(lines[x]), then Table(np.array(filt))
for x in range(len(allfilts)):
    name = allfilts[x]
    idx = int(name[1:])
    temp_filt = []
    n = np.where(fidx == idx)[0][0] # nth filter counting from zero
    for y in range(sum(fnl[:n]) + fidx[n], sum(fnl[:n + 1]) + fidx[n]):
        temp_filt.append(lines[y])
    filt_profiles[name] = Table(np.array(temp_filt), dtype=[int, float, float], names=['idx', 'lam', 'Response'])
f.close()

def filterlam(filt):
    '''
    Translate filter EAZY names into an array of wavelengths in AA.

    Parameters:
    -----------
    filt: list or np.array
        List of filters as EAZY names

    Returns:
    --------
    lam: np.array
        numpy array of wavelength in the same order as input EAZY filter names
    '''
    lam = np.array([eazy2lam_dict[filt[i]] for i in range(len(filt))])
    return lam




def spec_pts(filt=None, lam=None, spec=None, return_scale0=False, fnu=True):
    '''
    Convolve input spectra with filters. 
    Return data points wavelengths and fluxes (arbitrary scaling)

    If both filt and lam are not provided, then output data points will use all available filters to convolve with the spectra.

    Parameters:
    -----------
    filt: 
        List of filters with EAZY filter names (ex: 'F350') to convolve.
    lam: 
        List of wavelengths (AA) to convolve. Will be ignored if filt is provided
    spectra: 
        File name of input spectra
    return_scale0:
        If True, return the scaling factor that makes the first point 1.0 (basically the first flux)
    fnu:
        If True, convert it to f_nu. If False, retain f_lambda

    Returns:
    --------
    lam: numpy.array
        Wavelength of output datapoints
    flux: numpy.array
        Convolved fluxes (arbitrary scaling with the first flux = 1.0)

    '''

    if filt is not None:
        filts = filt
    elif lam is not None:
        filts = [lam2eazy_dict[lam[i]] for i in range(len(lam))]
    else:
        filts = list(filt_profiles.keys())

    if type(filts) == list:
        filts = np.array(filts)
    lams = [eazy2lam_dict[filts[i]] for i in range(len(filts))]
    lams = np.array(lams)

    lam_order = lams.argsort()

    spec = fits.open(spec)
    spec_loglam = spec[1].data['loglam']
    spec_lam = 10**(spec_loglam)
    spec_fl = spec[1].data[spectype]   # f_lambda
    if fnu:
        spec_fnu = spec_fl * spec_lam**2 / (3e18)
    else:
        spec_fnu = spec_fl

    fluxes = []
    for i in range(len(filts)):
        filt_profile = filt_profiles[filts[i]]
        xl = filt_profile['lam']
        response = filt_profile['Response']
        h = (spec_lam <= np.max(xl)) & (spec_lam >= np.min(xl))
        spec_lam1 = spec_lam[h]
        spec_fnu1 = spec_fnu[h]
        response_interp_func = interp1d(xl, response)
        response_interp = response_interp_func(spec_lam1)
        ave_flux = simpson(response_interp * spec_fnu1, spec_lam1) / simpson(response_interp, spec_lam1)
        fluxes.append(ave_flux)
    fluxes = np.array(fluxes)
    scale0 = fluxes[lam_order[0]]
    fluxes = fluxes / scale0
    if return_scale0:
        return(lams, fluxes, scale0)
    else:
        return(lams, fluxes)


# def output_calibrate(tab, spec, idx=0, extinction=True):
#     '''
#     Calibrate input ICL table with BCG spectra, apply extinction.

#     Parameters:
#     -----------
#     tab: astropy.table.table.Table
#         astropy table read from clustername_icl_data.ecsv
#     spec: string
#         filename of BCG spectra fits file
#     idx: int
#         Index number for the object to use to calibrate, default is 0 (the BCG)
#     extinction: bool
#         If True, apply extinction correction

#     Returns:
#     --------
#     tab: astropy.table.table.Table
#         Calibrated astropy table in EAZY compatible format

#     '''
    
#     sig = 1.0
#     cols = tab.colnames
#     fcols = cols[1:-3:3]
#     ecols = [fcols[i].replace('F','E') for i in range(len(fcols))]
#     ext_cols = [fcols[i].replace('F','ext_') for i in range(len(fcols))]
#     tab1 = tab.copy()

#     # subtract smallest value in flux
#     for i in range(len(fcols)):
#         fcol = fcols[i]
#         ecol = ecols[i]
#         idxmin = np.argmin(tab1[fcol])
#         fluxmin = tab1[fcol][idxmin]
#         errmin = tab1[ecol][idxmin]
#         tab1[fcol] = tab1[fcol] - fluxmin
#         tab1[ecol] = np.sqrt(tab1[ecol]**2 + errmin**2)   # TEMP error propagation

#     # read spectra fits file
#     slam, sflux = spec_pts(filt=fcols, spec=spec)

#     lam_order = slam.argsort()

#     flux_bcg = np.array(list(tab1[idx][fcols]))
#     err_bcg = np.array(list(tab1[idx][ecols]))

#     # the initial ratio between flux_bcg and sflux from spectra is just the first flux (the flux with the smallest wavelength), 
#     # since from spec_pts the smallest wavelength flux is scaled to 1.0
#     ratio0 = flux_bcg[lam_order[0]]

#     def scalefunc(a,x):
#         return a * x

#     a0, var0 = curve_fit(scalefunc, sflux*ratio0, flux_bcg, sigma=err_bcg)
#     diff = sflux * ratio0 * a0 - flux_bcg
#     mean_diff = np.mean(diff)
#     std_diff = np.std(diff)
#     diff_h = (diff < mean_diff + sig*std_diff) & (diff > mean_diff - sig*std_diff)

#     a1, var1 = curve_fit(scalefunc, sflux[diff_h]*ratio0*a0, flux_bcg[diff_h], sigma=err_bcg[diff_h])
#     aa = a1[0] * ratio0 * a0[0]    # this is the ratio for scaling spectra convolved fluxes
#     sig_aa = np.sqrt(var1[0])

#     sflux1 = sflux * aa
#     rcals = sflux1 / flux_bcg   # calibration ratios of each data point
#     drcals = np.sqrt(((sflux/flux_bcg)*sig_aa)**2 + ((aa*sflux/flux_bcg**2)*err_bcg)**2)    # TEMP error propagation
    
#     for i in range(len(fcols)):
#         fcol = fcols[i]
#         ecol = ecols[i]
#         rcal = rcals[i]
#         drcal = drcals[i]
#         tab1[fcol] = tab1[fcol] * rcal
#         for j in range(len(tab1[fcol])):
#             tab1[ecol][j] = np.sqrt((rcal*tab1[ecol][j])**2 + (tab1[fcol][j]*drcal)**2)    # TEMP error propagation

#     # apply extinction
#     if extinction:
#         for i in range(len(fcols)):
#             fcol = fcols[i]
#             ecol = ecols[i]
#             ext_col = ext_cols[i]
#             tab1[fcol] = tab1[fcol] * tab1[ext_col][0]
#             tab1[ecol] = tab1[ecol] * tab1[ext_col][0]

#     cols1 = [cols[i] for i in range(len(cols)) if cols[i][:3]!='ext']
#     cols1 = cols1[:-1]
#     tab2 = tab1[cols1]
#     return(tab2)


def output(tab, extinction=True, zspec=True, zphot=True, clashonly=False, ave=False):
    '''
    Apply extinction and output table in EAZY compatible format.

    Parameters:
    -----------
    tab: astropy.table.table.Table
        astropy table read from clustername_icl_data.ecsv
    extinction: bool
        If True, apply extinction correction
    zspec: bool
        If False, the zspec column will be removed
    zphot: bool
        If False, the zphot column will be removed
    clashonly: bool
        If True, output a table with only CLASH data
    ave: bool
        If True, the zero point to subtract is calculated from average of last 10 data points
        default=False will find the smallest value to subtract

    Returns:
    --------
    tab: astropy.table.table.Table
        astropy table in EAZY compatible format

    '''
    
    avepts = 1

    cols = tab.colnames
    # fcols = cols[1:-4:3]
    # figure out fcols
    for j in range(len(cols)):
        idx = len(cols) - j - 1
        colj = cols[idx]
        if colj.split('_')[0] == 'ext' and cols[idx-1][0] == 'E' and cols[idx-2][0] == 'F':
            end_fcol_idx = idx
            break
    fcols = cols[1:end_fcol_idx+1:3]

    ecols = [fcols[i].replace('F','E') for i in range(len(fcols))]
    ext_cols = [fcols[i].replace('F','ext_') for i in range(len(fcols))]
    tab1 = tab.copy()

    lams = [eazy2lam_dict[fcols[i]] for i in range(len(fcols))]
    lams = np.array(lams)
    clash_idx = np.where(np.diff(lams)<0)[0][0] + 1
    nfcols = fcols[:clash_idx]
    necols = ecols[:clash_idx]

    for i in range(len(fcols)):
        fcol = fcols[i]
        ecol = ecols[i]

        if not ave:
            # subtract smallest value in flux
            idxmin = np.argmin(tab1[fcol])
            fluxmin = tab1[fcol][idxmin]
            errmin = tab1[ecol][idxmin]
            tab1[ecol] = np.sqrt(tab1[ecol]**2 + errmin**2)   # TEMP error propagation
        else:
            fluxmin = np.mean(tab1[fcol][-avepts:])
            dfluxmin = 1/avepts*np.sqrt(np.sum(tab1[ecol][-avepts:]**2))

        tab1[fcol] = tab1[fcol] - fluxmin
        tab1[ecol] = np.sqrt(tab1[ecol]**2 + dfluxmin**2)

    # apply extinction
    if extinction:
        for i in range(len(fcols)):
            fcol = fcols[i]
            ecol = ecols[i]
            ext_col = ext_cols[i]
            tab1[fcol] = tab1[fcol] * tab1[ext_col][0]
            tab1[ecol] = tab1[ecol] * tab1[ext_col][0]

    # cols1 = [cols[i] for i in range(len(cols)) if cols[i][:3]!='ext']

    # remove any column after z_spec
    idx_z_spec = np.where(np.array(cols)=='z_spec')[0][0]
    cols1 = cols[:idx_z_spec+1]

    tab2 = tab1[cols1]
    # remove extinction columns
    for j in range(len(ext_cols)):
        tab2.remove_column(ext_cols[j])
    
    # cols1 = cols1[:-1]
    # tab2 = tab1[cols1]
    if not zspec:
        tab2.remove_column('z_spec')
    if not zphot:
        tab2.remove_column('z_phot')
    if clashonly:
        tab2.remove_columns(nfcols)
        tab2.remove_columns(necols)

    return(tab2)


def plotbcg(tab, spec, idx=0, bcg=True):
    '''
    Calculate the scaling factor for BCG spectra and plot both measured fluxes and BCG convolved fluxes.

    Parameters:
    -----------
    tab: astropy.table.table.Table
        astropy table read from clustername_icl_data.ecsv or in EAZY compatible format
    spec: string
        filename of BCG spectra fits file
    idx: int
        index for the object to be used, idx=0 is the BCG, 1 is the first ICL annulus

    Returns:
    --------
    pyplot

    '''
    
    sig = 1.0
    cols = tab.colnames
    if cols[3][:3] == 'ext':
        for j in range(len(cols)):
            idxj = len(cols) - j - 1
            colj = cols[idxj]
            if colj.split('_')[0] == 'ext' and cols[idxj-1][0] == 'E' and cols[idxj-2][0] == 'F':
                end_fcol_idx = idxj
                break
        fcols = cols[1:end_fcol_idx+1:3]
    else:
        for j in range(len(cols)):
            idxj = len(cols) - j - 1
            # colj = cols[idx]
            if cols[idxj][0] == 'E' and cols[idxj-1][0] == 'F':
                end_fcol_idx = idxj
                break
        fcols = cols[1:end_fcol_idx+1:2]

    ecols = [fcols[i].replace('F','E') for i in range(len(fcols))]
    ext_cols = [fcols[i].replace('F','ext_') for i in range(len(fcols))]
    tab1 = tab.copy()

    # subtract smallest value in flux
    for i in range(len(fcols)):
        fcol = fcols[i]
        ecol = ecols[i]
        idxmin = np.argmin(tab1[fcol])
        fluxmin = tab1[fcol][idxmin]
        errmin = tab1[ecol][idxmin]
        tab1[fcol] = tab1[fcol] - fluxmin
        tab1[ecol] = np.sqrt(tab1[ecol]**2 + errmin**2)   # TEMP error propagation

    # read spectra fits file
    slam, sflux, scale0 = spec_pts(filt=fcols, spec=spec, return_scale0=True)
    clash_idx = np.where(np.diff(slam)<0)[0][0] + 1

    lam_order = slam.argsort()

    flux_bcg = np.array(list(tab1[idx][fcols]))
    err_bcg = np.array(list(tab1[idx][ecols]))

    # the initial ratio between flux_bcg and sflux from spectra is just the first flux (the flux with the smallest wavelength), 
    # since from spec_pts the smallest wavelength flux is scaled to 1.0
    ratio0 = flux_bcg[lam_order[0]]

    def scalefunc(a,x):
        return a * x

    a0, var0 = curve_fit(scalefunc, sflux*ratio0, flux_bcg, sigma=err_bcg)
    diff = sflux * ratio0 * a0 - flux_bcg
    mean_diff = np.mean(diff)
    std_diff = np.std(diff)
    diff_h = (diff < mean_diff + sig*std_diff) & (diff > mean_diff - sig*std_diff)

    a1, var1 = curve_fit(scalefunc, sflux[diff_h]*ratio0*a0, flux_bcg[diff_h], sigma=err_bcg[diff_h])
    aa = a1[0] * ratio0 * a0[0]    # this is the ratio for scaling spectra convolved fluxes

    f = fits.open(spec)
    slam0 = 10**(f[1].data['loglam'])
    flux = f[1].data['flux'] * slam0**2 / 3e18 / scale0
    model = f[1].data['model'] * slam0**2 / 3e18 / scale0
    f.close()

    plt.figure()
    # fig=plt.figure()
    if bcg:
        plt.plot(slam0, aa * flux, alpha=0.3, linewidth=1, label='SDSS spectra')
        plt.plot(slam0, aa * model, alpha=0.6, linewidth=1, label='SDSS best fit')
        plt.plot(slam, aa * sflux, 'x',alpha=0.8, label='Spectra convolved')
    # plt.errorbar(slam[lam_order], flux_bcg[lam_order], err_bcg[lam_order], fmt='o',alpha=0.8, label='Measured')
    plt.errorbar(slam[:clash_idx], flux_bcg[:clash_idx], err_bcg[:clash_idx], fmt='o',alpha=0.8, label='Measured')
    plt.errorbar(slam[clash_idx:], flux_bcg[clash_idx:], err_bcg[clash_idx:], fmt='o',alpha=0.95)
    plt.ylim(0, 1.4*np.max(flux_bcg))
    plt.grid()
    plt.legend()


def readfast(filename): # TODO
    '''
    Read FAST++ output file into table.

    Parameters:
    -----------
    filename: string

    Returns:
    --------
    astropy.table.table.Table
    '''

    t = ascii.read(filename, header_start=16)


def fitted(filename, clash=False):
    '''
    Plot fitted results.
    tab and fcols are optional. 
    If either one is supplied, narrowband filters and CLASH filters will have different color.

    Parameters:
    -----------
    filename: string
        The filename of the fitted SED
    clash: bool
        Use clash=True if the file only contains CLASH data

    Returns:
    --------
    pyplot

    '''

    sed = ascii.read(filename, names=['w1','f1'])
    fit = ascii.read(filename.replace('.fit','.input_res.fit'), names=['w1','fmodel','fobs','eobs'])

    if not clash:
        clash_idx = np.where(np.diff(fit['w1'])<0)[0][0] + 1


    plt.figure()
    plt.plot(sed['w1'], sed['f1'], alpha=0.4, label='Fitted SED')
    plt.plot(fit['w1'], fit['fmodel'], 'x', label='Fitted Flux')
    if not clash:
        plt.errorbar(fit['w1'][:clash_idx], fit['fobs'][:clash_idx], fit['eobs'][:clash_idx], fmt='o', label='Observed')
        plt.errorbar(fit['w1'][clash_idx:], fit['fobs'][clash_idx:], fit['eobs'][clash_idx:], fmt='o', label='Observed')
    else:
        plt.errorbar(fit['w1'], fit['fobs'], fit['eobs'], fmt='o', label='Observed')
        
    plt.xlim(3000,9500)
    plt.grid()
    plt.title(filename.split('/')[-1])
    #plt.xlabel('wavelength [AA]')

def profiles(tab):
    '''
    Plot ICL profiles.

    Parameters:
    -----------
    tab: astropy table
        Original table from clusternmae_icl_data.ecsv
    
    Returns:
    --------
    pyplot

    '''

    c1 = 'steelblue'
    c2 = 'salmon'

    tab2 = output(tab)
    r1 = tab['b1'][1:]
    r2 = tab['b2'][1:]
    r = (r1 + r2)/2

    cols = tab.colnames
    for j in range(len(cols)):
        idx = len(cols) - j - 1
        colj = cols[idx]
        if colj.split('_')[0] == 'ext' and cols[idx-1][0] == 'E' and cols[idx-2][0] == 'F':
            end_fcol_idx = idx
            break
    fcols = cols[1:end_fcol_idx+1:3]
    ecols = [fcols[i].replace('F','E') for i in range(len(fcols))]
    cols = tab2.colnames[1:-1]
    lams = filterlam(fcols)
    clash_idx = np.where(np.diff(lams)<0)[0][0] + 1

    fig, ax = plt.subplots(1,2, figsize=(12,6))

    tab_nf = tab[fcols[:clash_idx]][1:]
    tab2_nf = tab2[fcols[:clash_idx]][1:]
    tab_clashf = tab[fcols[clash_idx:]][1:]
    tab2_clashf = tab2[fcols[clash_idx:]][1:]

    tab_ne = tab[ecols[:clash_idx]][1:]
    tab2_ne = tab2[ecols[:clash_idx]][1:]
    tab_clashe = tab[ecols[clash_idx:]][1:]
    tab2_clashe = tab2[ecols[clash_idx:]][1:]

    fcols_n = fcols[:clash_idx]
    fcols_clash = fcols[clash_idx:]
    ecols_n = ecols[:clash_idx]
    ecols_clash = ecols[clash_idx:]

    for j in range(len(tab_nf.colnames)):
        fcol = fcols_n[j]
        ecol = ecols_n[j]
        ax[0].errorbar(r, tab_nf[fcol], tab_ne[ecol], fmt='.-', color=c1, alpha=0.4)
        ax[1].errorbar(r, tab2_nf[fcol], tab2_ne[ecol], fmt='.-', color=c1, alpha=0.4)
        
    for j in range(len(tab_clashf.colnames)):
        fcol = fcols_clash[j]
        ecol = ecols_clash[j]
        ax[0].errorbar(r, tab_clashf[fcol], tab_clashe[ecol], fmt='.-', color=c2, alpha=0.6)
        ax[1].errorbar(r, tab2_clashf[fcol], tab2_clashe[ecol], fmt='.-', color=c2, alpha=0.6)

    ax[0].grid()
    ax[1].grid()
    ax[0].set_title('original')
    ax[1].set_title('processed')
    ax[0].set_xlabel('pix')
    ax[1].set_xlabel('pix')
    fig.tight_layout()

