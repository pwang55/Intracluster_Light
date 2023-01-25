import time
from astropy.io import fits, ascii
from astropy.coordinates import SkyCoord
from astropy.table import Table
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
#plt.ion()
from photutils import MedianBackground,SExtractorBackground,Background2D
from photutils.aperture import CircularAnnulus, aperture_photometry, CircularAperture, EllipticalAnnulus, EllipticalAperture
from astropy.stats import SigmaClip,sigma_clip,sigma_clipped_stats
from astropy.visualization import simple_norm
from astropy.wcs import WCS
from astropy.nddata.utils import Cutout2D
from astropy import units as u
import subprocess
import os
import sys


script_dir = sys.path[0] + '/'

path_filelist = sys.argv[1]
filelist = path_filelist.split('/')[-1]
path = path_filelist[:-len(filelist)]



# prior knowledge of where the brightest star is (for zwicky1953)
ra_bstar=132.6312055
dec_bstar=36.1070771
c_bstar=SkyCoord(ra_bstar,dec_bstar,unit='deg')
r_bstar=160 # radius for masking brightest star
dr_bstar=20 # annulus width for calculating sky median value near brightest star


# BCG of abell611
ra_bcg = 120.2368074
dec_bcg = 36.0565097
cbcg = SkyCoord(ra_bcg, dec_bcg, unit='deg')

threshold=5 # when comparing the skybackground and annulus background median, criteria is median(annulus)- threshold * std of the mean < median(sky) + threshold * std of the mean
peak_thres=10000
# peak_thres=45   # for clash images
large_boxratio=18 # this*a as halfbox
small_boxratio=12 # this*a as halfbox
large_aratio=4.
small_aratio=3.
max_ra=80 # max allowed radius (semimajor axis a) in pixel

files=[]
filters=['1sw', '3nw', '1se', '2nw', '3ne', '2ne', '3sw', '2sw', '3se', '2se', '4nw', '4ne', '4sw', '4se', '1ne', '1nw']

# for i in range(len(filters)):
# #     if os.path.isfile('cutout_stack_*{}.fits'.format(filters[i])):
#     if len(glob('cutout_stack_*{}.fits'.format(filters[i])))>0:
#         file0=glob('cutout_stack_*{}.fits'.format(filters[i]))[0]
#         files.append(file0)

with open(path_filelist) as f:
    for l in f:
        files.append(l.strip())

# print(path_filelist)

# run source extractor # TEMP disable
# for i in range(len(files)):
#     fname=files[i]
#     wtname=fname.replace('.fits','.wt.fits')
#     catname=fname.replace('.fits','.cat')
#     subprocess.run(['sex','-c', script_dir + 'extra/default.sex',fname,'-WEIGHT_IMAGE',wtname,'-CATALOG_NAME',catname,'-VERBOSE_TYPE','QUIET', '-PARAMETERS_NAME', script_dir + 'extra/param.param', '-FILTER_NAME', script_dir + 'extra/gauss_3.0_7x7.conv'])

for i in range(len(files)):
# for i in range(1,2):
    fname=files[i]
    wtname=fname.replace('.fits','.wt.fits')
    catname=fname.replace('.fits','.cat')
    
    cat=ascii.read(catname)
#     stars=ascii.read('zwicky1953_star_sdss_radec.csv')
#     gals=ascii.read('zwicky1953_gal_sdss_radec.csv')
    f=fits.open(fname)
    f0=f[0].data
    f3=f0.copy()
    x=cat['X_IMAGE']
    y=cat['Y_IMAGE']

    # figure out the brightest star's id
    # autoflux=cat['FLUX_AUTO']
    # id_brightest=np.argmax(autoflux)

    coords=SkyCoord(cat['ALPHA_J2000'],cat['DELTA_J2000'],unit='deg')
    # prior knowledge of where the brightest star is
#     ra_bstar=132.6312055
#     dec_bstar=36.1070771
#     c_bstar=SkyCoord(ra_bstar,dec_bstar,unit='deg')
    # idx,d2d,d3d=c_bstar.match_to_catalog_sky(coords)
    idx,d2d,d3d=cbcg.match_to_catalog_sky(coords)
    id_brightest=int(idx)
    
    # print(id_brightest)
    # r_bstar=160 # radius for masking brightest star
    # dr_bstar=20 # annulus width for calculating sky median value near brightest star

    for gi in range(len(x)):
        #r=5.0
        #rhalfbox=15
        cx=x[gi].astype(int)
        cy=y[gi].astype(int)
        obj=cat[gi]
        peak=obj['FLUX_MAX']
        min_d2edge=min(cx,cy,4400-cx,4400-cy)
        if f3[cy,cx]==0:
            pass
        elif gi!=id_brightest:
        # else:
            dx=x[gi]-cx
            dy=y[gi]-cy
            obj=cat[gi]
            a=obj['A_IMAGE']
            b=obj['B_IMAGE']
            theta=obj['THETA_IMAGE']*np.pi/180
            peak=obj['FLUX_MAX']
            if peak>peak_thres:
                aratio=large_aratio
                boxratio=large_boxratio
            else:
                aratio=small_aratio
                boxratio=small_boxratio
    #         aratio=4
    #         boxratio=20
            ra=min(aratio*a,max_ra)
            rb=min(aratio*b,max_ra)
            b2aratio=rb/ra
            #rhalfbox=20
            #halfbox=int(rhalfbox*a)
            halfbox=min(int(boxratio*a),min_d2edge-1)
            ft=f3[cy-halfbox:cy+halfbox+1,cx-halfbox:cx+halfbox+1]
            position=(halfbox+dx,halfbox+dy)

            if peak>peak_thres:
                ra=(ra+rb)/2
                aa=CircularAnnulus(position,r_in=ra,r_out=ra+5)
            else:
                aa=EllipticalAnnulus(position,a_in=ra,a_out=ra+5,b_out=(ra+5)*b2aratio,theta=theta)

            ftdat=sigma_clip(ft[ft>0],sigma=2.5,masked=False)
            med1=np.median(ftdat)
            std1=np.std(ftdat)
            stdmean1=std1/np.sqrt(len(ftdat))
            thres=med1+3*stdmean1
            dat=aa.to_mask('center').multiply(ft)
            dat1=sigma_clip(dat[dat>0],sigma=2.5,masked=False)
            dat1lower=np.median(dat1)-3*np.std(dat1)/np.sqrt(len(dat1))
            r_increment=2
            while dat1lower>thres and ra<max_ra:
                #print(r,dat1lower,thres)
                ra=ra+r_increment
                rb=ra*b2aratio
                halfbox=min(halfbox+r_increment+1,min_d2edge-1)
                ft=f3[cy-halfbox:cy+halfbox+1,cx-halfbox:cx+halfbox+1]
                position=(halfbox+dx,halfbox+dy)
                if peak>peak_thres:
                    aa=CircularAnnulus(position,r_in=ra,r_out=ra+5)
                else:
                    aa=EllipticalAnnulus(position,a_in=ra,a_out=ra+5,b_out=(ra+5)*b2aratio,theta=theta)
                ftdat=sigma_clip(ft[ft>0],sigma=2.5,masked=False)
                med1=np.median(ftdat)
                std1=np.std(ftdat)
                stdmean1=std1/np.sqrt(len(ftdat))
                thres=med1+3*stdmean1
                dat=aa.to_mask('center').multiply(ft)
                dat1=sigma_clip(dat[dat>0],sigma=2.5,masked=False)
                dat1lower=np.median(dat1)-3*np.std(dat1)/np.sqrt(len(dat1))
            #ap=EllipticalAperture(position,r*a,r*b,theta*np.pi/180)
            if peak>peak_thres:
                ap=CircularAperture(position,ra)
            else:
                ap=EllipticalAperture(position,ra,rb,theta)
            am=ap.to_mask(method='center')
            mask=am.to_image((2*halfbox+1,2*halfbox+1))
            mlength=len(mask[mask>0])
            #mean1,med1,std1=sigma_clipped_stats(ft[ft>0],sigma=2.5)
            # TEMP fill in np.nan instead of med and std
            # ft[mask>0]=np.random.normal(med1,std1,mlength)
            ft[mask>0]=0.0
            f3[cy-halfbox:cy+halfbox+1,cx-halfbox:cx+halfbox+1]=ft

    # TEMP disable masking brightest star
    # # mask brightest star
    # obj=cat[id_brightest]
    # cxb=obj['X_IMAGE']
    # cyb=obj['Y_IMAGE']
    # # halfbox=300
    # # ft=f3[int(cyb)-halfbox:int(cyb)+halfbox+1,int(cxb)-halfbox:int(cxb)+halfbox+1]
    # apbstar=CircularAperture((int(cxb),int(cyb)),r_bstar)
    # aabstar=CircularAnnulus((int(cxb),int(cyb)),r_bstar,r_bstar+dr_bstar)
    # imapbstar=apbstar.to_mask('center').to_image((4400,4400))
    # dat=aabstar.to_mask('center').multiply(f3)
    # mean0,med0,std0=sigma_clipped_stats(dat[dat>0],sigma=2.5)
    # f3[imapbstar>0]=np.random.normal(med0,std0,len(imapbstar[imapbstar>0]))

    # save to fits file
    hduI=fits.PrimaryHDU()
    hduI.data=f3
    hduI.header=f[0].header
    hduA=fits.HDUList([hduI])
    hduA.writeto(fname.replace('.fits','.noobj.fits'),overwrite=True)
    


