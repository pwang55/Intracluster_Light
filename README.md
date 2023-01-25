

**1. Create star mask**

`$ python path_to_script/starmask.py master_stack_clustername_cutout.fits`

Requires:
- master_stack_clustername_cutout.fits
- master_stack_clustername_cutout.wt.fits
- clustername_star_sdss(panstarrs)_radec.csv

Creates:
- clustername_starmask.csv

Visually inspect the star apertures by open fits file in DS9, use catalog tools
to load the generated csv file and choose $radius as marker circle radius.

Can work on any fits file with weight image, but ideally master_stack_clustername_cutout.fits works best since it's the closest to each image.

------------------------------------

**2. Create all PSFs, and visually check potential final PSFs**

`$ path_to_script/make_psf.sh cutoutlist.txt`
`$ python path_to_script/match_psf.py filelist.txt`

cutoutlist.txt contains all cutout_stack_clustername_filter.fits
filelist.txt contains all above AND master_stack_clustername_cutout.fits

------------------------------------

**3. Trim the master stack cutout image**

`$ python path_to_script/master_trimconv.py master_stack_clustername_cutout.fits`

Requires:
- clustername_starmask.csv
- master_stack_clustername_cutout.fits
- master_stack_clustername_cutout.wt.fits
  
Creates:
- master_stack_clustername_cutout.trim.fits
- master_stack_clustername_cutout.trim.wt.fits
- master_stack_clustername_cutout.trimconv.fits
- master_stack_clustername_cutout.trimconv.wt.fits

Visually inspect the final images, move *trimconv*fits to trimconvs/.

------------------------------------

**4. Estimate BCG size and shape**

`$ python path_to_script/bcg_shape.py master_stack_clustername_cutout.trimconv.fits`

Requires:
- master_stack_clustername_cutout.trimconv.fits
- master_stack_clustername_cutout.trimconv.wt.fits
  
Creates:
- bcg_shape_clustername.csv

Changing parameters of this code to get the best setting.

------------------------------------

**5. Trim all cutout_stack_clustername_filter.fits**

`$ python path_to_script/trimconv.py cutoutlist.txt`

Requires:
- master_stack_clustername_cutout.trimconv.fits
- clustername_starmask.csv
- cutout_stack_clustername_filter.fits
- cutout_stack_clustername_filter.wt.fits
  
Creates:
- cutout_stack_clustername_filter.trim.fits
- cutout_stack_clustername_filter.trim.wt.fits

Move `*trim*` to trims/ folder.

------------------------------------

**6. Calculate zero point with trim images**

1. In trims/, copy all files from stuff_for_photometry to here, `ls *trim.fits > trimlist.txt`
2. Copy a set of trim.fits and trim.wt.fits to trims/photometry/, rename them to .fits and .wt.fits (`rename 's/.trim.fits/.fits/g' *trim.fits` and similar for .trim.wt.fits)
3. Copy all from stuff_for_photometry/ to trims/photometry/, creates cutoutlist.txt, then run:
    `make_each_catalog.sh cutoutlist.txt sdss`
    `combine_catalog_calibration.py sdss extinction=false`
4. Change filename *b_values_sdss(panstarrs).ecsv* to *b_values_sdss(panstarrs)_noext_clustername.ecsv*, then move it to trimconvs/

------------------------------------

**7. Create .trimconv.fits and .trimconv.bkg.fits**

1. Back in trims/, copy clustername_starmask.csv to trims/, run `python path_to_script/conv.py trimlist.txt`, move .trimconv*fits to trimconvs/

Requires:
- clustername_starmask.csv
- cutout_stack_clustername_filter.trim.fits
- cutout_stack_clustername_filter.trim.wt.fits
- trimlist.txt

2. Back in trims/, run `python path_to_script/background.py trimlist.txt bsize=(desired bsize) fsize=(desired fsize)`, move *bkg.fits to trimconvs/


------------------------------------

**8. Trim and convolve all CLASH images**

`$ python path_to_script/clash_trimconv.py clashlist.txt`

Requires:
- master_stack_clustername_cutout.trimconv.fits
- clashimage.fits
- clashimage.wt.fits
- clustername_starmask.csv

Creates:
- clashimage.trim.fits
- clashimage.trim.wt.fits
- clashimage.trim_psf.psf
- clashimage.trim.psfs.fits
- clashimage.trimconv.fits
- clashimage.trimconv.wt.fits

Visually examine each clashimage.trimconv.fits.
Move `clash*trimconv*fits` to trimconvs/ folder.

------------------------------------

**9. Create object masks from master_stack_clustername_cutout.trimconv.fits**

`$ python path_to_script/mask.py master_stack_clustername_cutout.trimconv.fits`

Requires:
- master_stack_clustername_cutout.trimconv.fits
  
Creates:
- master_stack_clustername_cutout.trimconv.unsharp.fits
- master_stack_clustername_cutout.trimconv.aper.fits
- master_stack_clustername_cutout.trimconv.iclmask.cat
- master_stack_clustername_cutout.trimconv.iclmask.fits

Visually inspect aper.fits and iclmask.fits and adjust parameters accordingly.

------------------------------------

**10. Get zero points for clash images**

Read CLASH data readme files and write a csv file with filename *zeropoint_clash_clustername.csv* in the following format:
(use tab between entries and all lower case)
```
# filter        zeropoint
b       27.33
v       26.49
rc      27.59
ic      27.45
ip      27.81
z       27.18
```

------------------------------------

**11. Create radial profile catalog**

`$ python path_to_script/radial_profile.py filelist.txt master_stack_clustername_cutout.iclmask.fits`

Requires:
- All fits and wt.fits needed for measuring ICL, and filelist.txt contain names of all fits
- master_stack_clustername_cutout.iclmask.fits
- b_values_sdss(panstarrs)_noext_clustername.ecsv
- zeropoint_clash_clustername.csv

Creates:
- clustername_icls_data.ecsv


