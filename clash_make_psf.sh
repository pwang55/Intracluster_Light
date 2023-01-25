#!/bin/bash


usage='

    In script folder:
    $ ./make_psf.sh path_to_files/clashlist.txt (25)
    or
    $ ./make_psf.sh path_to_files/clash_subaru_abell611_rc.fits (25)

    In data folder:
    $ path_to_script/make_psf.sh clashlist.txt (25)
    or
    $ path_to_script/make_psf.sh clash_subaru_abell611_rc.fits (25)


This script runs SExtractor and PSFEx to generate _psf.psf file for input files.
Second argunet is the PSF size and default is 25 by 25 if left empty.

'

# If no argument or only one is given, print doc and exit
if (( "$#" < 1 )); then
	echo "$usage"
	exit 1
fi


# Get the absolute directory path of this script so that it can find extra/files
script_dir=$(cd `dirname $0` && pwd)

list_in=$1
psfsize=25
if [ -n "$2" ]; then
    psfsize=$2
fi

path=""

# If argument 1 is a full path, path variable will be set to the path excluding the list name
len_list=`echo $list_in | awk '{n=split($1,a,"/"); print n}'`
if (( $len_list > 1 )); then
        path=`dirname $list_in`
        path=$path/
fi

# # filename=`echo $file_in | sed -e 's/\.fits//g'`
# # conv_filter=gauss_1.5_3x3.conv
# # Get the initial catalog for fwhm estimation
# # sex -c ${script_dir}/extra/find_fwhm.sex ${path}$file -CATALOG_NAME ${path}${filename}_fwhm.cat -PARAMETERS_NAME ${script_dir}/extra/fwhm.param -FILTER_NAME ${script_dir}/extra/gauss_4.0_7x7.conv -WEIGHT_IMAGE ${path}${filename}.wt.fits -VERBOSE_TYPE QUIET
# # Get the catalog for psf
# # sex -c ${script_dir}/extra/psf.sex ${path}$file_in -CATALOG_NAME ${path}${filename}_psf.cat -CATALOG_TYPE FITS_LDAC -PARAMETERS_NAME ${script_dir}/extra/psf.param -FILTER_NAME ${script_dir}/extra/${conv_filter} -WEIGHT_IMAGE ${path}${filename}.wt.fits -VERBOSE_TYPE QUIET
# sex -c ${script_dir}/extra/psf.sex ${path}$file_in -CATALOG_NAME ${path}${filename}_psf.cat -CATALOG_TYPE FITS_LDAC -PARAMETERS_NAME ${script_dir}/extra/psf.param -FILTER N -WEIGHT_IMAGE ${path}${filename}.wt.fits -VERBOSE_TYPE QUIET
# psfex -c ${script_dir}/extra/psf.psfex ${path}${filename}_psf.cat -PSF_SIZE ${psfsize}
# rm ${path}${filename}_psf.cat

# Check if the input is a .fits or .txt
extension=`echo $list_in | awk '{n=split($1,a,"."); print a[n]}'`
if [ $extension = txt ]; then
    for file in `cat $list_in`; do
        echo -e "\n$file"
        filename=`echo $file | sed -e 's/\.fits//g'`
        # Get the initial catalog for fwhm estimation
        # sex -c ${script_dir}/extra/find_fwhm.sex ${path}$file -CATALOG_NAME ${path}${filename}_fwhm.cat -PARAMETERS_NAME ${script_dir}/extra/fwhm.param -FILTER_NAME ${script_dir}/extra/gauss_4.0_7x7.conv -WEIGHT_IMAGE ${path}${filename}.wt.fits -VERBOSE_TYPE QUIET
        # conv_filter=$(python ${script_dir}/extra/fwhm.py ${path}${filename}_fwhm.cat 2>&1)
        # rm ${path}${filename}_fwhm.cat
        # Get the catalog for psf
        sex -c ${script_dir}/extra/psf.sex ${path}$file -CATALOG_NAME ${path}${filename}_psf.cat -CATALOG_TYPE FITS_LDAC -PARAMETERS_NAME ${script_dir}/extra/psf.param -FILTER N -WEIGHT_IMAGE ${path}${filename}.wt.fits -VERBOSE_TYPE QUIET
        psfex -c ${script_dir}/extra/psf.psfex ${path}${filename}_psf.cat -PSF_SIZE ${psfsize}
        rm ${path}${filename}_psf.cat
    done
elif [ $extension = fits ]; then
    file=`echo $list_in | awk '{n=split($1,a,"/"); print a[n]}'`
    echo -e "\n$file"
    filename=`echo $file | sed -e 's/\.fits//g'`
    # Get the initial catalog for fwhm estimation
    # sex -c ${script_dir}/extra/find_fwhm.sex ${path}$file -CATALOG_NAME ${path}${filename}_fwhm.cat -PARAMETERS_NAME ${script_dir}/extra/fwhm.param -FILTER_NAME ${script_dir}/extra/gauss_4.0_7x7.conv -WEIGHT_IMAGE ${path}${filename}.wt.fits -VERBOSE_TYPE QUIET
    # conv_filter=$(python ${script_dir}/extra/fwhm.py ${path}${filename}_fwhm.cat 2>&1)
    # rm ${path}${filename}_fwhm.cat
    # Get the catalog for psf
    sex -c ${script_dir}/extra/psf.sex ${path}$file -CATALOG_NAME ${path}${filename}_psf.cat -CATALOG_TYPE FITS_LDAC -PARAMETERS_NAME ${script_dir}/extra/psf.param -FILTER N -WEIGHT_IMAGE ${path}${filename}.wt.fits -VERBOSE_TYPE QUIET
    psfex -c ${script_dir}/extra/psf.psfex ${path}${filename}_psf.cat -PSF_SIZE ${psfsize}
    rm ${path}${filename}_psf.cat
else
    echo "file type unknown!"
fi




