#!/bin/bash
asegstats2table --subjects `cat ../list_fs` --meas volume --skip --statsfile wmparc.stats --all-segs --tablefile group_wmparc_stats.txt 
asegstats2table --subjects `cat ../list_fs` --meas volume --skip --tablefile group_aseg_stats.txt
aparcstats2table --subjects `cat ../list_fs` --hemi lh --meas volume --skip --tablefile group_aparc_volume_lh.txt
aparcstats2table --subjects `cat ../list_fs` --hemi lh --meas thickness --skip --tablefile group_aparc_thickness_lh.txt
aparcstats2table --subjects `cat ../list_fs` --hemi lh --meas area --skip --tablefile group_aparc_area_lh.txt
aparcstats2table --subjects `cat ../list_fs` --hemi lh --meas meancurv --skip --tablefile group_aparc_meancurv_lh.txt
aparcstats2table --subjects `cat ../list_fs` --hemi rh --meas volume --skip --tablefile group_aparc_volume_rh.txt 
aparcstats2table --subjects `cat ../list_fs` --hemi rh --meas thickness --skip --tablefile group_aparc_thickness_rh.txt 
aparcstats2table --subjects `cat ../list_fs` --hemi rh --meas area --skip --tablefile group_aparc_area_rh.txt
aparcstats2table --subjects `cat ../list_fs` --hemi rh --meas meancurv --skip --tablefile group_aparc_meancurv_rh.txt
aparcstats2table --hemi lh --subjects `cat ../list_fs` --parc aparc.a2009s --meas volume --skip -t group_lh.a2009s.volume.txt
aparcstats2table --hemi lh --subjects `cat ../list_fs` --parc aparc.a2009s --meas thickness --skip -t group_lh.a2009s.thickness.txt
aparcstats2table --hemi lh --subjects `cat ../list_fs` --parc aparc.a2009s --meas area --skip -t group_lh.a2009s.area.txt
aparcstats2table --hemi lh --subjects `cat ../list_fs` --parc aparc.a2009s --meas meancurv --skip -t group_lh.a2009s.meancurv.txt
aparcstats2table --hemi rh --subjects `cat ../list_fs` --parc aparc.a2009s --meas volume --skip -t group_rh.a2009s.volume.txt
aparcstats2table --hemi rh --subjects `cat ../list_fs` --parc aparc.a2009s --meas thickness --skip -t group_rh.a2009s.thickness.txt
aparcstats2table --hemi rh --subjects `cat ../list_fs` --parc aparc.a2009s --meas area --skip -t group_rh.a2009s.area.txt
aparcstats2table --hemi rh --subjects `cat ../list_fs` --parc aparc.a2009s --meas meancurv --skip -t group_rh.a2009s.meancurv.txt
aparcstats2table --hemi lh --subjects `cat ../list_fs` --parc BA --meas volume --skip -t group_lh.BA.volume.txt
aparcstats2table --hemi lh --subjects `cat ../list_fs` --parc BA --meas thickness --skip -t group_lh.BA.thickness.txt
aparcstats2table --hemi lh --subjects `cat ../list_fs` --parc BA --meas area --skip -t group_lh.BA.area.txt
aparcstats2table --hemi lh --subjects `cat ../list_fs` --parc BA --meas meancurv --skip -t group_lh.BA.meancurv.txt
aparcstats2table --hemi rh --subjects `cat ../list_fs` --parc BA --meas volume --skip -t group_rh.BA.volume.txt
aparcstats2table --hemi rh --subjects `cat ../list_fs` --parc BA --meas thickness --skip -t group_rh.BA.thickness.txt
aparcstats2table --hemi rh --subjects `cat ../list_fs` --parc BA --meas area --skip -t group_rh.BA.area.txt
aparcstats2table --hemi rh --subjects `cat ../list_fs` --parc BA --meas meancurv --skip -t group_rh.BA.meancurv.txt
