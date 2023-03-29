Steps to coarse-grain the data:
1. Run horizontal_coarse-graining/convert_qi_and_qc_to_nc.nc and horizontal_coarse-graining/horizontal_coarse_graining.sh concurrently until all files are processed.
2. Run horizontal_coarse-graining/check_completeness.ipynb and potentially Step 1 again, if some files were deleted.
3. Run horizontal_coarse-graining/diagnose_clouds_from_qi_qc.ipynb, when horizontal_coarse-graining/convert_qi_and_qc_to_nc.nc has finished.
4. Rename variables if necessary or in the filenames (with horizontal_coarse-graining/rename_files.ipynb)
5. Coarse-grain clc data to cloud area fraction [../vertical_coarse-graining/vertical_coarse-graining_cl_area.py]
-> Run ../vertical_coarse-graining/generate_weights_dyamond.py
-> Run ../vertical_coarse-graining/cloud_area_fraction_dyamond.py 
-> Horizontally coarse-grain cl_area
6. Coarse-grain data vertically [../vertical_coarse-graining/vertical_coarse-graining.py]
-> state_vars
-> cloud_vars

----------------------------------------------------------------------------

Horizontal coarse-graining
==========================
horizontal_coarse_graining.sh:
------------------------------
Coarse-grains the data (specified variables) from $inpath and stores the output in $outpath.

check_completeness.ipynb:
-------------------------
If the file has the wrong number of time steps and has not been modified in the last hour, we directly remove it

Cloud Cover
===========
convert_qi_and_qc_to_nc.nc:
---------------------------
Copies qi and qc files to $outpath. Converts them to nc.

diagnose_clouds_from_qi_qc.ipynb:
---------------------------------
Manually creates clc file from qi and qc files in $outpath (diagnoses it from qi+qc). Removes original qi and qc files from $outpath.
