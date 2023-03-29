Steps to coarse-grain the data:
1. Run [convert_qi_and_qc_to_nc.nc](horizontal_coarse-graining/convert_qi_and_qc_to_nc.sh) and [horizontal_coarse_graining.sh](horizontal_coarse-graining/horizontal_coarse_graining.sh) concurrently until all files are processed.
2. Run [check_completeness.ipynb](horizontal_coarse-graining/check_completeness.ipynb) and potentially Step 1 again, if some files were deleted.
3. Run [diagnose_clouds_from_qi_qc.ipynb](horizontal_coarse-graining/diagnose_clouds_from_qi_qc.ipynb), when [convert_qi_and_qc_to_nc.nc](horizontal_coarse-graining/convert_qi_and_qc_to_nc.nc) has finished.
4. Rename variables if necessary or in the filenames (with [rename_files.ipynb](horizontal_coarse-graining/rename_files.ipynb))
5. Coarse-grain clc data to cloud area fraction [vertical_coarse-graining_cl_area.py](../vertical_coarse-graining/vertical_coarse-graining_cl_area.py)
-> Run [generate_weights_dyamond.py](../vertical_coarse-graining/generate_weights_dyamond.py)
-> Run [cloud_area_fraction_dyamond.py](../vertical_coarse-graining/cloud_area_fraction_dyamond.py)
-> Horizontally coarse-grain cl_area
6. Coarse-grain data vertically [vertical_coarse-graining.py](../vertical_coarse-graining/vertical_coarse-graining.py)
-> state_vars
-> cloud_vars
