# Data-Driven Equation Discovery of a Cloud Cover Parameterization
A hierarchical modeling framework to discover new machine learning-based equations for cloud cover, including symbolic regression

> Grundner, A., Beucler, T., Gentine, P., & Eyring, V. (2023). Data-Driven Equation Discovery of a Cloud Cover Parameterization.

Author: Arthur Grundner, [arthur.grundner@dlr.de](mailto:arthur.grundner@dlr.de)

## List of Figures

- [Fig 1] Comparison of the coarse-grained DYAMOND and ERA5 data: [sec2_data/analyze_data.ipynb](sec2_data/analyze_data.ipynb)
- All cloud cover schemes in a performance x complexity plot: [Fig 2](sec5_results/sec512_balancing_performance_and_complexity/performance_vs_complexity_logscale_pysr_fixed.pdf)
- [Fig 3] Predicted cloud cover distributions: sec5_results/sec52_split_by_cloud_regimes/distributions_selected_schemes_pd.pdf 
- [Fig 4.1] Transfer learning to ERA5 data (selected schemes): sec5_results/sec53_transferability_to_era5/era5_1979-2021/tf_main_scatter.pdf
- [Fig 4.2] Transfer learning to ERA5 data (polynomials & NNs): sec5_results/sec53_transferability_to_era5/era5_1979-2021/tf_add_scatter.pdf
- [Fig 5.1]: Plots of the terms I_1, I_2, I_3: sec6_physical_interpretation/I1_I2_I3.pdf
- [Fig 5.2]: Conditional average w.r.t. RH and T: sec6_physical_interpretation/rh_and_T_vs_cl_area.pdf
- [Fig 5.3]: Conditional average w.r.t. dzRH: sec6_physical_interpretation/rh_z_vs_cl_area_new.pdf
- [Fig 6.1]: Contour plot of dzRH: sec6_physical_interpretation/derivative_of_f_wrt_rh.pdf
- [Fig 6.2]: Cloud cover w.r.t. RH with and without modification to satisfy the RH-physical constraint: sec6_physical_interpretation/RH_vs_cl_area_mod.pdf
- [Fig 7]: Ablation study of our analytic scheme on DYAMOND and ERA5 data: sec6_physical_interpretation/ablation_study_dyamond/dyamond_era5_ablation_study_results.pdf
- [Fig A1]: Maps of I1, I2, I3 on a specific vertical layer on ~1490m averaged over 10 days of DYAMOND data: appendix/I\[1,2,3\]_lv_41_20160811-0820_timmean.pdf

The scripts to generate these figures can be found in the same folders.

## Data

To reproduce the results it is first necessary to coarse-grain and preprocess the DYAMOND and ERA5/ERA5.1 data sets:
- Guide for how to coarse-grain the DYAMOND data: sec2_data/sec21_DYAMOND/strategy.txt
- To then pre-process the DYAMOND data: sec2_data/sec21_DYAMOND/preprocessing.ipynb
- Scripts to coarse-grain ERA5 data (1979-2021, first day of every quarter): sec22_ERA5/horizontal_coarse-graining

It suffices to coarse-grain the variables: clc/cc, cli/ciwc, clw/clwc, hus/q, pa, ta/t, ua/u, va/v, zg/z

## Some Key Python Packages

- PySR 0.10.1 [https://github.com/MilesCranmer/PySR]
- GP-GOMEA [https://github.com/marcovirgolin/GP-GOMEA]
- mlxtend 0.20.0 [https://github.com/rasbt/mlxtend]
- scikit-learn 1.0.2 [https://scikit-learn.org/]
- TensorFlow 2.7.0 [https://tensorflow.org/]

## License
This code is released under Apache 2.0. See [LICENSE](LICENSE) for more information.
