# Data-Driven Equation Discovery of a Cloud Cover Parameterization
A hierarchical modeling framework to discover new machine learning-based equations for cloud cover, including symbolic regression

> Grundner, A., Beucler, T., Gentine, P., & Eyring, V. (2023). Data-Driven Equation Discovery of a Cloud Cover Parameterization.

Author: Arthur Grundner, [arthur.grundner@dlr.de](mailto:arthur.grundner@dlr.de)

## List of Figures

- [Fig 1](sec2_data/era5_dyamond_comp_4_vars.pdf), [Code](sec2_data/analyze_data.ipynb): Comparison of the coarse-grained DYAMOND and ERA5 data
- [Fig 2](sec5_results/sec512_balancing_performance_and_complexity/performance_vs_complexity_logscale_pysr_fixed.pdf), [Code](sec5_results/sec512_balancing_performance_and_complexity/combine_results_dyamond.ipynb): All cloud cover schemes in a performance x complexity plot
- [Fig 3](sec5_results/sec52_split_by_cloud_regimes/distributions_selected_schemes_pd.pdf), [Code](sec5_results/sec52_split_by_cloud_regimes/combining_selected_distributions.ipynb): Predicted cloud cover distributions
- [Fig 4.1](sec5_results/sec53_transferability_to_era5/era5_1979-2021/tf_main_scatter.pdf), [Code](sec5_results/sec53_transferability_to_era5/combine_results.ipynb): Transfer learning to ERA5 data (selected schemes)
- [Fig 4.2](sec5_results/sec53_transferability_to_era5/era5_1979-2021/tf_add_scatter.pdf), [Code](sec5_results/sec53_transferability_to_era5/combine_results.ipynb): Transfer learning to ERA5 data (polynomials & NNs)
- [Fig 5.1](sec6_physical_interpretation/I1_I2_I3.pdf), [Code](sec6_physical_interpretation/optimize_coefs_EQ4_check_physical_eqns.ipynb): Plots of the terms I_1, I_2, I_3
- [Fig 5.2](sec6_physical_interpretation/rh_and_T_vs_cl_area.pdf), [Code](sec6_physical_interpretation/analyzing_eqns.ipynb): Conditional average w.r.t. RH and T
- [Fig 5.3](sec6_physical_interpretation/rh_z_vs_cl_area_new.pdf), [Code](sec6_physical_interpretation/analyzing_eqns.ipynb): Conditional average w.r.t. dzRH
- [Fig 6.1](sec6_physical_interpretation/derivative_of_f_wrt_rh.pdf), [Code](sec6_physical_interpretation/dzRH_contour_plot.ipynb): Contour plot of dzRH
- [Fig 6.2](sec6_physical_interpretation/RH_vs_cl_area_mod.pdf), [Code](sec6_physical_interpretation/pdp_plot_rh.ipynb): Cloud cover w.r.t. RH with and without modification to satisfy the RH-physical constraint
- [Fig 7](sec6_physical_interpretation/ablation_study_dyamond/dyamond_era5_ablation_study_results.pdf), [Code](sec6_physical_interpretation/ablation_study_dyamond/plot_results.ipynb): Ablation study of our analytic scheme on DYAMOND and ERA5 data
- [Fig A1.1](appendix/I1_lv_41_20160811-0820_timmean.pdf), [Fig A1.2](appendix/I2_lv_41_20160811-0820_timmean.pdf), [Fig A1.3](appendix/I3_lv_41_20160811-0820_timmean.pdf), [Code](appendix/plotting_I1_I2_I3_geographical.ipynb): Maps of I1, I2, I3 on a specific vertical layer on ~1490m averaged over 10 days of DYAMOND data

## Data

To reproduce the results it is first necessary to coarse-grain and preprocess the DYAMOND and ERA5/ERA5.1 data sets:
- Guide for how to coarse-grain the DYAMOND data: [strategy.txt](sec2_data/sec21_DYAMOND/strategy.txt)
- To then pre-process the DYAMOND data: [preprocessing.ipynb](sec2_data/sec21_DYAMOND/preprocessing.ipynb) 
- Scripts to coarse-grain ERA5 data (1979-2021, first day of every quarter): [horizontal](sec22_ERA5/horizontal_coarse-graining), [vertical](sec2_data/vertical_coarse-graining)

It suffices to coarse-grain the variables: clc/cc, cli/ciwc, clw/clwc, hus/q, pa, ta/t, ua/u, va/v, zg/z

## Some Key Python Packages

- PySR 0.10.1 [https://github.com/MilesCranmer/PySR]
- GP-GOMEA [https://github.com/marcovirgolin/GP-GOMEA]
- mlxtend 0.20.0 [https://github.com/rasbt/mlxtend]
- scikit-learn 1.0.2 [https://scikit-learn.org/]
- SymPy 1.10.1 [https://github.com/sympy]
- SciPy 1.8.1 [https://github.com/scipy/]
- TensorFlow 2.7.0 [https://tensorflow.org/]

## License
This code is released under Apache 2.0. See [LICENSE](LICENSE) for more information.
