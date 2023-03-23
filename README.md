# Data-Driven Equation Discovery of a Cloud Cover Parameterization
A hierarchical modeling framework to discover new machine learning-based equations for cloud cover, including symbolic regression

> Grundner, A., Beucler, T., Gentine, P., & Eyring, V. (2023). Data-Driven Equation Discovery of a Cloud Cover Parameterization.

Author: Arthur Grundner, [arthur.grundner@dlr.de](mailto:arthur.grundner@dlr.de)

------------------------------------------------------------------------

## List of Figures

- [Fig 1] Comparison of the coarse-grained DYAMOND and ERA5 data: sec2_data/analyze_data.ipynb
- [Fig 2] All cloud cover schemes in a performance x complexity plot: sec512_balancing_performance_and_complexity/performance_vs_complexity_logscale_pysr_fixed.pdf
- [Fig 3] Predicted cloud cover distributions: sec5_results/sec52_split_by_cloud_regimes/distributions_selected_schemes_pd.pdf 
- [Fig 4.1] Transfer learning to ERA5 data (selected schemes): sec5_results/sec53_transferability_to_era5/era5_1979-2021/tf_main_scatter.pdf
- [Fig 4.2] Transfer learning to ERA5 data (polynomials & NNs): sec5_results/sec53_transferability_to_era5/era5_1979-2021/tf_add_scatter.pdf
- [Fig 5.1]: Plots of the terms I_1, I_2, I_3: sec3_data-driven_modeling/sec33_symbolic_regression_fits/pysr_results/I1_I2_I3.pdf 

- [Fig 6.1]: Contour plot of dzRH: sec3_data-driven_modeling/sec33_symbolic_regression_fits/pysr_results/derivative_of_f_wrt_rh.pdf

------------------------------------------------------------------------

## Data

To reproduce the results it is first necessary to coarse-grain and preprocess the DYAMOND and ERA5/ERA5.1 data sets:
- Guide for how to coarse-grain the DYAMOND data: sec2_data/sec21_DYAMOND/strategy.txt
- To then pre-process the DYAMOND data: sec2_data/sec21_DYAMOND/preprocessing.ipynb
- Scripts to coarse-grain ERA5 data (1979-2021, first day of every quarter): sec22_ERA5/horizontal_coarse-graining

It suffices to coarse-grain the variables: clc/cc, cli/ciwc, clw/clwc, hus/q, pa, ta/t, ua/u, va/v, zg/z

------------------------------------------------------------------------

## Some Key Python Packages

- PySR 0.10.1 [https://github.com/MilesCranmer/PySR]
- GP-GOMEA [https://github.com/marcovirgolin/GP-GOMEA]
- mlxtend 0.20.0 [https://github.com/rasbt/mlxtend]
- scikit-learn 1.0.2 [https://scikit-learn.org/]
- TensorFlow 2.7.0 [https://tensorflow.org/]

------------------------------------------------------------------------

## License
This code is released under Apache 2.0. See [LICENSE](LICENSE) for more information.