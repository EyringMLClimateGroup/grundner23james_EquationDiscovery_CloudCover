# Data-Driven Equation Discovery of a Cloud Cover Parameterization
A hierarchical modeling framework to discover new machine learning-based equations for cloud cover, including symbolic regression

> Grundner, A., Beucler, T., Gentine, P., & Eyring, V. (2023). Data-Driven Equation Discovery of a Cloud Cover Parameterization.

Author: Arthur Grundner, [arthur.grundner@dlr.de](mailto:arthur.grundner@dlr.de)

------------------------------------------------------------------------

## List of Figures

- [Fig 1] Comparison of the coarse-grained DYAMOND and ERA5 data: sec2_data/analyze_data.ipynb [Fig 1]
- [Fig 3] Predicted cloud cover distributions: sec5_results/sec52_split_by_cloud_regimes/distributions_selected_schemes_pd.pdf 


- sec3_data-driven_modeling/sec33_symbolic_regression_fits/pysr_results/derivative_of_f_wrt_rh.pdf [Fig XX]
- sec3_data-driven_modeling/sec33_symbolic_regression_fits/pysr_results/I1_I2_I3.pdf [Fig XX]


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