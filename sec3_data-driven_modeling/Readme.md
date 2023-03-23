The **sec31_existing_schemes** folder contains the scripts for all pre-existing schemes, and also the NNs themselves, now fit to the DYAMOND data.
First of all, there are three different types of neural networks from Grundner et al., 2021. We can re-train them by pre-processing the data (preprocessing.ipynb), training the NNs (cross_validation.ipynb) and evaluating them (evaluate_nns.ipynb). Second, it contains three different diagnostic cloud cover schemes from the literature. The Sundqvist scheme is currently used in the ICON model.

The **sec321_linear_models_and_polynomials** folder contains scripts and linear models and polynomials fit to the DYAMOND data with a sequential feature selection approach. 

The **sec322_neural_networks** folder contains scripts and neural networks fit to the DYAMOND data with a sequential feature selection approach.

The **sec33_symbolic_regression_fits** folder contains scripts and equations found using symbolic regression (PySR and GP-GOMEA). The folder also contains the analysis of whether the equations satisfy physical constraints, and a futher optimization of their parameters.