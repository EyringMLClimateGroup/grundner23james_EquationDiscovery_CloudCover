This folder contains the scripts to analyze the analytic scheme and to run ablation tests.

In **optimize_coefs_EQ4_check_physical_eqns.ipynb** we plot the behavior of the analytic scheme, split into I1, I2, I3.

In **analyzing_eqns.ipynb** we create conditional average plots for different cloud cover schemes. We also analyze whether the continuity constraint is satisfied by our analytic scheme and the cases in which the removal of the I2-term induces a large error.

The other scripts are for the ablation study, analyzing the change of error if we remove terms on the ERA5 and the DYAMOND data, in the equation phrased in terms of physical variables and in terms of normalized variables. The ablation study includes a re-tuning of the remaining coefficients. In **optimize_coefs_EQ4_mod_naive_ablation_study_phys.ipynb** we found that the re-tuning is indeed necessary. 