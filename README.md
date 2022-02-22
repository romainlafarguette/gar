# Conditional Density Projection via Quantile Regressions, Resampling and Multifit Models

The developer version of the Growth at Risk model used at the IMF -- still beta

The official, IMF-approved version is available in https://github.com/IMFGAR/GaR

This version contains the new functionalities I am developing, without  being reviewed by colleagues. Use them at your own risk !!

https://romainlafarguette.github.io/software/

The project is split along different steps, that have to be ran sequentially:

- step 001: Group variables into partitions, to reduce parametric noise and
  provides more degrees of freedoms. The partitions are estimated using either
  Principal Component Analysis (PCA) or Partial Least Squares (PLS - also
  called projections on latent structures)
  
- step 002: estimate the quantile regressions, project GDP growth at different
  horizons and generate term structure and fan chart plots. Note that the fan
  charts rely on quantile rearrangement

- step 003: fit the sampled density using kernel and parametric
  densities. Best parametric family is assessed using AIC, BIC or RSS
  criteria. 

- step 004: fit the density using Gaussian mixtures

- step 005: measure the performance of the density forecasts using PIT,
  logscores and entropy tests

- step 006: try different quantiles interpolation methods. This script is not
  very important for a "standard" GaR use, it was rather to test the
  robustness of the rearrangement approach in quantiles uncrossing and
  sampling. 
