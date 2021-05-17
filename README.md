# Growth at Risk
The developer version of the Growth at Risk model used at the IMF -- still beta

The official, IMF-approved version is available in https://github.com/IMFGAR/GaR

This version contains the new functionalities I am developing, without  being reviewed by colleagues. Use them at your own risk !!

Contact: Romain Lafarguette, rlafarguette "at" imf "dot" org

Economist, International Monetary Fund, Monetary and Capital Markets Department (MCM)


- step 001: Group variables into partitions, to reduce parametric noise and
  provides more degrees of freedoms. The partitions are estimated using either
  Principal Component Analysis (PCA) or Partial Least Squares (PLS - also
  called projections on latent structures)
  
- step 002: estimate the quantile regressions, project GDP growth at different
  horizons and generate term structure and fan chart plots. Note that the fan
  charts rely on quantile rearrangement
