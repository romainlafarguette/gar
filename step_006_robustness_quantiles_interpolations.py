# -*- coding: utf-8 -*-
"""
Try different quantile interpolations methods
Romain Lafarguette, https://romainlafarguette.github.io/
Time-stamp: "2021-05-17 19:57:31 RLafarguette"
"""

###############################################################################
#%% Preamble
###############################################################################
# Main modules
import os, sys, importlib                                # System packages
import pandas as pd                                      # Dataframes 
import numpy as np                                       # Numerical tools
import scipy

# Functional imports
from datetime import datetime as date                    # Dates
from dateutil.relativedelta import relativedelta         # Add time

# Statistics
import statsmodels as sm                                # Statistical models
import statsmodels.formula.api as smf                   # Formulas
from scipy import interpolate                           # Linear interpolation
from scipy.stats import norm, t                         # Gaussian and t

# Local imports
sys.path.append('modules')
import quantiles_spacing; importlib.reload(quantiles_spacing)
from quantiles_spacing import qs_ppf

# Graphics
import matplotlib
matplotlib.use('TkAgg') # Must be called before importing plt
import matplotlib.pyplot as plt                          # Graphical package
import seaborn as sns                                    # Graphical tools

# Computation time
import time
start_time = time.time()

###############################################################################
#%% Data loading
###############################################################################
# Original frame
data_path = os.path.join('data', 'raw', 'example_data.csv')
df = pd.read_csv(data_path, parse_dates=['date'], index_col='date')

# PLS frames
pls_transf_p = os.path.join('data', 'clean', 'step_001_pls_transformed.csv')
dpls = pd.read_csv(pls_transf_p, parse_dates=['date'], index_col=['date'])

###############################################################################
#%% Parameters
###############################################################################
groups_l = [x for x in list(dpls.columns) if x not in ['gdp_growth']]
horizon_l = [1, 4, 8] # Horizons, in quarter
quantile_l = [0.1, 0.25, 0.5, 0.75, 0.9]
alpha_ci = 0.05 # Confidence interval threshold
stats_names = ['coeff', 'tval', 'pval', 'lower_ci', 'upper_ci'] # Qreg output
len_bs = 1000 # Number of points to sample from 

###############################################################################
#%% Estimate a set of quantile regressions => get the Beta^{quantile}
###############################################################################
# Generate the formula for the quantile regression
regressors_l = groups_l[0]
for v in groups_l[1:]: regressors_l += f' + {v}'
quantile_formula = f'gdp_growth ~ {regressors_l}'
                      
# Estimate the quantile regressions for all quantiles and store the results 
dres_l = list() # Container to store the results
for tau in quantile_l:
    qfit = smf.quantreg(formula=quantile_formula, data=dpls).fit(q=tau)

    # Retrieve all the statistics of interest in a frame
    stats = [qfit.params, qfit.tvalues, qfit.pvalues, qfit.conf_int(alpha_ci)]
    dres = pd.concat(stats, axis=1); dres.columns = stats_names

    # Add specific information
    dres['pseudo_r2'] = qfit.prsquared
    dres.insert(0, 'tau', tau)

    # Store it
    dres_l.append(dres)

# Concatenate the coefficients and main statistics in a single frame
dresults = pd.concat(dres_l)

###############################################################################
#%% Select a conditioning vector X (historical or simulated) with intercept !
###############################################################################
dpls['Intercept'] = 1
X0 = dpls.loc['2018-06-30', ['Intercept'] + groups_l] # Order is important

###############################################################################
#%% Compute the conditional quantiles = Beta^{q} * X
###############################################################################
# Need to rearrange the matrices so that they conform
# Beta should be (num quantiles x num of regressors)
# X should be (num of regressors x 1)
X = X0.values.reshape(-1, 1) # -1 in numpy is a "joker": let numpy find a way


# Reshape the coefficients using an index approach: reduce mistakes risks
beta = pd.DataFrame(index=quantile_l, columns=['Intercept'] + groups_l)
for quantile in quantile_l:    
    # Isolate the results associated with a quantile
    drq = dresults.loc[dresults['tau']==quantile, :].copy()

    # Find the right coefficient for each regressor
    for regressor in ['Intercept'] + groups_l:
        beta.loc[quantile, regressor] = drq.loc[regressor, 'coeff']

# Compute the conditional quantiles: simply a matrix multiplication (dot())
condquant_l = beta.dot(X).values.flatten()
print(condquant_l)
    
###############################################################################
#%% Inverse Transform Sampling (linear interpolation of conditional quantiles)
###############################################################################
# Important to extrapolate to get the points at the extreme
lin_interp = interpolate.interp1d(quantile_l, condquant_l,
                                  fill_value='extrapolate')

# Generate a uniform random sample, for the inverse transform sampling
U = np.random.uniform(0, 1, 1000)

# Now, inverse transform the uniform sample using the linear interpolation
y_its_sample = lin_interp(U) # Sample of y from the conditional quantiles

###############################################################################
#%% Quantile spacing
###############################################################################
# Use a Gaussian distribution as base function
y_qsp_sample_norm = [qs_ppf(u, quantile_l, condquant_l, base=norm.ppf)
                     for u in U]

# Use a t distribution 2 degrees of freedom as base function
y_qsp_sample_t = [qs_ppf(u, quantile_l, condquant_l, base=t(2).ppf) for u in U]

###############################################################################
#%% Compare the samples obtained from these approaches
###############################################################################
# Compute KDE just for the plots
y_val = dpls['gdp_growth'].values
y_range = max(y_val) - min(y_val); y_rhalf = 0.01*y_range # Boundaries
y_support = np.linspace(min(y_val) - y_rhalf, max(y_val) + y_rhalf, 1000)

y_kde_its = scipy.stats.gaussian_kde(y_its_sample)(y_support)
y_kde_qsp_norm = scipy.stats.gaussian_kde(y_qsp_sample_norm)(y_support)
y_kde_qsp_t = scipy.stats.gaussian_kde(y_qsp_sample_t)(y_support)

# Plot
sns.set(style='white', font_scale=2, palette='deep', font='Arial')
fig, axs = plt.subplots(3, 1, sharex=True)
ax1, ax2, ax3 = axs.ravel()

ax1.hist(y_its_sample, bins=20, density=True, label='Histogram', alpha=0.8)
ax1.plot(y_support, y_kde_its, label='KDE', lw=2, ls='--', color='blue')
ax1.legend(frameon=False)
ax1.set_title('Sample from Linear Interpolation', y=1.02)

ax2.hist(y_qsp_sample_norm, bins=20, density=True, label='Histogram',
         alpha=0.8)
ax2.plot(y_support, y_kde_qsp_norm, label='KDE',
        lw=2, ls='--', color='blue')
ax2.legend(frameon=False)
ax2.set_title('Sample from Quantile Spacing with Gaussian Base', y=1.02)

ax3.hist(y_qsp_sample_t, bins=120, density=True, label='Histogram', alpha=0.8)
ax3.plot(y_support, y_kde_qsp_t,
        label='KDE', lw=2, ls='--', color='blue')
ax3.legend(frameon=False)
ax3.set_title('Sample from Quantile Spacing with t(2) base function', y=1.02)
ax3.set_xlim((-0.5, 1))

# Layout
fig.set_size_inches(25, 16)
fig.tight_layout()

plt.show()

###############################################################################
#%% Plot the interpolation on a fix support to illustrate the idea
###############################################################################
# Just for the plot: this is not a random sample, just to show the linearity
q_support = np.linspace(0, 1, 1000)
cq_support_lin = lin_interp(q_support)
cq_support_qsp_norm = [qs_ppf(x, quantile_l, condquant_l, base=norm.ppf)
                       for x in q_support]
cq_support_qsp_t = [qs_ppf(x, quantile_l, condquant_l, base=t(2).ppf)
                    for x in q_support]

# Plot
fig, axs = plt.subplots(3, 1, sharex=True)
ax1, ax2, ax3 = axs.ravel()

# Linear
ax1.plot(q_support, cq_support_lin, lw=3, 
         label='Linear quantile interpolation')
ax1.scatter(quantile_l, condquant_l, marker='D', color='red', s=100,
            label='Estimated conditional quantiles')
ax1.legend(frameon=False)

# Qspacing with Gaussian
ax2.plot(q_support, cq_support_qsp_norm, lw=3, 
         label='Quantile Spacing with Gaussian base function')
ax2.scatter(quantile_l, condquant_l, marker='D', color='red', s=100,
            label='Estimated conditional quantiles')
ax2.legend(frameon=False)

# Qspacing with T distribution
ax3.plot(q_support, cq_support_qsp_t, lw=3, 
         label='Quantile Spacing with t(2) base function')
ax3.scatter(quantile_l, condquant_l, marker='D', color='red', s=100,
            label='Estimated conditional quantiles')
ax3.legend(frameon=False)
ax3.set_xlabel('Quantiles')

fig.suptitle('Quantiles Interpolation Methods')

# Layout
fig.set_size_inches(25, 16)
fig.subplots_adjust(top=0.90, left=0.05, right=0.95, bottom=0.1)

plt.show()

###############################################################################
#%% End
###############################################################################
