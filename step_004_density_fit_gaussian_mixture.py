# -*- coding: utf-8 -*-
"""
Fit sample with Gaussian Mixtures for the GaR project
Romain Lafarguette, https://romainlafarguette.github.io/
Time-stamp: "2021-05-18 11:33:41 RLafarguette"
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
from sklearn.mixture import GaussianMixture              # Gaussian mixture
from statsmodels.distributions.empirical_distribution import ECDF
from scipy import stats

# Graphics
import matplotlib
matplotlib.use('TkAgg') # Must be called before importing plt
import matplotlib.pyplot as plt                          # Graphical package
import seaborn as sns                                    # Graphical tools

# Computation time
import time
start_time = time.time()

###############################################################################
#%% Load the conditional samples from the projection
###############################################################################
s_f = os.path.join('data', 'clean', 'step_002_conditional_sample_pls.csv')
ds = pd.read_csv(s_f)

###############################################################################
#%% Parameters
###############################################################################
len_support = 1000 # Support for the density estimation
horizon_l = sorted(set(ds.horizon))

# Keep a constant support over the horizons, for readability
y_sample = ds.loc[ds['horizon']==1, 'gdp_growth'].values

y_range = max(y_sample) - min(y_sample); y_rhalf = 0.5*y_range # Boundaries
y_support = np.linspace(min(y_sample) - y_rhalf, max(y_sample) + y_rhalf,
                        len_support)

###############################################################################
#%% Gaussian Mixtures Model (GMM)
###############################################################################
from sklearn.mixture import GaussianMixture as GM
import scipy.stats as stats

g = GM(n_components=2, covariance_type='full')
g_fit = g.fit(y_sample.reshape(-1, 1))
weight_l = list(g_fit.weights_.flatten())
mean_l = list(g_fit.means_.flatten())
covar_l = list(g_fit.covariances_.flatten()) # Covariances
std_l = [np.sqrt(x) for x in covar_l] # We need the standard dev for the norm

# Individual pdf formula = weight * pdf(mean, standard deviation)
y0_pdf = [weight_l[0]*stats.norm.pdf(y, mean_l[0],std_l[0]) for y in y_support]
y1_pdf = [weight_l[1]*stats.norm.pdf(y, mean_l[1],std_l[1]) for y in y_support]

# Individual cdf formula
y0_cdf = [weight_l[0]*stats.norm.cdf(y, mean_l[0],std_l[0]) for y in y_support]
y1_cdf = [weight_l[1]*stats.norm.cdf(y, mean_l[1],std_l[1]) for y in y_support]

# Mixed density: simply the weighted sum of the two !
ym_pdf = [y0 + y1 for y0, y1 in zip(y0_pdf, y1_pdf)]

# Only for Gaussian: the CDF of the mixture is the mixture of CDF
ym_cdf = [y0 + y1 for y0, y1 in zip(y0_cdf, y1_cdf)]

# Only for benchmarking: Gaussian Kernel
y_kde = scipy.stats.gaussian_kde(y_sample) # This is a function
y_kde_pdf = y_kde(y_support)
y_kde_cdf = [y_kde.integrate_box_1d(-np.inf, y) for y in y_support]

###############################################################################
#%% Plots
###############################################################################
# Set the style 
sns.set(style='white', font_scale=1.5, palette='deep', font='Arial')
fig, axs = plt.subplots(3, 1, sharex=True)
ax1, ax2, ax3 = axs.ravel()

# Histogram with KDE
ax1.hist(y_sample, bins=20, density=True, label='Histogram', alpha=0.8)
ax1.plot(y_support, y_kde_pdf, label='KDE', lw=2, ls='--', color='blue') 
ax1.legend(frameon=False)
ax1.set_title('Histogram and KDE non-parametric fit', y=1.02)

# Gaussian with KDE
ax2.plot(y_support, ym_pdf, label='Gaussian Dual Mixture', lw=2, color='red')
ax2.plot(y_support, y_kde_pdf, label='KDE', lw=2, ls='--', color='blue') 
ax2.legend(frameon=False)
ax2.set_title('Gaussian Mixture and KDE', y=1.02)

# Decomposition of the Gaussian
ax3.plot(y_support, y0_pdf, label='First Gaussian', ls='--', color='green')
ax3.plot(y_support, y1_pdf, label='Second Gaussian', ls='--', color='blue')
ax3.plot(y_support, ym_pdf, label='Mixture', ls='-', color='red')
ax3.legend(frameon=False)
ax3.set_xlabel('GDP percentage points', labelpad=20, fontsize=20)
ax3.set_title('Gaussian Mixture and Individual Densities', y=1.02)

# Layout
fig.set_size_inches(25, 16)
fig.tight_layout()

# Save the figure
fig.savefig(os.path.join('output', 'step_004_gaussian_mixture.pdf'))
plt.show()

###############################################################################
#%% End 
###############################################################################
