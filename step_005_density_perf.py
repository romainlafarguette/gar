# -*- coding: utf-8 -*-
"""
Estimation of the density performance for the GaR project
Romain Lafarguette, https://romainlafarguette.github.io/
Time-stamp: "2021-05-17 19:39:09 RLafarguette"
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
from statsmodels.distributions.empirical_distribution import ECDF
from scipy import stats

# Graphics
import matplotlib
matplotlib.use('TkAgg') # Must be called before importing plt
import matplotlib.pyplot as plt                          # Graphical package
import seaborn as sns                                    # Graphical tools

# Local imports
sys.path.append('modules')
import quantileproj; importlib.reload(quantileproj)
from quantileproj import QuantileProj

# Computation time
import time
start_time = time.time()

###############################################################################
#%% Graphical parameters
###############################################################################
plt.rcParams["figure.figsize"] = 25, 16 

# Graphical parameters defined here for consistency
sns.set(style='white', font_scale=1.5, palette='dark', font='Arial')

# I change the order of the official palette to have first blue, red, green
official_l = list(sns.color_palette('dark'))
color_l = [official_l[i] for i in [0, 3, 2, 1, 4]]  +  official_l[5:]

# I repeat the linestyle to make sure that we cover at least 10 options
linestyle_l = ['-', '--', '-.', ':']*3

# Hatches style
hatch_l = ['X', '', '\\', '//', '+']*2

# Legend parameters
legend_d = {'frameon': False, 'handlelength': 1}

###############################################################################
#%% Data loading
###############################################################################
# PCA frame
pca_transf_p = os.path.join('data', 'clean', 'step_001_pls_transformed.csv')
df = pd.read_csv(pca_transf_p, parse_dates=['date'], index_col=['date'])

# Parameters
df['autoreg'] = df['gdp_growth'].copy()
groups_l = [x for x in list(df.columns) if x not in ['gdp_growth']]
horizon_l = [1, 4, 10, 16, 28, 40] # Horizons, in month
quantile_l = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95] # List of quantiles to fit
alpha_ci = 0.05 # Confidence interval threshold

###############################################################################
#%% Parameters 
###############################################################################
# Constant support (easier for the charts, but can be changed)
y_true = df['gdp_growth'].dropna().values
y_range = max(y_true) - min(y_true); y_rhalf = 0.15*y_range # Boundaries
y_support = np.linspace(min(y_true) - y_rhalf, max(y_true) + y_rhalf, 1000)

np.random.seed(18041202) # Fix the random generating seed, L'Empereur !!

###############################################################################
#%% Time series of quantile projections using the PLS specification at 1-y h
###############################################################################
# Fit the coefficients on all the historical data
horizon = 1
qp_pls = QuantileProj('gdp_growth', groups_l, df, [horizon]) # Init
qp_pls_fit = qp_pls.fit(quantile_l=quantile_l, alpha=alpha_ci) # Fit

# Project at each periods
dates_l = df.dropna().index
sample_l = list() # Container to store the projected samples at each date

for cdate in dates_l: # For current date        
    # Estimation
    cond_frame = df.loc[[cdate], groups_l].copy() # X at current date
    projection = qp_pls_fit.proj(cond_frame)
    sample = projection.sample(len_sample=1000)['gdp_growth'].values #Sampling
    
    # Store the sample in a dataframe
    ndate = cdate + relativedelta(months=3*horizon) # Date at projection
    ds = pd.DataFrame(sample, columns=['forecasted_val'])
    ds.insert(0, 'forecasted_date', ndate) # Keep track of the date
    sample_l.append(ds) # Store in the list

# Also fit against the unconditional distribution
unc_frame = df.loc[:, groups_l].mean().to_frame().T # Historical average
unc_projection = qp_pls_fit.proj(unc_frame)
unc_sample = unc_projection.sample(len_sample=1000)['gdp_growth'].values

# Concatenate all the samples into one final frame
dsample = pd.concat(sample_l, axis='index')    
dsample = dsample.set_index('forecasted_date')

# Add the historical true value, based on the index of df and dsample
joint_idx = [x for x in df.index if x in dsample.index]
dsample.loc[joint_idx, 'true_val'] = df['gdp_growth'].copy()

###############################################################################
#%% Recursive Gaussian KDE & parametric fits at each period
###############################################################################
fdates_l = sorted(set(dsample.index)) # List of forecasted dates

# Use the KDE on the unconditional distribution as benchmark
unc_kde = scipy.stats.gaussian_kde(unc_sample) # This is a function

# Store the performance results in a new dataframe
dperf = dsample[['true_val']].dropna().drop_duplicates().copy()
dperf['kde_pit'] = np.nan
dperf['kde_score'] = np.nan

dperf['parametric_pit'] = np.nan
dperf['parametric_score'] = np.nan

dperf['unconditional_kde_pit'] = np.nan
dperf['unconditional_kde_score'] = np.nan

# Create a new variable to store the pdf over the support
dsample['kde_pdf'] = np.nan
dsample['parametric_pdf'] = np.nan

for fdate in fdates_l:
    # Quantities of interest
    y_true = list(set(dsample.loc[fdate, 'true_val']))[0] # One value
    forecasted_sample = dsample.loc[fdate, 'forecasted_val']
    
    #### Fit the Gaussian kernel
    y_kde = scipy.stats.gaussian_kde(forecasted_sample) # This is a function

    # Estimate the pit: the cdf of the true value
    dperf.loc[fdate, 'kde_pit'] = y_kde.integrate_box_1d(-np.inf, y_true)

    # Estimate the kde pdf over the support
    kde_pdf = y_kde.pdf(y_support)
    kde_total_mass = np.sum(kde_pdf) # To have a pdf summing to 1
    dsample.loc[fdate, 'kde_pdf'] = kde_pdf/kde_total_mass

    # Estimate the forecasted PDF of the true value (the score) 
    dperf.loc[fdate, 'kde_score'] = float(y_kde.pdf([y_true]))/kde_total_mass

    #### Parametric fit
    params = scipy.stats.lomax.fit(forecasted_sample) 
    pfit_rv = scipy.stats.lomax(*params) # Create the random variable
    pfit_pdf = pfit_rv.pdf(y_support)
    pfit_total_mass = np.sum(pfit_pdf) # To have a pdf summing to 1
    dsample.loc[fdate, 'parametric_pdf'] = pfit_pdf/pfit_total_mass

    dperf.loc[fdate, 'parametric_pit'] = pfit_rv.cdf(y_true)
    dperf.loc[fdate, 'parametric_score'] = pfit_rv.pdf(y_true)/pfit_total_mass

    #### Unconditional 
    unc_pdf = unc_kde.pdf(y_support)
    unc_total_mass = np.sum(unc_pdf) # To have a pdf summing to 1
    dsample.loc[fdate, 'unc_pdf'] = unc_pdf/unc_total_mass
    dperf.loc[fdate, 'unc_score'] = float(unc_kde.pdf([y_true]))/unc_total_mass
    dperf.loc[fdate, 'unc_pit'] = unc_kde.integrate_box_1d(-np.inf, y_true)
    
    print(f'Estimation done for {fdate: %b %Y}')
    
###############################################################################
#%% Quick pdf chart to check the estimation process
###############################################################################
# Pick randomly a forecasted date
fdate = date(2016, 3, 31)

#### PDF plot
fig, axs = plt.subplots(3, 1, sharey=True) # 3 rows 
y_true = float(dperf.loc[fdate, 'true_val'])

# Unconditional distribution distribution
unc_pdf = dsample.loc[fdate, 'unc_pdf'].values
axs[0].plot(y_support, unc_pdf, lw=3, color='navy',
                 label='Forecasted pdf')
axs[0].axvline(x=y_true, lw=3, color='firebrick',
                    label='Realized value')
axs[0].set_title(f'Unconditional', y=1.02)
axs[0].legend(fontsize=20, loc='upper left', frameon=False)

    
# KDE distribution
kde_pdf = dsample.loc[fdate, 'kde_pdf'].values
axs[1].plot(y_support, kde_pdf, lw=3, color='navy',
                 label='Forecasted pdf')
axs[1].axvline(x=y_true, lw=3, color='firebrick',
                    label='Realized value')
axs[1].set_title(f'Kernel', y=1.02)
axs[1].legend(fontsize=20, loc='upper left', frameon=False)

# Parametric distribution
param_pdf = dsample.loc[fdate, 'parametric_pdf'].values
axs[2].plot(y_support, param_pdf, lw=3, color='navy',
                 label='Forecasted pdf')
axs[2].axvline(x=y_true, lw=3, color='firebrick',
                    label='Realized value')
axs[2].set_title(f'Parametric', y=1.02)
axs[2].legend(fontsize=20, loc='upper left', frameon=False)

    
plt.subplots_adjust(hspace=0.5)
plt.suptitle('Forecasted density fit and realized value under different models')

# Save the chart
kde_density_plots = os.path.join('output', 'step_005_pdf_fit.pdf')
plt.savefig(kde_density_plots)

plt.show()

plt.close('all')

###############################################################################
#%% Plot the PIT
###############################################################################
# Data work (note that the pit are computed by default)
pit_support = np.arange(0,1, 0.01)

### For KDE
kde_pits = dperf['kde_pit'].dropna().copy()

# Compute the empirical CDF on the pits
kde_ecdf = ECDF(kde_pits)

# Fit it on the support
kde_pit_line = kde_ecdf(pit_support)

# Compute the KS statistics (in case of need)
kde_ks_stats = scipy.stats.kstest(kde_pits, 'uniform')
kde_ks_pval = round(100*kde_ks_stats[1], 1)

# Confidence intervals based on Rossi and Shekopysan JoE 2019
kde_ci_u = [x+1.34*len(kde_pits)**(-0.5) for x in pit_support]
kde_ci_l = [x-1.34*len(kde_pits)**(-0.5) for x in pit_support]

### For parametric fit
param_pits = dperf['parametric_pit'].dropna().copy()

# Compute the empirical CDF on the pits
param_ecdf = ECDF(param_pits)

# Fit it on the support
param_pit_line = param_ecdf(pit_support)

# Compute the KS statistics (in case of need)
param_ks_stats = scipy.stats.kstest(param_pits, 'uniform')
param_ks_pval = round(100*param_ks_stats[1], 1)

# Confidence intervals based on Rossi and Shekopysan JoE 2019
param_ci_u = [x+1.34*len(param_pits)**(-0.5) for x in pit_support]
param_ci_l = [x-1.34*len(param_pits)**(-0.5) for x in pit_support]

### For unconditional distribution
unc_pits = dperf['unc_pit'].dropna().copy()

# Compute the empirical CDF on the pits
unc_ecdf = ECDF(unc_pits)

# Fit it on the support
unc_pit_line = unc_ecdf(pit_support)

# Compute the KS statistics (in case of need)
unc_ks_stats = scipy.stats.kstest(unc_pits, 'uniform')
unc_ks_pval = round(100*unc_ks_stats[1], 1)

# Confidence intervals based on Rossi and Shekopysan JoE 2019
unc_ci_u = [x+1.34*len(unc_pits)**(-0.5) for x in pit_support]
unc_ci_l = [x-1.34*len(unc_pits)**(-0.5) for x in pit_support]

# Prepare the plots
fig, (ax0, ax1, ax2) = plt.subplots(1, 3)

# Unconditional distribution
ax0.plot(pit_support, unc_pit_line, color='blue',
        label='Out-of-sample empirical CDF',
        lw=2)
ax0.plot(pit_support, pit_support, color='red', label='Theoretical CDF')
ax0.plot(pit_support, unc_ci_u,color='red',label='5 percent critical values',
        linestyle='dashed')
ax0.plot(pit_support, unc_ci_l, color='red', linestyle='dashed')
ax0.legend(frameon=False, loc='upper left', fontsize=20)
ax0.set_title(f'Unconditional PIT Test \n KS pvalue: {unc_ks_pval}%',
             y=1.02)

# KDE
ax1.plot(pit_support, kde_pit_line, color='blue',
        label='Out-of-sample empirical CDF',
        lw=2)
ax1.plot(pit_support, pit_support, color='red', label='Theoretical CDF')
ax1.plot(pit_support, kde_ci_u, color='red', label='5 percent critical values',
        linestyle='dashed')
ax1.plot(pit_support, kde_ci_l, color='red', linestyle='dashed')
ax1.set_title(f'KDE PIT Test \n KS pvalue: {kde_ks_pval}%',
             y=1.02)

# Parametric
ax2.plot(pit_support, param_pit_line, color='blue',
        label='Out-of-sample empirical CDF',
        lw=2)
ax2.plot(pit_support, pit_support, color='red', label='Theoretical CDF')
ax2.plot(pit_support, param_ci_u,color='red',label='5 percent critical values',
        linestyle='dashed')
ax2.plot(pit_support, param_ci_l, color='red', linestyle='dashed')
ax2.set_title(f'Parametric PIT Test \n KS pvalue: {param_ks_pval}%',
             y=1.02)

# Save the chart
density_plots = os.path.join('output', 'step_005_pit_test.pdf')
plt.savefig(density_plots)

plt.show()

plt.close('all')

###############################################################################
#%% Log score comparisons via Diebold Mariano test statistic
###############################################################################
def logscore_diff(mod1, mod2, dperf, title="Mod 1 vs Mod2"):
    dperfn = dperf[[mod1, mod2]].dropna().copy()
    model_diff = np.log(dperfn[mod1]) - np.log(dperfn[mod2])
    model_diff = model_diff[np.isinf(model_diff)==False]
    norm_factor = np.sqrt(np.nanvar(model_diff)/len(dperfn))
    tt = np.nanmean(model_diff)/norm_factor # Follows a N(0,1)
    pval = 1-stats.norm.cdf(tt, 0, 1) # Two-sided test
    print(f'{title} test statistic: {tt:.3f}, pval:{pval:.3f}')
    
logscore_diff('kde_score', 'unc_score', dperf,
              title='KDE against Unconditional')

logscore_diff('parametric_score', 'unc_score', dperf,
              title='Parametric against Unconditional')

logscore_diff('kde_score', 'parametric_score', dperf,
              title='KDE against parametric')

# Basically, in terms of performance:
# KDE > Parametric > Unconditional
    
###############################################################################
#%% Entropy, downside entropy, upside entropy, relative entropy
###############################################################################
# Look at the online appendix of Vulnerable Growth
# https://www.aeaweb.org/content/file?id=9245

# Definition on samples, convenient: we do have samples each time !!
from scipy.stats import entropy

dates_l = sorted(set(dsample.index))

cols_l = ['parametric_entropy', 'parametric_entropy_down',
          'parametric_entropy_up', 'kde_entropy', 'kde_entropy_down',
          'kde_entropy_up', 'relative_entropy_param_kde']


depy = pd.DataFrame(index=dates_l, columns=cols_l) # Container
for fd in dates_l:
    # Prepare the data, up and down as well
    ds = dsample.loc[fd, :].copy()
    mv = np.mean(ds['forecasted_val'])
    ds_up = ds.loc[ds['forecasted_val'] > mv, :].copy()
    ds_down = ds.loc[ds['forecasted_val'] <= mv, :].copy()

    # Compute entropy metrics using scipy.stats    
    depy.loc[fd, 'parametric_entropy'] = entropy(ds['parametric_pdf'])
    depy.loc[fd, 'parametric_entropy_up'] = entropy(ds_up['parametric_pdf'])
    depy.loc[fd,'parametric_entropy_down'] = entropy(ds_down['parametric_pdf'])

    depy.loc[fd, 'kde_entropy'] = entropy(ds['kde_pdf'])
    depy.loc[fd, 'kde_entropy_up'] = entropy(ds_up['kde_pdf'])
    depy.loc[fd, 'kde_entropy_down'] = entropy(ds_down['kde_pdf'])
    
###############################################################################
#%% Entropy charts, as in Vulnerable Growth (AER 2019)
###############################################################################

# Prepare the plot
fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
ax1, ax2, ax3, ax4 = axs.ravel()

# Entropy on full sample
ax1.plot(depy['parametric_entropy'], label='Entropy for parametric dist',
         lw=4, ls='-', color=color_l[0])
ax1.plot(depy['kde_entropy'], label='Entropy for kde',
         lw=4, ls='--', color=color_l[1])
ax1.legend(**legend_d)


# Downside entropy
ax2.plot(depy['parametric_entropy_down'],
         label='Downside entropy parametric dist',
         lw=4, ls='-', color=color_l[0])
ax2.plot(depy['kde_entropy_down'], label='Downside entropy kde',
         lw=4, ls='--', color=color_l[1])
ax2.legend(**legend_d, loc='lower right')

# Upside entropy
ax3.plot(depy['parametric_entropy_up'],
         label='Upside entropy parametric dist',
         lw=4, ls='-', color=color_l[0])
ax3.plot(depy['kde_entropy_up'], label='Upside entropy kde',
         lw=4, ls='--', color=color_l[1])
ax3.legend(**legend_d, loc='lower right')

# hide axes
fig.patch.set_visible(False)
ax4.axis('off')
ax4.axis('tight')


# Title
fig.suptitle('Entropy Metrics for the Parametric and KDE fits')


# Layout
fig.autofmt_xdate() # Adjust the dates automatically
fig.set_size_inches(25, 16)
#fig.tight_layout()
fig.subplots_adjust(top=0.88, bottom=0.15, left=0.05, right=0.95)

# Save the figure
fig.savefig(os.path.join('output', 'step_005_entropy_metrics.pdf'))
plt.show()
plt.close('all')
   
###############################################################################
#%% End 
###############################################################################



















































