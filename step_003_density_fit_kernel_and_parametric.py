# -*- coding: utf-8 -*-
"""
Density fit
Romain Lafarguette, https://romainlafarguette.github.io/
Time-stamp: "2021-05-17 19:13:58 RLafarguette"
Density fit is very easy, as it is already packaged on sklearn and scipy
Different from Adrian et al. (2019) who are fitting conditional quantiles
Here we work at the sample level, so we have many more choices
"""

###############################################################################
#%% Preamble
###############################################################################
# Main modules
import os, sys, importlib                                # System packages
import pandas as pd                                      # Dataframes 
import numpy as np                                       # Numerical tools
import scipy                                             # Scientific Python
import warnings                                          # Warnings management

# Functional imports
from collections import namedtuple                       # Factory functions
from sklearn.neighbors import KernelDensity              # Machine learning

# Graphics - Special frontend compatible with Emacs, but can be taken out
import matplotlib                                          # Graphical packages
matplotlib.use('TkAgg') # You can comment it 
import matplotlib.pyplot as plt
import seaborn as sns                                      # Graphical tools
sns.set(style="white", font_scale=2)  # set style

# Local modules
sys.path.append(os.path.abspath('modules'))
import joyplot; importlib.reload(joyplot)                # Nice density plots 
from joyplot import joyplot                            

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
y_val = ds.loc[ds['horizon']==1, 'gdp_growth'].values

y_range = max(y_val) - min(y_val); y_rhalf = 0.5*y_range # Boundaries
y_support = np.linspace(min(y_val) - y_rhalf, max(y_val) + y_rhalf,len_support)

###############################################################################
#%% Distplot chart with seaborn
## Everything is automatic here, see below for decomposition/customization
###############################################################################
y_val = ds.loc[ds['horizon']==1, 'gdp_growth'].values 
fig, ax1 = plt.subplots()
ax1 = sns.distplot(y_val, ax=ax1)
ax1.set_xlabel('GDP growth', labelpad=20)
ax1.set_ylabel('pdf', labelpad=20)
ax1.set_title('Gaussian Kernel for GDP Q+1', y=1.02)
plt.tight_layout()

# Save in the output folder (can be pdf, jpg, etc.)
gaussian_kernel = os.path.join('output', 'step_003_hist_gauss_kernel.pdf')
plt.savefig(gaussian_kernel)

plt.show()
plt.close('all')

###############################################################################
#%% Joyplot (superposition of Kernel fit at different horizons)
###############################################################################
import joyplot; importlib.reload(joyplot)                # Nice density plots 
from joyplot import joyplot                            

cmap = matplotlib.cm.get_cmap('autumn') # Reversed ("_r")
label_l = [x for x in ds.horizon]

# Invert the order (I should code these feature directly in the module...)
ds['horizon_inv'] = 1 + max(ds['horizon']) - ds['horizon']
label_l = [f'+{x}Q' for x in sorted(set(ds['horizon']))][::-1]

# Joyplot
ax = joyplot(ds, by="horizon_inv", column="gdp_growth", range_style='own',
             figsize=(20,15), x_range=(-2, 2), 
             grid="y", linewidth=1, legend=False, fade=True,             
             labels=label_l, colormap=cmap)
plt.title('Term Structure of GDP')

# Save in the output folder (can be pdf, jpg, etc.)
joyplot = os.path.join('output', 'step_003_joyplot.pdf')
plt.savefig(joyplot)

plt.show()
plt.close('all')

###############################################################################
#%% Non parametric Kernel fit (Gaussian)
###############################################################################
# Gaussian Kernel fit
# See: https://jakevdp.github.io/blog/2013/12/01/kernel-density-estimation/
# For more details and options to change the kernel
fhorizon = 1
y_val = ds.loc[ds['horizon']==fhorizon, 'gdp_growth'].values
y_kde = scipy.stats.gaussian_kde(y_val) # This is a function
y_kernel_pdf = y_kde(y_support)
default_bw = y_kde.scotts_factor() # Default used by Scipy

# Plot the non parametric fit
fig, ax = plt.subplots()
ax.plot(y_support, y_kernel_pdf, label='Default bandwidth',
        ls='-', lw=3, color='navy')
ax.set_xlabel('GDP percentage', labelpad=20)
ax.set_ylabel('Probability distribution function', labelpad=20)
ax.legend()
ax.set_title('Gaussian Kernel Fit')

# Save in the output folder (can be pdf, jpg, etc.)
kernel = os.path.join('output', 'step_003_kernel_fit.pdf')
plt.savefig(kernel)

plt.show()
plt.close('all')

# For comparison, decreases the bandwitdh (increases overfit)
y_kde_small_bw = scipy.stats.gaussian_kde(y_val, default_bw/8) 
y_kernel_pdf_small_bw = y_kde_small_bw(y_support)

# Bandwidth comparison
fig, ax = plt.subplots()
ax.plot(y_support, y_kernel_pdf, label='Default bandwidth',
        ls='-', lw=3, color='navy')
ax.plot(y_support, y_kernel_pdf_small_bw, label='Small bandwidth',
        ls='--', lw=3, color='firebrick')
ax.set_xlabel('GDP percentage', labelpad=20)
ax.set_ylabel('Probability distribution function', labelpad=20)
ax.legend()
ax.set_title('Gaussian Kernel Fit with Different Bandwidths')

# Save in the output folder (can be pdf, jpg, etc.)
bandwith = os.path.join('output', 'step_003_bandwith_comp.pdf')
plt.savefig(bandwith)

plt.show()
plt.close('all')

###############################################################################
#%% Choice of the best parametric family
###############################################################################
dist_rv_d = dict() # Container to store the distribution in Scipy form
aic_d = dict() # Container for AIC criteria
bic_d = dict() # Container for BIC criteria
rss_d = dict() # Container for Residual Sum of Squares criteria

# 101 distributions availables on scipy !
from scipy.stats._continuous_distns import _distn_names

print(_distn_names)

# But only work on a subset of potential candidates.
# Please add more if you like
dist_l = ['beta', 'chi2', 'exponnorm', 'gamma', 'gennorm',
          'laplace', 'wald',
          'weibull_min', 'logistic'          
          ,'t','norm', 'cauchy', 'nct'       #t-family
          ,'pareto','lomax','genpareto'        #paretos family
          ,'gumbel_r','gumbel_l','levy','levy_l','skewnorm'  #skew familiy
          ]

for dist_name in dist_l: # Try all distributions available

    # Ignore warnings from data that can't be fit
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
    
    dist = getattr(scipy.stats, dist_name) # The dist object
    dist_params = dist.fit(y_val)
    dist_rv = dist(*dist_params) # Create the random variable
    dist_rv_d[dist_name] = dist_rv
    
    # Information criteria (lower the score, better the model)
    log_likelihood = dist.logpdf(y_val, *dist_params).sum()
    aic = 2*len(dist_params) - (2*log_likelihood)
    bic = len(dist_params)*np.log(len(y_val)) - (2*log_likelihood)

    # Residual sum of squares (RSS)
    # Can minimize either on the histogram or on the kernel, choosing kernel
    # I have chosen the kernel with small badnwdith for increased fit
    dist_pdf = dist_rv.pdf(y_support)
    rss = np.sum(np.power((y_kernel_pdf_small_bw - dist_pdf),2))

    # Store
    aic_d[dist_name] = aic
    bic_d[dist_name] = bic
    rss_d[dist_name] = rss


# Minimize the information criteria
aic_best = [k for k in aic_d if aic_d[k] == min(aic_d.values())] 
bic_best = [k for k in bic_d if bic_d[k] == min(bic_d.values())] 
rss_best = [k for k in rss_d if rss_d[k] == min(rss_d.values())]

(f'Best family according to:'
      f'\n AIC:{aic_best} \n bic:{bic_best} \n RSS:{rss_best}')

# Either beta or lomax
aic_best_rv = dist_rv_d[aic_best[0]]
bic_best_rv = dist_rv_d[bic_best[0]] # Prefer BIC to AIC, as it penalizes complexity better
rss_best_rv = dist_rv_d[rss_best[0]]

###############################################################################
#%% Plots on best parametric fit
###############################################################################
fig, (ax1, ax2) = plt.subplots(2, 1)
fig.suptitle(f'GDP (%Var) {fhorizon} periods ahead', fontsize='medium') 

ax1.plot(y_support, y_kernel_pdf, label='Non-parametric fit',
         ls='--', lw=3, color='firebrick')
ax1.plot(y_support, aic_best_rv.pdf(y_support),
         label=f'{aic_best[0]} distribution parametric fit',
         ls='-', lw=3, color='navy')
ax1.legend(loc='best', fontsize='x-small', frameon=False)
ax1.set_title(f'bic best fit : {bic_best[0]} distribution', fontsize='small')
ax1.get_xaxis().set_visible(False)

ax2.plot(y_support, y_kernel_pdf, label='Non-parametric fit',
         ls='--', lw=3, color='firebrick')
ax2.plot(y_support, rss_best_rv.pdf(y_support),
         label=f'{rss_best[0]} distribution parametric fit',
         ls='-', lw=3, color='navy')
ax2.legend(loc='best', fontsize='x-small', frameon=False)
ax2.set_title(f'rss best fit : {rss_best[0]} distribution', fontsize='small')

# Save it
best_parametric = os.path.join('output', 'step_003_best_fit.pdf')
plt.savefig(best_parametric)

plt.show()

# OK lomax looks much better !

###############################################################################
#%% Standard skewed (like in the GaR paper) => doesn't work well !
###############################################################################
fhorizon = 1
y_val = ds.loc[ds['horizon']==fhorizon, 'gdp_growth'].values
params = scipy.stats.skewnorm.fit(y_val) # Estimate the parameters via MLE
nsk_rv = scipy.stats.skewnorm(*params) # Create the random variable
y_skew_pdf = nsk_rv.pdf(y_support)

# Example: horizon 4, plot the PDF
fig, ax = plt.subplots()
ax.plot(y_support, y_skew_pdf, label='Skewed pdf',
        ls='-', lw=3, color='navy')

params_d = {'a':params[0], 'loc':params[1], 'scale':params[2]}
q5 = scipy.stats.skewnorm.ppf(q=0.05, **params_d)
q10 = scipy.stats.skewnorm.ppf(q=0.1, **params_d)
q90 = scipy.stats.skewnorm.ppf(q=0.9, **params_d)
q95 = scipy.stats.skewnorm.ppf(q=0.95, **params_d)
Median = scipy.stats.skewnorm.ppf(q=0.5, **params_d)
Mean = scipy.stats.skewnorm.mean(**params_d)

ax.set_xlabel('GDP percentage', labelpad=20)
ax.set_ylabel('Probability distribution function', labelpad=20)
ax.set_title(f'Skewed GDP Distribution fit at {fhorizon}-quarters',
             y=1.02)

# Save in the output folder (can be pdf, jpg, etc.)
tskew_plot = os.path.join('output', 'step_003_skew_fit.pdf')
plt.savefig(tskew_plot)

plt.show()

# Quantiles
quant_1 = {'Names':  ["q5", "q10", "Mean", "Median", "q90", "q95"],
                'Quantiles':  [q5, q10, Mean, Median, q90, q95]}
quant_summary = pd.DataFrame (quant_1, columns = ['Names','Quantiles'])

# Save in the output folder (can be pdf, jpg, etc.)
pca_transf_p = os.path.join('data', 'clean', 'step_003_quantiles.csv')
quant_summary.to_csv(pca_transf_p, encoding='utf-8', index=True)

###############################################################################
#%% End
###############################################################################



















