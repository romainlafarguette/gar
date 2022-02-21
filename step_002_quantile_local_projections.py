# -*- coding: utf-8 -*-
"""
Quantiles local projections for the GaR project
Romain Lafarguette, https://romainlafarguette.github.io/
Time-stamp: "2022-02-16 15:57:57 RLafarguette"
"""

###############################################################################
#%% Preamble
###############################################################################
# Main modules
import os, sys, importlib                                # System packages
import pandas as pd                                      # Dataframes 
import numpy as np                                       # Numerical tools

# Graphics - Special frontend compatible with Emacs, but can be taken out
import matplotlib                                          # Graphical packages
matplotlib.use('TkAgg') # You can comment it 
import matplotlib.pyplot as plt
import seaborn as sns                                      # Graphical tools
sns.set(style="white", font_scale=2)  # set style

# Local imports
sys.path.append(os.path.join('modules'))
from quantileproj import QuantileProj         # Quantile projection

###############################################################################
#%% Data loading
###############################################################################
# Original frame
data_path = os.path.join('data', 'raw', 'example_data.csv')
df = pd.read_csv(data_path, parse_dates=['date'], index_col='date')

# PCA and PLS frames
pca_transf_p = os.path.join('data', 'clean', 'step_001_pca_transformed.csv')
dpca = pd.read_csv(pca_transf_p, parse_dates=['date'], index_col=['date'])

pls_transf_p = os.path.join('data', 'clean', 'step_001_pls_transformed.csv')
dpls = pd.read_csv(pls_transf_p, parse_dates=['date'], index_col=['date'])

# Check if there are no different PCA and PLS group (normally it shouldn't)
assert dpca.columns.all() == dpls.columns.all(), "PCA and PLS groups differ"

###############################################################################
#%% Parameters
###############################################################################
# Add the autoregressive term under a different name than gdp_growth
dpca['autoreg'] = dpca['gdp_growth'].copy()
dpls['autoreg'] = dpls['gdp_growth'].copy()

reg_l = [x for x in list(dpca.columns) if x != 'gdp_growth']
horizon_l = [1, 4, 8] # Horizons, in quarters
quantile_l = [0.1, 0.25, 0.5, 0.75, 0.9] # List of quantiles to fit
alpha_ci = 0.05 # Confidence interval threshold

###############################################################################
###############################################################################
#%% PCA
###############################################################################
###############################################################################

###############################################################################
#%% Quantile regressions coefficients on the PCA series
###############################################################################
qp_pca = QuantileProj('gdp_growth', reg_l, dpca, horizon_l) # Init
qp_pca_fit = qp_pca.fit(quantile_l=quantile_l, alpha=alpha_ci) # Qreg

# Matrix of coefficients and summary statistics
print(qp_pca_fit.coeffs.head(50))

# Plot the coefficients
label_d = { # Please change as you like
    'leverage':'Leverage',
    'trade_partners_macro': 'Partners macro conditions',
    'euro_area_fci': 'Euro area FCI',
    'world_fci': 'World FCI',
    'autoreg': 'Autoregressive term',
} 

sns.set(style='white', font_scale=1.5, palette='deep', font='Arial') # Style
qp_pca_fit.plot.coeffs_grid(horizon=4, label_d=label_d,
                            title='Quantile coefficients of the PCA partitions'
                            '1-month horizon \n Confidence interval at 5%')

# Layout
plt.tight_layout()
plt.subplots_adjust(top=0.83, hspace=0.6, wspace=0.4)

# Save in the output folder (can be pdf, jpg, etc.)
pca_qreg_f = os.path.join('output', 'step_002_qreg_coeffs_pca.pdf')
plt.savefig(pca_qreg_f)

plt.show()

###############################################################################
#%% Term structure plots
###############################################################################
# For a given variable: can be done with any other regressor
sns.set(style='white', font_scale=1.5, palette='deep', font='Arial') # Style
qp_pca_fit.plot.term_structure(variable='domestic_fci',
                               title='Term structure of Domestic FCI ' 
                                      'at different horizons')

# Layout
plt.tight_layout()
plt.subplots_adjust(top=0.85, hspace=0.6, wspace=0.4)

# Save in the output folder (can be pdf, jpg, etc.)
pca_ts_fci_f = os.path.join('output',
                                'step_002_term_struct_domestic_fci_pca.pdf')
plt.savefig(pca_ts_fci_f)

plt.show()
plt.close('all')

###############################################################################
#%% Term quantile coefficients
###############################################################################
sns.set(style='white', font_scale=1.5, palette='deep', font='Arial') # Style
qp_pca_fit.plot.term_coefficients(variable='domestic_fci',
                              tau_l=[0.25, 0.5, 0.75], 
                              title='Term Quantile Coefficients for Domestic '
                              'Financial Conditions, at 5% confidence')

# Layout
plt.tight_layout()
plt.subplots_adjust(top=0.85, hspace=0.5, wspace=0.4)

# Save in the output folder (can be pdf, jpg, etc.)
pca_tq_dom_fci_f = os.path.join('output',
                                'step_002_term_quant_external_fci_pca.pdf')
plt.savefig(pca_tq_dom_fci_f)

plt.show()
plt.close('all')

###############################################################################
#%% Projections and fan charts
###############################################################################
# Design a conditioning frame (here last observation for instance)
cond_frame = dpca.loc[[max(df.index)], reg_l].copy()
qp_pca_proj = qp_pca_fit.proj(cond_frame)

#### Conditional quantile plot (plot the forecasted conditional quantiles)
# The forecasting horizon is based on the horizon list provided as parameter
sns.set(style='white', font_scale=1.5, palette='deep', font='Arial') # Style
qp_pca_proj.plot.fitted_quantile(quantile=0.5, ylabel='GDP percentage points')

# Layout
plt.tight_layout()
plt.legend(fontsize='small', frameon=False, handlelength=1)

# Save in the output folder (can be pdf, jpg, etc.)
pca_cq_f = os.path.join('output', 'step_002_forecasted_cond_quant_pca.pdf')
plt.savefig(pca_cq_f)

plt.show()
plt.close('all')

#### Fan chart (from the resampled conditional quantile)
sns.set(style='white', font_scale=1.5, palette='deep', font='Arial') # Style
qp_pca_proj.plot.fan_chart(title='GDP fan chart up to a 2-year horizon: '
                           'PCA regressors',
                           ylabel='GDP percentage points', 
                           len_sample=10000, seed=18041202) # L'Empereur ! :-)
# Fix the legend size
plt.legend(fontsize='small', frameon=False, handlelength=1)

# Layout
plt.tight_layout()

# Save in the output folder (can be pdf, jpg, etc.)
pca_fan_chart_f = os.path.join('output', 'step_002_fan_chart_last_obs_pca.pdf')
plt.savefig(pca_fan_chart_f)

plt.show()
plt.close('all')

###############################################################################
#%% Save the conditional samples from the projection
###############################################################################
dpca_sample = qp_pca_proj.sample()
pca_sample_p = os.path.join('data', 'clean',
                            'step_002_conditional_sample_pca.csv')
dpca_sample.to_csv(pca_sample_p, encoding='utf-8', index=False)

###############################################################################
#%% Can do the same thing with PLS
###############################################################################


###############################################################################
#%% Quantile regressions coefficients on the PLS series
###############################################################################
qp_pls = QuantileProj('gdp_growth', reg_l, dpls, horizon_l) # Init
qp_pls_fit = qp_pls.fit(quantile_l=quantile_l, alpha=alpha_ci) # Qreg

# Matrix of coefficients and summary statistics
print(qp_pls_fit.coeffs.head(50))

# Plot the coefficients
sns.set(style='white', font_scale=1.5, palette='deep', font='Arial') # Style
qp_pls_fit.plot.coeffs_grid(horizon=4, label_d=label_d,
                            title='Quantile coefficients of the PLS partitions'
                            '16-month horizon \n Confidence interval at 5%')

# Layout
plt.tight_layout()
plt.subplots_adjust(top=0.83, hspace=0.6, wspace=0.4)

# Save in the output folder (can be pdf, jpg, etc.)
pls_qreg_f = os.path.join('output', 'step_002_qreg_coeffs_pls.pdf')
plt.savefig(pls_qreg_f)

plt.show()
plt.close('all')

###############################################################################
#%% Term structure plots
###############################################################################
# For all the partitions
for i in reg_l:
    sns.set(style='white', font_scale=1.5, palette='deep', font='Arial') 
    qp_pls_fit.plot.term_structure(variable=i,
                        title=f'Term structure of {i} at different horizons')

    # Layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.85, hspace=0.6, wspace=0.4)

    # Save in the output folder (can be pdf, jpg, etc.)
    pls_ts= os.path.join('output', f'step_002_term_struct_{i}_pls.pdf')
    plt.savefig(pls_ts)

    #plt.show()
    plt.close('all')

###############################################################################
#%% Term quantile coefficients
###############################################################################
#For all the partitions
for j in reg_l:
    sns.set(style='white', font_scale=3, palette='deep', font='Arial') # Style
    qp_pls_fit.plot.term_coefficients(variable=j,
                              tau_l=[0.25, 0.5, 0.75], 
                              title='Term Quantile Coefficients for '+ j+
                              ', at 5% confidence')
    # Layout
    plt.subplots_adjust(top=0.85, hspace=0.5, wspace=0.4)

    # Save in the output folder (can be pdf, jpg, etc.)
    pls_tq = os.path.join('output',
                                f'step_002_term_quant_{j}_pls.pdf')
    plt.savefig(pls_tq)

    #plt.show()
    plt.close('all')   
    
###############################################################################
#%% Projections and fan charts
###############################################################################
# Design a conditioning frame (here last observation for instance)
cond_frame = dpls.loc[[max(df.index)], reg_l].copy()
qp_pls_proj = qp_pls_fit.proj(cond_frame)

#### Conditional quantile plot (plot the forecasted conditional quantiles)
# The forecasting horizon is based on the horizon list provided as parameter
sns.set(style='white', font_scale=1.5, palette='deep', font='Arial') # Style
qp_pls_proj.plot.fitted_quantile(quantile=0.5, ylabel='GDP percentage points')

# Layout
plt.legend(fontsize='small', frameon=False, handlelength=1)
plt.tight_layout()

# Save in the output folder (can be pdf, jpg, etc.)
pls_cq_f = os.path.join('output', 'step_002_forecasted_cond_quant_pls.pdf')
plt.savefig(pls_cq_f)

plt.show()
plt.close('all')

#### Fan chart (from the resampled conditional quantile)
sns.set(style='white', font_scale=1.5, palette='deep', font='Arial') # Style
qp_pls_proj.plot.fan_chart(title='GDP fan chart up to a 3-year horizon: '
                           'PLS regressors',
                           ylabel='GDP percentage points', 
                           len_sample=10000, seed=18041202) # L'Empereur ! :-)
# Fix the legend size
plt.legend(fontsize='small', frameon=False, handlelength=1)

# Layout
plt.tight_layout()

# Save in the output folder (can be pdf, jpg, etc.)
pls_fan_chart_f = os.path.join('output', 'step_002_fan_chart_last_obs_pls.pdf')
plt.savefig(pls_fan_chart_f)

plt.show()
plt.close('all')


###############################################################################
#%% Save the conditional samples from the projection
###############################################################################
dpls_sample = qp_pls_proj.sample()
pls_sample_p = os.path.join('data', 'clean',
                            'step_002_conditional_sample_pls.csv')
dpls_sample.to_csv(pls_sample_p, encoding='utf-8', index=False)

###############################################################################
#%% Save the coefficients
###############################################################################
pls_fit_p = os.path.join('data', 'clean', 'step_002_pca_coeffs.csv')
qp_pca_fit.coeffs.to_csv(pls_fit_p, encoding='utf-8', index=True)

pls_transf_p = os.path.join('data', 'clean', 'step_002_pls_coeffs.csv')
qp_pls_fit.coeffs.to_csv(pls_transf_p, encoding='utf-8', index=True)

###############################################################################
#%% End
###############################################################################







