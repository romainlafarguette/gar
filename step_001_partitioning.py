# -*- coding: utf-8 -*-
"""
Partitioning for the GaR project
Romain Lafarguette, https://romainlafarguette.github.io/
Time-stamp: "2021-05-17 17:06:46 RLafarguette"
"""

###############################################################################
#%% Preamble
###############################################################################
# Main modules
import os, sys, importlib                                # System packages
import pandas as pd                                      # Dataframes 
import numpy as np                                       # Numerical tools

# Functional imports
from sklearn.decomposition import PCA                    # PCA
from sklearn.preprocessing import scale                  # Zscore

# Local imports
sys.path.append(os.path.join('modules', 'plswrapper'))
from plswrapper import PLS                               # My own PLS class 

# Graphics - Special frontend compatible with Emacs, but can be taken out
import matplotlib                                          # Graphical packages
matplotlib.use('TkAgg') # You can comment it 
import matplotlib.pyplot as plt
import seaborn as sns                                      # Graphical tools
sns.set(style="white", font_scale=2)  # set style

###############################################################################
#%% Data loading
###############################################################################
data_path = os.path.join('data', 'raw', 'example_data.csv')
df = pd.read_csv(data_path, parse_dates=['date'], index_col='date')

###############################################################################
#%% Data cleaning
###############################################################################
# 'q_real_gdp_rolling_sum4' is the 4-quarter rolling sum of quarterly GDP
df['rolling_gdp_real_yoy'] = df['q_real_gdp_rolling_sum4'].pct_change(4)
df = df.dropna().copy() # Use chained index if you have short sample variables

# The advantage of this approach is that we have a rolling yearly GDP growth
# At the end of the year, it matches the yearly growth rate
###############################################################################
#%% Variables grouping
###############################################################################
gdp_growth_l = ['rolling_gdp_real_yoy'] # Just one, no partition

domestic_fci_l = ['repo_1W_rate', 'repo_on_rate_diff', 'cpi_inflation_yoy',
                  'eur_all_vol', 'repo_on_vol']

leverage_l = ['credit_gdp', 'share_non_resident_liabilities',
              'loans_deposits_ratio']

trade_partners_macro_l = ['ITA_gdp_yoy', 'ITA_unemployment_rate',
                          'GRE_gdp_yoy', 'GRE_unemployment_rate']

euro_area_fci_l = ['EA_vix', 'ITA_bond_10Y_rate', 'EA_headline_cpi_yoy',
                   'euribor_1W']

world_fci_l = ['vix', 'oil_price']

# Create a dictionary, convenient for labeling later
var_groups_d = {
    'gdp_growth': gdp_growth_l,
    'leverage': leverage_l,
    'trade_partners_macro': trade_partners_macro_l,
    'euro_area_fci': euro_area_fci_l,
    'world_fci': world_fci_l, 
}

# Check if the variables are available in the dataframe
all_vars_l = [x for sublist in var_groups_d.values() for x in sublist]
missing_vars_l = [x for x in all_vars_l if x not in df.columns]
assert len(missing_vars_l)==0, f'{missing_vars_l} are missing in data'

###############################################################################
#%% Aggregate through PCA
###############################################################################
pca_1c = PCA(n_components=1) # Initialize the PCA with one component

pca_fit_l = list() # Container for the PCA fit frames
dpca_transf = pd.DataFrame(index=df.index) # Container for transformed series

# Iterate over each group
for group, var_l in var_groups_d.items():
    
    # Subselect the variables of interest and drop the missing observations
    dfv = df[var_l].dropna() # Can not fit PCA with missing values
    X_scaled = scale(dfv) # Need to scale the variables for a PCA

    # Fit the PCA
    pca_fit = pca_1c.fit(X_scaled) # Fit the PCA
    dfit = pd.DataFrame(pca_fit.components_.T, index=var_l,
                        columns=['loadings'])
    dfit['explained_variance_ratio'] = float(pca_fit.explained_variance_ratio_)
    dfit.insert(0, 'group', group) # Add information about the group
    pca_fit_l.append(dfit) # Store the summary fit frame
    
    # Transform the data and extract the first component
    pca_transform = pca_fit.transform(X_scaled)
    pca_series = pd.Series(pca_transform.ravel(), index=dfv.index)
    dpca_transf.loc[pca_series.index, group] = pca_series # Store it
    
dpca_fit = pd.concat(pca_fit_l)    

# Inspect the two resulting frames
print(dpca_fit.head(50))

print(dpca_transf.dropna(how='all').head(50))

# Save the frames for future use
pca_fit_p = os.path.join('data', 'clean', 'step_001_pca_fit.csv')
dpca_fit.to_csv(pca_fit_p, encoding='utf-8', index=True)

pca_transf_p = os.path.join('data', 'clean', 'step_001_pca_transformed.csv')
dpca_transf.to_csv(pca_transf_p, encoding='utf-8', index=True)

###############################################################################
#%% Aggregate through PLS
###############################################################################
# Define the target dictionary per group, the target can be multivariate
# In this case, I use future GDP growth, one-year ahead, for all...
df['rolling_gdp_real_yoy_fwd4'] = df['rolling_gdp_real_yoy'].shift(-4)

target_groups_d = {
    'gdp_growth': ['rolling_gdp_real_yoy_fwd4'],
    'leverage': ['rolling_gdp_real_yoy_fwd4'],
    'trade_partners_macro': ['rolling_gdp_real_yoy_fwd4'],
    'euro_area_fci': ['rolling_gdp_real_yoy_fwd4'],
    'world_fci': ['rolling_gdp_real_yoy_fwd4'],     
}

# I have written a class which fits the PLS automatically
# Containers to store the results of the PLS computation
pls_fit_l = list()
dpls_transf = pd.DataFrame(index=df.index)

# Iterate over each group
for group, var_l in var_groups_d.items():
    target = target_groups_d[group]
    pls_fit = PLS(target, var_l, df, num_vars='all')

    # Return the summary of the fit
    dfit = pls_fit.summary
    dfit.insert(0, 'group', group) # Insert the group for information
    pls_fit_l.append(dfit)

    # Transform (== predict in-sample) the PLS
    dtransf = pls_fit.predict(df)
    dpls_transf.loc[dtransf.index, group] = dtransf

dpls_fit = pd.concat(pls_fit_l)    

# Inspect the two resulting frames
print(dpls_fit.head(50))

print(dpls_transf.dropna(how='all').head(50))

# Save the frames for future use
pls_fit_p = os.path.join('data', 'clean', 'step_001_pls_fit.csv')
dpls_fit.to_csv(pls_fit_p, encoding='utf-8', index=True)

pls_transf_p = os.path.join('data', 'clean', 'step_001_pls_transformed.csv')
dpls_transf.to_csv(pls_transf_p, encoding='utf-8', index=True)

###############################################################################
#%% Plot the PCA partitions
###############################################################################
plot_cols = var_groups_d.keys()

# 2 axes for 2 subplots
fig, axes = plt.subplots(5,1, figsize=(25,20), sharex=True)
dpca_transf[plot_cols].plot(subplots=True, ax=axes, lw=3)

for ax in axes: # Customize each sub chart (the "ax")
    ax.legend(loc=0, fontsize="small", frameon=False, handlelength=1)
    ax.axvspan('2008-07-31', '2009-12-31', color='grey',alpha=0.5)

# Customize the figure    
plt.xlabel('')
plt.suptitle('Evolution of the Partitions with PCA aggregation', 
             fontsize='large')

# Save in the output folder (can be pdf, jpg, etc.)
pca_ts_all_f = os.path.join('output',
                                'step_001_partitions_pca.pdf')
plt.savefig(pca_ts_all_f)
plt.show()


###############################################################################
#%% Plot the PLS partitions
###############################################################################
plot_cols = target_groups_d.keys()

# 2 axes for 2 subplots
fig, axes = plt.subplots(5,1, figsize=(25,20), sharex=True)
dpls_transf[plot_cols].plot(subplots=True, ax=axes, lw=3)

for ax in axes: # Customize each sub chart (the "ax")
    ax.legend(loc=0, fontsize="small", frameon=False, handlelength=1)
    ax.axvspan('2008-07-31', '2009-12-31', color='grey',alpha=0.5)

# Customize the figure    
plt.xlabel('')
plt.suptitle('Evolution of the Partitions with PLS aggregation', 
             fontsize='large')

# Save in the output folder (can be pdf, jpg, etc.)
pls_ts_all_f = os.path.join('output',
                                'step_001_partitions_pls.pdf')
plt.savefig(pls_ts_all_f)
plt.show()

###############################################################################
#%% End
###############################################################################


