# -*- coding: utf-8 -*-
"""
Plotting module for the GaR project
Contact: rlafarguette@imf.org
Time-stamp: "2019-03-25 17:45:37 RLafarguette"
"""

###############################################################################
#%% Modules import
###############################################################################
## Globals
import pandas as pd                                     ## Dataframes
import matplotlib.pyplot as plt                         ## Charts

## Functional imports: manage the formatting
from matplotlib.ticker import FormatStrFormatter

###############################################################################
#%% Ancillary functions
###############################################################################
## Round only if number
def round_if_num(x, digits=2):
    try:
        round_x = round(float(x),digits)
    except:
        round_x = x
    return(round_x)

###############################################################################
#%% Quantile plots
###############################################################################
def single_coeff_plot(coeff_frame, variable, ax):
    
    """ Plot the quantile coefficients for a given variable """
    
    ## Initialization (depends if ax has been supplied or not)
    plt.sca(ax) # Important
        
    ## Clean the frame
    dcoeffc = coeff_frame.loc[variable,:].copy()
    dcoeffc['tau'] = dcoeffc['tau'].apply(round_if_num).astype(str).copy()
    dcoeffc = dcoeffc.set_index(dcoeffc['tau'])

    ## Manage the index with "mean" next to the median
    qlist = list(dcoeffc['tau'])
    qlist_num = [x for x in qlist if x!= 'mean']

    med_index = qlist_num.index('0.5')
    qlist_num.insert(med_index, 'mean')
    dcoeffc = dcoeffc.reindex(qlist_num).copy()

    ## Compute the error terms
    dcoeffc['errors'] = (dcoeffc['upper'] - dcoeffc['lower'])/2

    ## Barplot with error terms
    dcoeffc['coeff'].plot.bar(color='blue',
                              yerr=dcoeffc.errors, axes=ax)
    
    ## Some fine-tuning
    ax.axhline(y=0, c='black', linewidth=0.7)
    ax.set_title('{0}'.format(variable), fontsize=25, y=1.05)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.set_xlabel('')

    ## Don't return anything

def r2_plot(coeff_frame, ax):
    
    """ Plot the quantile coefficients for a given variable """
    
    ## Initialization (depends if ax has been supplied or not)
    plt.sca(ax)
    
    ## Clean the frame
    dr2 = coeff_frame.loc['Intercept',:].copy()
    dr2['tau'] = dr2['tau'].apply(round_if_num).astype(str).copy()
    dr2 = dr2.set_index(dr2['tau'])

    ## Manage the index with "mean" next to the median
    qlist = list(dr2['tau'])
    qlist_num = [x for x in qlist if x!= 'mean']

    med_index = qlist_num.index('0.5')
    qlist_num.insert(med_index, 'mean')
    dr2 = dr2.reindex(qlist_num).copy()

    ## Barplot 
    dr2['R2_in_sample'].plot.bar(color='blue', axes=ax)
    
    ## Some fine-tuning
    ax.axhline(y=0, c='black', linewidth=0.7)
    ax.set_title('R-squared', fontsize=25, y=1.05)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.set_xlabel('')

    ## Don't return anything
