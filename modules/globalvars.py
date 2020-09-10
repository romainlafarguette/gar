# -*- coding: utf-8 -*-
"""
This module contains the global variables for the GaR project on Israel
Contact: rlafarguette@imf.org
Time-stamp: "2019-05-08 17:16:16 RLafarguette"
"""

##############################################################################
#%% Modules
##############################################################################
import os,sys                                       ## System packages
import numpy as np                                  ## Numerical Python

###############################################################################
#%% Define relative paths
###############################################################################

## Root dir, defined as a relative path from a string
basename = 'Israel GaR'
root_dir = os.path.join(os.getcwd().split(basename)[0], basename)

## First level directories: data, codes, output
data_dir = os.path.join(root_dir, 'Data')
output_dir = os.path.join(root_dir, 'Output')
codes_dir = os.path.join(root_dir, 'Codes')

## Data > second level directories
raw_data_dir = os.path.join(data_dir, 'Raw')
clean_data_dir = os.path.join(data_dir, 'Clean')
final_data_dir = os.path.join(data_dir, 'Final')

## Codes > second level directories
modules_dir = os.path.join(codes_dir, 'Modules')
analytical_dir = os.path.join(codes_dir, 'Analytical')
codes_figures_dir = os.path.join(codes_dir, 'Figures')

## Output > second level directories
partitions_dir = os.path.join(output_dir, 'Partitions')
quantileregs_dir = os.path.join(output_dir, 'QuantileRegressions')

###############################################################################
#%% Global functions
###############################################################################
def log_ret(series, horizon=1):
    lr = np.log(series) - np.log(series(horizon))
    return(lr)

def future_CAGR(series, horizon, freq=4):
    """
    Compound annualized quarterly growth rate over a certain horizon
    """
    cagr = ((series.shift(-horizon)/series)**(1/horizon)) - 1
    
    ## Need to annualize the growth rate, based on the frequency
    annual_cagr = ((1+cagr)**freq) - 1
    return(annual_cagr)


###############################################################################
#%% Some parameters
###############################################################################
horizon_l = [1, 4, 8, 12]
quantiles_l = [0.05, 0.25, 0.5, 0.75, 0.95]
