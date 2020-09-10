# -*- coding: utf-8 -*-
"""
Tskew conditional fitting (following Giot and Laurent 2003)
Contact: rlafarguette@imf.org
Time-stamp: "2019-05-14 22:02:37 RLafarguette"
"""

###############################################################################
#%% Modules import
###############################################################################
## Globals
import os, sys, importlib                               # System packages
import numpy as np                                      # Numeric Python

## Locals (reload when editing on the fly)
import sampling; importlib.reload(sampling)             # Sampling tools
import tskew; importlib.reload(tskew)                   # Tskew distribution

# Functions
from sampling import quantiles_uncrossing
from tskew import tskew_ppf
from scipy.optimize import minimize

###############################################################################
#%% Euclidean distance
###############################################################################

# Objective function: distance between quantiles
# Euclidean distance
def euclidean_tskew_dist(cond_quant_dict, tsk_params):
    """ 
    Euclidean distance between theoretical and empirical quantiles 
    
    NB: tsk_params should contains: 
    tsk_params = {'df':df, 'loc': loc, 'scale': scale, 'skew':skew}
    """
    
    total_distance = 0 # Initialize
    
    for tau in cond_quant_dict.keys(): # For each quantile
        empirical_quant = cond_quant_dict[tau] 
        theoretical_quant = tskew_ppf(tau, **tsk_params) 
        distance = np.power(empirical_quant - theoretical_quant, 2)
        total_distance = total_distance + distance # Update
        
    return(total_distance)    


###############################################################################
#%% Optimal TSkew fit based on a set of conditional quantiles and a location
###############################################################################
def tskew_fit(cq_dict, method='linear'):
    """ 
    Optimal TSkew fit based on a set of conditional quantiles
    Inputs:
        - cq_dict (dictionary): quantiles 
        - method: optional, str, either 'linear' or probabilistic

    Output:
        - A dictionary with optimal scale and skewness, as well as df and loc 
    """

    # Uncross the quantiles
    cq_dict = quantiles_uncrossing(cq_dict, method=method)

    # Extract the quantiles list
    q_list = sorted(cq_dict.keys())
    
    ## Target function: should depends only on a vector of parameters
    def target_distance(x): # x is a vector
        """ Multiple parameters estimation """
        
        ## Unpack the vector
        loc = x[0]
        scale = x[1]
        skew = x[2]

        tsk_params = {'df': 2, 'loc': loc, 'scale': scale, 'skew': skew}
        
        # Compute the distance relative to the tskew
        distance = euclidean_tskew_dist(cq_dict, tsk_params)
        return(distance)

    
    ## Initial values and boundaries
    # Interquartile range (proxy for volatility)
    try:
        IQR = np.absolute(cq_dict[0.75] -
                          cq_dict[0.25])
        IQR = np.clip(IQR, 1, 10) # Avoid degenerate interquartile range
        # At least 1 pp growth and at most 10 ppt growth in the interquartile
    except:
        raise ValueError('Need to provide estimate for 25% and 75% quantiles')

    # Location: avoid degenerate distributions, impose to lie within IQR
    location_up = cq_dict[0.75]
    location_down = cq_dict[0.25]
    location_start = cq_dict[0.5]
    
    # location_up = 10
    # location_down = -10
    # location_start = cq_dict[0.5]

    
    # Scale (variance)
    scale_up = IQR/1.63 + 0.2 # When skew=1, variance exactly = IQR/1.63
    scale_down = np.sqrt(IQR)/2 + 0.1 # Good lower bound approximation
    scale_start = (scale_up + scale_down)/2
    
    # scale_up = 10 # When skew=1, variance exactly = IQR/1.63
    # scale_down = 0.1 # Good lower bound approximation
    # scale_start = 1
        
    # Skewness
    skew_up = 5 # Good upper band approximation
    skew_down = 0.1 # Good lower band approximation
    skew_start = 1 # No skewness

    # Initial values
    x0_f = [location_start, scale_start, skew_start]

    # Bounds
    bnds_f = ((location_down, location_up),
              (scale_down, scale_up),
              (skew_down , skew_up))

    ## Optimization
    # Run the optimizer with boundaries
    res = minimize(target_distance, x0=x0_f,
                   bounds=bnds_f, method='SLSQP',
                   options={'maxiter':1000,  'ftol': 1e-04, 'eps': 1.5e-06})

    
    # Package the results into a dictionary
    fit_dict = {'loc': float("{:.4f}".format(res.x[0])),
                'df': 2,
                'scale': float("{:.4f}".format(res.x[1])),
                'skew': float("{:.4f}".format(res.x[2]))}
    
    return(fit_dict)

