# -*- coding: utf-8 -*-
"""
Asymmetric Tskew conditional fitting (following Zhu and Galbraith JoE 2010)
Contact: rlafarguette@imf.org
Time-stamp: "2019-05-14 22:22:12 RLafarguette"
"""

###############################################################################
#%% Modules import
###############################################################################
## Globals
import os, sys, importlib                               # System packages
import numpy as np                                      # Numeric Python

## Locals (reload when editing on the fly)
import sampling; importlib.reload(sampling)             # Sampling tools
import asymtskew; importlib.reload(asymtskew)           # Asymmetric Tskew

# Functions
from sampling import quantiles_uncrossing
from asymtskew import asymt_ppf
from scipy.optimize import minimize

###############################################################################
#%% Euclidean distance
###############################################################################

# Objective function: distance between quantiles
# Euclidean distance
def euclidean_asymtskew_dist(cond_quant_dict, asymtsk_params):
    """ 
    Euclidean distance between theoretical and empirical quantiles 
    
    NB: asymtsk_params should contains: 
    
    asymtsk_params: alpha=0.5, nu1=1, nu2=1, mu=0, sigma=1
    alpha is the skewness (0.5 no skewness)
    nu1 and nu2 the left and right kurtosis respectively
    mu the location
    sigma the variance
    """
    
    total_distance = 0 # Initialize
    
    for tau in cond_quant_dict.keys(): # For each quantile
        empirical_quant = cond_quant_dict[tau] 
        theoretical_quant = asymt_ppf(tau, **asymtsk_params) 
        distance = np.power(empirical_quant - theoretical_quant, 2)
        total_distance = total_distance + distance # Update
        
    return(total_distance)    

###############################################################################
#%% Optimal TSkew fit based on a set of conditional quantiles
###############################################################################
def asymtskew_fit(cq_dict, method='linear'):
    """ 
    Optimal Asym TSkew fit based on a set of conditional quantiles
    Inputs:
        - cq_dict (dictionary): quantiles 
        - method: optional, str
          either 'linear' or probabilistic
        - location: optional, fix the location of the distribution

    Output:
        - A dictionary with optimal fit parameters
    """

    # Uncross the quantiles
    cq_dict = quantiles_uncrossing(cq_dict, method=method)

    # Extract the quantiles list
    q_list = sorted(cq_dict.keys())
    
    ## Target function depends only on a vector of parameters
    def target_distance(x): # x is a vector
        """ Multiple parameters estimation """
        
        ## Unpack the vector        
        alpha = x[0]
        nu1 = x[1]
        nu2 = x[2]
        mu = x[3]
        sigma = x[4]

        asymtsk_params = {'alpha': alpha, 'nu1': nu1, 'nu2': nu2,
                          'mu': mu, 'sigma': sigma}
        
        # Compute the distance relative to the tskew
        distance = euclidean_asymtskew_dist(cq_dict, asymtsk_params)
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
    
    # Skewness
    alpha_up = 1 # Absolute maximum
    alpha_down = 0 # Absolute minimum
    alpha_start = 0.5 # No skewness

    # Kurtosis (left and right)
    nu1_up = 5
    nu1_down = 0
    nu1_start = 1

    nu2_up = 5
    nu2_down = 0
    nu2_start = 1

    # Location: avoid degenerate distributions, impose to lie within IQR
    mu_up = cq_dict[0.75]
    mu_down = cq_dict[0.25]
    mu_start = cq_dict[0.5]
    
    # sigma (variance)
    sigma_up = 10
    sigma_down = IQR
    sigma_start = IQR*2

    # Initial values
    x0_f = [alpha_start, nu1_start, nu2_start, mu_start, sigma_start]

    # Bounds
    bnds_f = ((alpha_down, alpha_up),
              (nu1_down, nu1_up),
              (nu2_down, nu2_up),              
              (mu_down , mu_up),
              (sigma_down , sigma_up))

    ## Optimization
    # Run the optimizer with boundaries
    res = minimize(target_distance, x0=x0_f,
                   bounds=bnds_f, method='SLSQP',
                   options={'maxiter':1000,  'ftol': 1e-04, 'eps': 1.5e-06})

    # Package the results into a dictionary
    fit_dict = {'alpha': float("{:.4f}".format(res.x[0])),
                'nu1': float("{:.4f}".format(res.x[1])),
                'nu2': float("{:.4f}".format(res.x[2])),
                'mu': float("{:.4f}".format(res.x[3])),
                'sigma': float("{:.4f}".format(res.x[4]))}
    
    return(fit_dict)


###############################################################################
#%% Constrained Asymmetric TSkew fit based on a given location
###############################################################################
def constrained_asymtskew_fit(cq_dict, location, method='linear'):
    """ 
    Constrained Asym TSkew fit based on a set of conditional quantiles
    Inputs:
        - cq_dict (dictionary): quantiles 
        - method: optional, str
          either 'linear' or probabilistic
        - location: float, fix the location of the distribution

    Output:
        - A dictionary with optimal fit parameters
    """

    # Uncross the quantiles
    cq_dict = quantiles_uncrossing(cq_dict, method=method)

    # Extract the quantiles list
    q_list = sorted(cq_dict.keys())
    
    ## Target function depends only on a vector of parameters
    def constrained_target_distance(x): # x is a vector
        """ Multiple parameters estimation with fixed location """
        
        ## Unpack the vector        
        alpha = x[0]
        nu1 = x[1]
        nu2 = x[2]
        sigma = x[3]

        asymtsk_params = {'alpha': alpha, 'nu1': nu1, 'nu2': nu2,
                          'mu': location, 'sigma': sigma}
        
        # Compute the distance relative to the tskew
        distance = euclidean_asymtskew_dist(cq_dict, asymtsk_params)
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
    
    # Skewness
    alpha_up = 1 # Absolute maximum
    alpha_down = 0 # Absolute minimum
    alpha_start = 0.5 # No skewness

    # Kurtosis (left and right)
    nu1_up = 5
    nu1_down = 0
    nu1_start = 1

    nu2_up = 10
    nu2_down = 0
    nu2_start = 1
    
    # sigma (variance)
    sigma_up = 5
    sigma_down = IQR
    sigma_start = IQR*2

    # Initial values
    x0_f = [alpha_start, nu1_start, nu2_start, sigma_start]

    # Bounds
    bnds_f = ((alpha_down, alpha_up),
              (nu1_down, nu1_up),
              (nu2_down, nu2_up),              
              (sigma_down , sigma_up))

    ## Optimization
    # Run the optimizer with boundaries
    res = minimize(constrained_target_distance, x0=x0_f,
                   bounds=bnds_f, method='SLSQP',
                   options={'maxiter':1000,  'ftol': 1e-04, 'eps': 1.5e-06})

    # Package the results into a dictionary
    fit_dict = {'alpha': float("{:.4f}".format(res.x[0])),
                'nu1': float("{:.4f}".format(res.x[1])),
                'nu2': float("{:.4f}".format(res.x[2])),
                'mu': float("{:.4f}".format(location)),
                'sigma': float("{:.4f}".format(res.x[3]))}
    
    return(fit_dict)


