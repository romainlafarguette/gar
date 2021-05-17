# -*- coding: utf-8 -*-
"""
Quantile Spacing in Numba for fast computation
Romain Lafarguette, https://github.com/romainlafarguette
June 2018
Time-stamp: "2020-11-26 02:07:07 Romain"
"""

###############################################################################
#%% Imports
###############################################################################
# Modules
import numpy as np                                      # Numeric tools
import numba as nb                                      # Python compiler
from numba import jit, vectorize                        # Decorators
from numba import float64, int64                        # Types

###############################################################################
#%% Numba ancillary: Index (equivalent or np.where)
###############################################################################
@jit('int64(float64[:], float64)', nopython=True, fastmath=True)
def index(array, item):
    # Numba doesn't accept "where"
    for idx, val in np.ndenumerate(array):
        if item == val:
            res = idx[0]
    return(res)

###############################################################################
#%% Numba ancillary: Item isin array
###############################################################################
@jit('int64(float64, float64[:])', nopython=True, fastmath=True)
def isin(val, array):
    # Numba by default doesn't accept "in"
    res = False    
    for i in range(array.shape[0]):
        if (array[i]==val):
            res=True
    return(res)

###############################################################################
#%% Statistics ancillary functions
###############################################################################
@jit('float64(float64)', nopython=True, fastmath=True)
def t2_ppf(tau):
    """ Quantile function of a Student distribution with 2 df """    
    alpha = 4*tau*(1-tau)
    quantile_val = 2*(tau - 0.5)*np.sqrt(2/alpha)
    return(quantile_val)

###############################################################################
#%% Quantile Spacing Interpolation
###############################################################################
@jit(nopython=True, fastmath=True) # Can not declare functions as types
def qs_ppf_fast(alpha, qlist, condqlist, base=t2_ppf):
    """ 
    Quantile interpolation function, following Schmidt and Zhu (2016) p12
    - Alpha is the quantile that needs to be interpolated
    - qlist is numpy array of quantile values
    - condqlist in numpy array of conditional quantiles
    Return:
    - The interpolated quantile
    """

    # Make sure that the arrays are 1-D
    # Use ravel, which returns a view, much faster than flatten
    #qlist = np.ravel(qlist)
    #condqlist = np.ravel(condqlist)
    
    # Extremes
    min_q = np.min(qlist)
    max_q = np.max(qlist)
    
    min_cq = np.min(condqlist)
    max_cq = np.max(condqlist)
    
    # Considering multiple cases    
    if isin(alpha, qlist)==True: # Just return the true value
        interp = condqlist[index(qlist, alpha)] # Choose float

    elif alpha < min_q: # Below the minimal quantile
        # Compute the slope (page 13) 
        b1_up = (max_cq - min_cq)
        b1_low = base(max_q) - base(min_q)
        b1 = b1_up/b1_low

        # Compute the intercept (page 12)
        a1 = min_cq - b1*base(min_q)

        # Compute the interpolated value
        interp = a1 + b1*base(alpha)
        
                
    elif alpha > max_q: # Above max quantile (same formula with different qf)
        # Compute the slope (page 13) 
        b1_up = (max_cq - min_cq)
        b1_low = base(max_q) - base(min_q)
        b1 = b1_up/b1_low

        # Compute the intercept (page 12)
        a1 = min_cq - b1*base(min_q)

        # Compute the interpolated value
        interp = a1 + b1*base(alpha)

    else: # In the belly
        # Need to identify the closest quantiles
        local_min = np.max(qlist[qlist<alpha]) # Immediately below
        local_max = np.min(qlist[qlist>alpha]) # Immediately above

        local_min_cq = condqlist[index(qlist, local_min)]
        local_max_cq = condqlist[index(qlist, local_max)]

        # Compute the slope
        b_up = (local_max_cq - local_min_cq)
        b_low = base(local_max) - base(local_min)
        b = b_up/b_low

        # Compute the intercept
        a = local_max_cq - b*base(local_max)

        # Compute the interpolated value
        interp = a + b*base(alpha)

    return(interp) 


###############################################################################
#%% Vectorized version: TODO
###############################################################################



###############################################################################
#%% Test with mock data (uncomment to try)
###############################################################################
# from timeit import default_timer as timer               # Benchmark tiem
# qlist = np.arange(0.05, 1, 0.05)
# condqlist = np.array(sorted(np.random.uniform(-5, 5, len(qlist))))

# U = np.random.uniform(0, 1, 10000)

# start = timer()
# test = [qs_ppf_fast(u, qlist, condqlist) for u in U]
# end = timer()
# print("Total time with Numba: %.3f ms" % (1000 * (end - start)))




