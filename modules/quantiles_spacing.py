# -*- coding: utf-8 -*-
"""
Quantile Spacing in Numba for fast computation
Romain Lafarguette, https://github.com/romainlafarguette
June 2018
Time-stamp: "2022-02-11 00:50:41 rlafarguette"
"""

###############################################################################
#%% Imports
###############################################################################
# Modules
import numpy as np                                      # Numeric tools
from scipy.optimize import bisect

###############################################################################
#%% Statistics ancillary functions
###############################################################################
def t2_ppf(tau):
    """ Quantile function of a Student distribution with 2 df """    
    alpha = 4*tau*(1-tau)
    quantile_val = 2*(tau - 0.5)*np.sqrt(2/alpha)
    return(quantile_val)

###############################################################################
#%% Single quantile interpolation
###############################################################################
def qs_ppf(alpha, qlist, condqlist, base=t2_ppf):
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
    qlist = np.ravel(qlist) # Needed for indexation
    condqlist = np.ravel(condqlist)
    
    # Extremes
    min_q = np.min(qlist)
    max_q = np.max(qlist)
    
    min_cq = np.min(condqlist)
    max_cq = np.max(condqlist)
    
    # Considering multiple cases    
    if alpha in qlist: # Just return the exact quantile value
        interp = condqlist[np.where(qlist == alpha)]

    # Else, interpolate
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
        
        local_min_cq = condqlist[np.where(qlist == local_min)]
        local_max_cq = condqlist[np.where(qlist == local_max)]

        # Compute the slope
        b_up = (local_max_cq - local_min_cq)
        b_low = base(local_max) - base(local_min)
        b = b_up/b_low

        # Compute the intercept
        a = local_max_cq - b*base(local_max)

        # Compute the interpolated value
        interp = a + b*base(alpha)

    return(float(interp))

###############################################################################
#%% CDF on a single list of conditional quantiles
###############################################################################
def qs_cdf(value, qlist, condqlist, base=t2_ppf, tol=1e-8, maxiter=500):
    """ Quantile Spacing CDF """

    # Make sure that the arrays are 1-D
    # Use ravel, which returns a view, much faster than flatten
    qlist = np.ravel(qlist)    
    condqlist = np.ravel(condqlist)
 
    # Considering multiple cases   
    # If the value is already known (on the quantile)
    if value in condqlist: # Just return the true value        
        proba = condqlist[np.where(qlist == value)]

    else: # Create the root function for a given tau
                        
        # Determine the initial guess
        min_q = np.min(qlist)
        max_q = np.max(qlist)
    
        min_cq = np.min(condqlist)
        max_cq = np.max(condqlist)

        if value < min_cq:
            init_qmin = tol
            init_qmax = min_q

        elif value > max_cq:
            init_qmin = max_q
            init_qmax = 1-tol

        else: # Find the two closest quantiles to initialize the bisection   
            init_cqmin = np.max(condqlist[condqlist<value]) 
            init_cqmax = np.min(condqlist[condqlist>value]) 
            
            init_qmin = condqlist[np.where(qlist == init_cqmin)]
            init_qmax = condqlist[np.where(qlist == init_cqmax)]


        # Wrap the function for optimization
        def wrap_qd(tau):
            return(qs_ppf(tau, qlist, condqlist, base) - value)
            
        # The probability is the root (the CDF) 
        proba = bisect(wrap_qd, init_qmin, init_qmax,
                       xtol=tol, maxiter=maxiter)
        
    return(float(proba))


###############################################################################
#%% PPF on a matrix of conditional quantiles
###############################################################################
def qs_ppf_mv(alpha, qlist, condqlist, base=t2_ppf, tol=1e-8, maxiter=500):
    """ 
    Quantile Spacing CDF on a matrix of conditional quantiles

    The conditioning matrix should be (TAU, S) 
      TAU is the list of quantiles
      S number of conditioning samples

    """

    # Make sure the size is correct
    qlist = np.ravel(qlist)
    
    if condqlist.shape[0] != len(qlist):
        print('Conditioning mat should be (TAU x S), TAU numb of quant')
        raise ValueError

    res = np.empty((condqlist.shape[1],1))
    
    for sample_idx in range(condqlist.shape[1]):
        condqlist_slice = condqlist[:, sample_idx]
        res[:] = qs_ppf(alpha, qlist, condqlist_slice, base=base)

    # Take the average (bootstrapped value) over each conditioning vector
    res_mean = np.mean(res)    
    return(res_mean)


###############################################################################
#%% CDF on a matrix of conditional quantiles
###############################################################################
def qs_cdf_mv(value, qlist, condqlist, base=t2_ppf, tol=1e-8, maxiter=500):
    """ 
    Quantile Spacing CDF on a matrix of conditional quantiles

    The conditioning matrix should be (TAU, S) 
      TAU is the list of quantiles
      S number of conditioning samples

    """

    # Make sure the size is correct
    qlist = np.ravel(qlist)
    
    if condqlist.shape[0] != len(qlist):
        print('Conditioning mat should be (TAU x S), TAU numb of quant')
        raise ValueError

    res = np.empty((condqlist.shape[1],1))
    
    for sample_idx in range(condqlist.shape[1]):
        condqlist_slice = condqlist[:, sample_idx]
        res[:] = qs_cdf(value, qlist, condqlist_slice,
                        base=base, tol=tol, maxiter=maxiter)

    # Take the average (bootstrapped value) over each quantile    
    res_mean = np.mean(res)    
    return(res_mean)


###############################################################################
#%% Sampling
###############################################################################
def qs_sampling(qlist, condqlist, len_sample=1000, base=t2_ppf):
    """ 
    Sampling using the quantile spacing approach 
    
    Inputs
    ------
    qlist: np.array
        Quantiles (between 0 and 1)

    condqlist: np.array
        Conditional quantiles (corresponding to the quantiles above)

    len_sample: int
        Length of the sample to draw
           
    """
    
    # Draw a uniform sample
    U = np.random.rand(len_sample)
    
    # Compute the quantile function on each element of the vector
    sample = np.array([qs_ppf(u, qlist, condqlist, base=base)
                       for u in U], dtype=np.float32) # Float 32 - faster
    
    return(sample)

###############################################################################
#%% Sampling over a conditioning matrix
###############################################################################
def qs_sampling_mv(qlist, condqlist,
                   len_inner_sample=1000,
                   len_sample=1000, base=t2_ppf):
    
    """ 
    Sample from a conditioning matrix
    """
    
    # Make sure the size is correct
    qlist = np.ravel(qlist)
    
    if condqlist.shape[0] != len(qlist):
        print('Conditioning mat should be (TAU x S), TAU numb of quant')
        raise ValueError

    # Prepare the results
    res = np.empty((len_inner_sample, condqlist.shape[1]))
    
    for sample_idx in range(condqlist.shape[1]):
        condqlist_slice = condqlist[:, sample_idx]
        res[:, sample_idx] = qs_sampling(qlist, condqlist_slice,
                                         len_sample=len_inner_sample,
                                         base=base)

    # To avoid the curse of dimensionality, sample from res
    final_sample = np.random.choice(np.ravel(res), size=len_sample)
    
    return(final_sample)


###############################################################################
#%% Conditional sample over a conditioning matrix
###############################################################################
def qs_sampling_cond(qlist, condqlist,
                     len_sample=1000,
                     len_inner_sample=1000, 
                     base=t2_ppf):
    
    """ 
    Sample from a conditioning matrix
    Conditional sampling: only sample one point from each conditioning set
    """
    
    # Make sure the size is correct
    qlist = np.ravel(qlist)
    
    if condqlist.shape[0] != len(qlist):
        # Numba doesn't accept standard assertion tests
        print('Conditioning mat should be (TAU x S), TAU numb of quant')
        raise ValueError 

    # Prepare the results
    sample_l = list()
    
    for sample_idx in range(condqlist.shape[1]):
        condqlist_slice = condqlist[:, sample_idx]
        sample = qs_sampling(qlist, condqlist_slice,
                             len_sample=len_inner_sample,
                             base=base)
        sample_l.append(np.random.choice(sample, 1)[0]) # Only sample 1 point

    final_sample = np.array(sample_l)
        
    return(final_sample)


###############################################################################
#%% Performance
###############################################################################
# I am 700 times faster than scipy !

# from scipy.stats import t  # Tdistribution
# from scipy.stats import norm  # Gaussian distribution
# from timeit import default_timer as timer


# start = timer()
# fullstart = start

# qlist = np.random.uniform(0, 1, 100)

# #for q in qlist: t(2).ppf(q)
# for q in qlist: t2_ppf(q)

# end = timer()
# print("Total time : %.1f ms" % (1000 * (end - fullstart)))

