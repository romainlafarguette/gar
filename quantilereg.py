# -*- coding: utf-8 -*-
"""
Estimate quantiles regressions, retrieve coefficients and conditional quantiles
rlafarguette@imf.org
Time-stamp: "2019-06-19 23:00:55 Romain"
Edit by cwang2@imf.org for also outputing Intercept
"""

###############################################################################
#%% Modules import
###############################################################################
## Globals
import importlib
import pandas as pd                                     ## Dataframes
import numpy as np                                      ## Numeric methods
import statsmodels as sm                                ## Statistical models
import statsmodels.formula.api as smf                   ## Formulas
import time                                             ## Processing time
from sklearn.preprocessing import scale                 ## Scale the variables 

## Global plotting modules
import matplotlib.pyplot as plt                         ## Pyplot
from matplotlib import gridspec                         ## Plotting grid

## Local plotting functions (local modules in the same directory)
import garplots; importlib.reload(garplots)             ## Plot functions
from garplots import single_coeff_plot, r2_plot         ## Plot coeffs

###############################################################################
#%% Ancillary functions
###############################################################################
## Break a list into sublists of length n
def sublist_chunks(long_list, n):
    sublists_l = [long_list[i:i + n]  for i in range(0, len(long_list), n)]
    return(sublists_l)

###############################################################################
#%% Class for the Quantiles Regressions
###############################################################################
class QuantileReg(object):
    """ 
    Fit a conditional regression model, via quantile regressions

    Inputs:
    - depvar: string, dependent variable 
    - indvars: list of independent variables
    - quantile_list: list of quantiles to run the fit on
    - data = data to train the model on
    - scaling: zscore of the variables: standardized coefficients
    - alpha: the level of confidence to compute the confidence intervals
    
    Output:
    - qfit_dict = regressions fit, per quantiles (statsmodels object)
    - mfit = OLS regression fit, for the conditional mean
    - coeff = coefficients of the quantile regression, for every quantile
    - cond_quant: conditional quantiles and mean 

    Usage:
    qr = QuantileReg('y_growth_fwd_4', indvars=p_indvars, quantile_list=ql,
                     data=df, scaling=True, alpha=0.1)

    """
    __description = "Conditional quantiles, based on quantile regressions"
    __author = "Romain Lafarguette, IMF/MCM, rlafarguette@imf.org"

    ## Initializer
    def __init__(self, depvar, indvars, quantile_list, data, scaling=True,
                 alpha=0.1):

        ## Parameters
        self.scaling = scaling
        self.alpha = alpha
        self.quantile_list = quantile_list
        
        ## Variables
        self.depvar = depvar

        ## List of regressors
        self.regressors = [x for x in indvars if x in data.columns]
        
        ## Data cleaning for the regression (no missing in dep and regressors)
        all_vars = [self.depvar] + self.regressors
        self.data = data.loc[:, all_vars].dropna().copy()

        # The predictor frame doesn't need to clean at the depvar level
        clean_data = data.loc[:, self.regressors].dropna().copy()
        clean_index = clean_data.index
        self.pred_data = data.loc[clean_index, :].copy()
        
        ## Formula regression
        self.reg_formula = self.__reg_formula()
        
        ## Depending on user input, scale the variables
        if self.scaling == True:
            self.data.loc[:, all_vars] = scale(self.data[all_vars].copy())
        else:
            pass
        
        ## From class methods (see below)
        self.qfit_dict = self.__qfit_dict()
        self.mfit = self.__mfit()
        self.coeff = self.__coeff()

        ## Conditional quantiles: use as predictors the historical data
        ## In-sample prediction, can be customized to fit counterfactual shocks
        self.cond_quant = self.cond_quantiles(predictors=self.pred_data)
        
    ## Methods
    def __reg_formula(self):
        """ Generate the specification for the quantile regressions """
        regressors_l = self.regressors[0]
        for v in self.regressors[1:]: regressors_l += ' + {0}'.format(v)
        reg_f = '{0} ~ {1}'.format(self.depvar, regressors_l)
        return(reg_f)

    def __qfit_dict(self): 
        """ Estimate the quantile fit for every quantile """
        qfit_dict = dict()
        for tau in self.quantile_list:
            reg_f = self.reg_formula
            qfit = smf.quantreg(formula=reg_f,data=self.data).fit(q=tau,
                                                                  maxiter=2000,
                                                                  p_tol=1e-05)
            qfit_dict[tau] = qfit
        return(qfit_dict)

    def __mfit(self): 
        """ Estimate the OLS fit for every quantile """
        mfit = smf.ols(self.reg_formula, data=self.data).fit()
        return(mfit)
    
    def __coeff(self):
        """ Extract the parameters and package them into pandas dataframe """
        params = pd.DataFrame()
        for tau in self.quantile_list:
            qfit = self.qfit_dict[tau]
            stats = [qfit.params, qfit.pvalues, qfit.conf_int(alpha=self.alpha)]
            stats_names = ['coeff', 'pval', 'lower', 'upper']
            dp = pd.concat(stats, axis=1); dp.columns = stats_names
            dp.insert(0, 'tau', qfit.q) # Insert as a first column
            dp['R2_in_sample'] = qfit.prsquared
            #dp = dp.loc[dp.index != 'Intercept',:].copy()
            ## Add the scaling information
            dp.loc[:,'normalized'] = self.scaling
            params = params.append(dp)
        
        ## For information, coeffs from an OLS regression (conditional mean)
        mfit = self.mfit
        stats = [mfit.params, mfit.pvalues, mfit.conf_int(alpha=self.alpha)]
        stats_names = ['coeff', 'pval', 'lower', 'upper']
        dmp = pd.concat(stats, axis=1); dmp.columns = stats_names
        dmp.insert(0, 'tau', 'mean') # Insert as a first column
        dmp['R2_in_sample'] = mfit.rsquared
        #dmp = dmp.loc[dmp.index != 'Intercept',:].copy()
        ## Add the scaling information
        dmp.loc[:,'normalized'] = self.scaling
        coeff = pd.concat([params, dmp], axis='index')
        
        ## Return the full frame
        return(coeff)
    
    def cond_quantiles(self, predictors):
        """ 
        Estimate the conditional quantiles in sample 
        - Predictors have to be a pandas dataframe with regressors as columns
        """
        cond_quantiles = pd.DataFrame()

        # Clean the frame, to make sure the index will match
        df_pred = predictors.dropna(subset=self.regressors).copy() 
        
        for tau in self.quantile_list:
            qfit = self.qfit_dict[tau]
            # Run the prediction over a predictors frame     
            dc = qfit.get_prediction(exog=df_pred).summary_frame()
            dc.columns = ['conditional_quantile_' + x for x in dc.columns]
            dc = dc.set_index(df_pred.index)
            
            ## Insert extra information
            dc.insert(0, 'tau', tau)
            dc.insert(1, 'realized_value', df_pred.loc[:, self.depvar])    
            cond_quantiles = cond_quantiles.append(dc)
                        
        ## Add the conditional mean
        dm = self.mfit.get_prediction(exog=df_pred).summary_frame()
        dm.columns = ['conditional_quantile_' + x for x in dm.columns]
        dm = dm.set_index(df_pred.index)
        
        # Insert extra information in the frame
        dm.insert(0, 'tau', 'mean')
        dm.insert(1, 'realized_value', df_pred.loc[:, self.depvar])
        
        ## Concatenate both frames
        cq = pd.concat([cond_quantiles, dm])

        return(cq)


    def plot_coeffs(self, title=None, **kwds):
        """ Plot the coefficients with confidence interval and R2 """
        
        rows_list = sublist_chunks(self.regressors, 3)
        
        fig = plt.figure(**kwds)

        ## Define the grid
        gs = gridspec.GridSpec(len(rows_list), 4, 
                               wspace=0.25, hspace=0.5) 

        ## Populate the grid
        for row_index, row in enumerate(rows_list):
            for col_index, variable in enumerate(row):
                ax_ind = fig.add_subplot(gs[row_index, col_index]) 
                single_coeff_plot(self.coeff, variable, ax=ax_ind)

        ax_r2 = fig.add_subplot(gs[:, -1])
        r2_plot(self.coeff, ax=ax_r2)

        title = title or 'Quantile Coefficients and Pseudo R2' 
        plt.suptitle(title, fontsize=28, y=0.999)
        return(fig)


        
    
    

