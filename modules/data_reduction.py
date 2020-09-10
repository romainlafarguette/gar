# -*- coding: utf-8 -*-
"""
Run a PLS on a set of variables and target variables
Romain Lafarguette, rlafarguette@imf.org
Time-stamp: "2019-04-05 13:42:28 RLafarguette"
"""

###############################################################################
#%% Modules and methods
###############################################################################
## Modules imports
import pandas as pd                                     ## Data management
import numpy as np                                      ## Numeric tools

## Functional import
from pandas import Timestamp as date                    ## Dates manipulation
from collections import namedtuple                      ## Containers 

## Method imports
from sklearn.decomposition import PCA                   ## PCA
from sklearn.cross_decomposition import PLSRegression   ## PLS
from sklearn.preprocessing import scale                 ## Zscore

## Global plotting modules
import matplotlib.pyplot as plt                         ## Pyplot
from matplotlib import gridspec                         ## Plotting grid

###############################################################################
#%% Ancillary functions 
###############################################################################
def zscore(series):
    """ Return the Z-score of a series """
    zs = (series-series.mean())/series.std(ddof=0)
    return(zs)

def range_index(df, var):
    """ 
    Return the min and max index (range) of data availability per variable 
    """
    dfs_ind = tuple(df.dropna(subset=[var]).index) # tuples are immutable
    return(dfs_ind)

def pca_reduction(regvars, df):
    avl_regs = [x for x in regvars if x in df.columns]
    pca = PCA(n_components=1) # Initialize the PCA with one component
    dfn = df[avl_regs].dropna()
    X = scale(dfn) # Need to sacle the variables for a PCA
    pca_transformation = pca.fit(X).transform(X).ravel()
    pca_series = pd.Series(pca_transformation, index=dfn.index)
    return(pca_series)

def pls_reduction(depvars, regvars, df):
    assert isinstance(depvars, list), 'Dependent variable(s) should be in list'
    avl_regs = [x for x in regvars if x in df.columns]
    pls_series = PLS(depvars, avl_regs, df).component
    return(pls_series)


def num_days(dates_tuple):
    """ Return the number of days in a tuple """
    min_ = pd.to_datetime(min(dates_tuple))
    max_ = pd.to_datetime(max(dates_tuple))
    return((max_ - min_).days)

###############################################################################
#%% PLS Discriminant Analysis Class Wrapper
###############################################################################
class PLS(object):
    """ 
    Data reduction through PLS-discriminant analysis and variables selection 

    Parameters
    ----------
    dep_vars : list; list of dependent variables
    reg_vars : list; list of regressors variables
    data : pandas df; data to train the model on
    num_vars : 'all', integer; number of variables to keep, ranked by VIP
        if 'all': keep all the variables
    
    Return
    ------
    first_component : the first component of the PLS of the Xs reduction
    output_frame : frame containing the variables and their transformation
    summary_frame : frame with the results of the model (loadings, vip, R2)

    """
    __description = "Partial Least Squares with variables selection"
    __author = "Romain Lafarguette, IMF, rlafarguette@imf.org"

    #### Class Initializer
    def __init__(self, dep_vars, reg_vars, data, num_vars='all'):

        #### Attributes
        self.dep_vars = dep_vars
        self.reg_vars = reg_vars
        self.df = data.dropna(subset=self.dep_vars + self.reg_vars)

        ## Put parametrized regression as attribute for consistency
        self.pls1 = PLSRegression(n_components=1, scale=True) # Always scale

        ## Unconstrained fit: consider all the variables 
        self.ufit = self.pls1.fit(self.df[self.reg_vars],
                                  self.df[self.dep_vars])

        ## Return the component and summary of the unconstrained model
        ## To save computation time, run it by default for both models        
        self.component_unconstrained = self.__component(self.ufit,
                                                        self.dep_vars,
                                                        self.reg_vars, self.df)

        self.target_unconstrained = self.__target(self.ufit,
                                                  self.dep_vars,
                                                  self.reg_vars, self.df)

        self.summary_unconstrained = self.__summary(self.ufit, self.dep_vars,
                                                    self.reg_vars, self.df)

        ## Variables selection
        if num_vars == 'all': # Unconstrained model: constrained is identical
            self.top_vars = self.reg_vars # The best variables are the full set
            self.fit = self.ufit
            self.component = self.component_unconstrained
            self.target = self.target_unconstrained
            self.summary = self.summary_unconstrained
            
        elif num_vars > 0: ## Constrained model
            self.num_vars = int(num_vars)
            
            ## Identify the most informative variables from the unconstrained
            self.top_vars = list(self.summary_unconstrained.sort_values(
                by=['vip'], ascending=False).index[:self.num_vars])

            ## Now run the constrained fit on these variables
            self.cfit = self.pls1.fit(self.df[self.top_vars],
                                      self.df[self.dep_vars])

            ## Return the main attributes, consistent names with unconstrained
            self.fit = self.cfit
            
            self.component = self.__component(self.cfit, self.dep_vars,
                                              self.top_vars, self.df)
            
            self.target = self.__target(self.cfit, self.dep_vars,
                                        self.top_vars, self.df)

            self.summary = self.__summary(self.cfit, self.dep_vars,
                                          self.top_vars, self.df)
                      
        else:
            raise ValueError('Number of variables parameter misspecified')

        
    #### Internal class methods (start with "__")
    def __vip(self, model):
        """ 
        Return the variable influence in the projection scores
        Input has to be a sklearn fitted model
        Not available by default on sklearn, so it has to be coded by hand
        """
        ## Get the score, weights and loadings
        t = model.x_scores_
        w = model.x_weights_
        q = model.y_loadings_
        p, h = w.shape

        ## Initialize the VIP
        vips = np.zeros((p,))
        s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
        total_s = np.sum(s)

        for i in range(p):
            weight = [(w[i,j] / np.linalg.norm(w[:,j]))**2 for j in range(h)]
            vips[i] = np.sqrt(p*(s.T @ weight)/total_s)
        return(vips)

    def __summary(self, fit, dep_vars, reg_vars, df):
        """
        Return the summary information about the fit
        """
        
        ## Store the information into a pandas dataframe
        dr = pd.DataFrame(reg_vars, columns=['variable'], index=reg_vars)
        dr['loadings'] = fit.x_loadings_ # Loadings
        dr['vip'] = self.__vip(fit) ## Variable importance in projection
        dr['score'] = fit.score(df[reg_vars],df[dep_vars]) # Score
        
        ## Return the sorted summary frame
        return(dr.sort_values(by=['vip'], ascending=False))
    
    ## Write short ancillary functions to export the results into pandas series
    def __component(self, fit, dep_vars, reg_vars, df):
        """
        Return the first component of the fit
        """
        comp = fit.fit_transform(df[reg_vars], df[dep_vars])[0]
        comp_series = pd.Series(comp.flatten(), index=self.df.index)
        return(comp_series)

    def __target(self, fit, dep_vars, reg_vars, df):
        """
        Return the target of the fit (reduced in case of multiple variables)
        """
        target = fit.fit_transform(df[reg_vars], df[dep_vars])[1]
        target_series = pd.Series(target.flatten(), index=self.df.index)
        return(target_series)

    
    #### Standard class methods (no "__")
    def predict(self, dpred):
        """ 
        Apply the dimension reduction learned on new predictors
        Input:
            - dpred: Pandas frame with the predictors 

        Output:
            - Reduced dataframe using the same loadings as estimated in-sample
 
        """
        
        ## Need to select exactly the predictors which have been estimated
        dp = dpred[self.top_vars].dropna()

        ## Run the projection
        dproj = pd.Series(self.fit.predict(dp).flatten(), index=dp.index)

        ## Scaling pb: prediction and fit don't match in sample (they should) !
        # Create the in-sample prediction
        dproj_in = pd.Series(self.fit.predict(self.df[self.top_vars]).flatten(),
                             index=self.df.index)
        
        # Adjust based on the in-sample projection !
        mean_adj =  dproj_in.mean() - self.component.mean()
        scale_adj = dproj_in.std()/self.component.std()

        # Nicely adjusted ! 
        dproj_mod = (dproj-mean_adj)/scale_adj
        
        return(dproj_mod)

###############################################################################
#%% Function: Chain-index partition
###############################################################################
def chain_index(regvars, df, method, depvars=None):
    """ 
    Create a chain-indexed partition

    Parameters
    ----------
    regvars : list
      variables to aggregate

    df :  pandas dataframe 
      Frame to run the model on 

    method : str
      So for, only "PCA" and "PLS" are available

    Return
    ------
    pandas series of the chained partition
    """

    ## Interpolate df, only for missing values on the right
    ## Ensure that the number of variables available is growing with time
    df = df.interpolate(method='linear', limit_direction='forward').copy()
    
    ## Group variables by data availability
    range_l = [range_index(df, var) for var in regvars]
    u_range_l = list(set(range_l)) # unique values

    sub_series_l = list()

    ## Create a namedtuple to store the information on the subseries
    fields = ['data', 'first_diff', 'num_vars']
    SubSeries = namedtuple('SubSeries', fields)
    
    for rng in u_range_l:
        dfg = df.loc[rng, set(depvars + regvars)].dropna(axis=1).copy()
        if dfg.shape[1] >1:
            if method == 'PCA':
                s_agg = zscore(pca_reduction(regvars, dfg))
            elif method == 'PLS':
                ## Check if the dependent is available
                msg = 'Dependent vars not available for the full time frame'
                assert set(depvars) <= set(dfg.columns), msg
                s_agg = zscore(pls_reduction(depvars, regvars, dfg))
            else:
                raise ValueError('Aggregation method not recognized')
        else: # If only one variable, no need to transform
            s_agg = zscore(dfg.iloc[:,0]) 

        s_agg.index = dfg.index # Insert the index
        s_agg_diff = s_agg - s_agg.shift(-1)  # Rev diff with reverse cumsum
        num_vars = dfg.shape[1] # Store the number of variables
        sdict = {'data':s_agg, 'num_vars': num_vars, 'first_diff': s_agg_diff}
        sub_series_l.append(SubSeries(**sdict)) # Generate through dict cleaner
        
    ## List of objects, sorted by the number of variables    
    series_l = sorted(sub_series_l, key=lambda x: x.num_vars, reverse=True)

    chain = series_l[0].first_diff # The one with the most variables is base

    ## Staple the series of first differences
    for series in series_l[1:]:
        o_diff = series.first_diff
        o_dates = [x for x in o_diff.index if x not in chain.index]
        chain = pd.concat([o_diff[o_dates], chain])

    ## Compute the reverse cumsum (needs to shift it for alignment)
    chain_retropolated = chain.sort_index(ascending=False).cumsum() 
    chain_retropolated = chain_retropolated.sort_index() # Put it back

    chain_retropolated.index = pd.to_datetime(chain_retropolated.index)

    ## Return the chain index
    return(chain_retropolated)


#### Proof of concept (also to try, just do the chained PLS on one variable)

# ## Take a series
# s_agg = pd.Series(np.random.sample(100))

# ## Compute the reverse difference
# s_agg_rev_diff = s_agg.shift(-1) - s_agg 

# ## Compute the cumsum and shift it to align properly
# s_agg_cum_sum = s_agg_rev_diff.cumsum().shift(1) 

# s_agg_cs_aligned = s_agg_cum_sum - s_agg_cum_sum.mean() + s_agg.mean()

# ## Check if the series are the same
# s_agg.plot(label='original', legend=True)
# s_agg_cs_aligned.plot(label='cum_sum', legend=True, secondary_y=True)

# plt.show()


###############################################################################
#%% Partition class : useful wrapper, with plots 
###############################################################################

class Partition(object):
    """ 
    Data reduction using either PLS or PCA (so far) 

    Parameters
    ----------
    vars_list : list of variables to aggregate

    target_vars : list; list of target variables to aggregate data upon

    data : pandas df; data to train the model on

    method : str; default= 'PLS'
        aggregation method. Only 'PLS' or 'PCA' are currently supported

    num_vars : int or str, optional
        number of variables to do the PLS upon. By default: 'all'

    name = str, optional
        Name to use when plotting

    Return
    ------
    A partition object with:

    component : pandas series
        the chained-index PLS or PCA

    summary : pandas dataframe
        Summary frame with information relative to the reduction technic

    plot_partition() : function, plot the partition and VIP

    plot_partition_full() : function, plot the partition, VIP, loadings, target

    """
    __description = "Data Reduction and Partition Class"
    __author = "Romain Lafarguette, IMF, rlafarguette@imf.org"



    #### Class Initializer
    def __init__(self, vars_list, target_vars, data,
                 method='PLS', num_vars='all'):

        #### Attributes
        self.vars_list = vars_list
        self.target_vars = target_vars
        self.data = data        
        self.method = method
        self.num_vars = num_vars
        
        ## Compute the chained-index partition
        self.component = chain_index(self.vars_list,
                                     self.data,
                                     method=self.method,
                                     depvars=self.target_vars)
        
        ## Provide summary information about the reduction
        self.summary = PLS(self.target_vars, self.vars_list,
                              self.data, num_vars='all').summary


    ### Public methods
    def plot_partition(self, title=None, **kwds):
        """ Plot the partition and the VIP """
        
        ## Initialize
        fig = plt.figure(**kwds)

        ## Define the grid
        gs = gridspec.GridSpec(2, 1, wspace=0.1, hspace=0.25) 
        
        ## Populate the grid
        ## Component
        ax_component = fig.add_subplot(gs[0, 0])
        self.component.plot(label=self.method, legend=True, axes=ax_component)
        ax_component.xaxis.set_label_text("")
        ax_component.set_title('Partition')

        ## VIP
        ax_vip = fig.add_subplot(gs[1, 0])
        self.summary['vip'].sort_values(ascending=True).plot.barh(axes=ax_vip)
        ax_vip.set_title('Variables Influence in the Prediction')

        ## Title
        title = title or 'Component and Variables Influence in the Prediction'

        fig.suptitle(title, fontsize=20)    
        
        return(fig)


    def plot_summary_partition(self, title=None, **kwds):
        """ Plot the partition, the VIP, the loadings and the targets """
        
        ## Initialize
        fig = plt.figure(**kwds)
        
        ## Define the grid
        gs = gridspec.GridSpec(2, 2, wspace=0.1, hspace=0.25) 
        
        ## Populate the grid
        
        ## VIP
        ax_vip = fig.add_subplot(gs[0, 0])
        self.summary['vip'].sort_values(ascending=True).plot.barh(axes=ax_vip)
        ax_vip.set_title('Variables Influence in the Prediction')

        ## Loadings
        ax_loadings = fig.add_subplot(gs[1, 0])
        self.summary['loadings'].sort_values().plot.barh(axes=ax_loadings)
        ax_loadings.axvline(x=0, ymin=0, ymax=len(self.summary['loadings']),
                            color='black')
        ax_loadings.set_title('Loadings')

        ## Component
        ax_component = fig.add_subplot(gs[0, 1])
        ax_component.plot(self.component.index, self.component.values)
        ax_component.legend([self.method])
        ax_component.xaxis.set_label_text("")
        ax_component.set_title('Partition')

        ## Target variables (can not use pandas plot due to multiple variables)
        ax_target = fig.add_subplot(gs[1, 1])
        ds = self.data.loc[self.component.index, self.target_vars].copy()
        dsz = (ds - ds.mean())/ds.std(ddof=0) ## Zscore to compare them       
        ax_target.plot(ds.index, dsz)
        ax_target.legend(tuple(self.target_vars))
        ax_target.set_xticks(ax_component.get_xticks()) # Have the same ones
        ax_target.xaxis.set_label_text("")
        ax_target.set_title('Target Variables (Normalized)')
                
        ## Title
        title = title or 'Summary Plots for the PLS Partition'
        fig.suptitle(title, fontsize=20)    

        
        return(fig)




