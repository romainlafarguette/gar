# -*- coding: utf-8 -*-
"""
Created on Mon May 13 09:06:10 2019

@author: CWang2
"""

'''
Import modules
'''
import pandas as pd # To import excel files into dataframe
import numpy as np # To import math functions
from tskew import tskew_pdf # To import specific functions 
from tskew import tskew_cdf
from tskew import tskew_ppf
from tskewfit import tskew_fit
import statsmodels as sm # Imports statistical models
import statsmodels.formula.api as smf # To do regression analysis
import matplotlib.pyplot as plt # To plot 
from mpl_toolkits.mplot3d import Axes3D # 3D plot
from matplotlib import cm # Colormap
from matplotlib.ticker import FormatStrFormatter # To format the figures


'''
This the code for PIT
need Data frame contains the target GDP growth
The fitted t-skew fit tsfit

'''


# Need df contains target growth rate
# fitted t-skew fit tsfit
cname='Isral'
pits=[]
if not np.isnan(df.loc[ftime,'target']): # For each realized value that is different from NaN
                pits.append(tskew_cdf(df.loc[ftime,'target'], df=tsfit['df'], loc=tsfit['loc'], scale=tsfit['scale'], skew=tsfit['skew'])) # Append the probability of the realized value read off of the quarter-specific CDF of the skew-t  

pits_cdf=[]
npits=len(pits)
for r in np.arange(0,1,0.01):
    pits_cdf.append(len([x for x in pits if x<=r])/npits) # Calculate how many realized values are below any given probability
      

figpit, axpit= plt.subplots(1, 1, figsize=(7,7))
axpit.plot(list(np.arange(0,1,0.01)),pits_cdf,'r-',label='Realized')
axpit.plot(list(np.arange(0,1,0.01)),list(np.arange(0,1,0.01)),'b-',label='U~(0,1)')
axpit.plot(list(np.arange(0,1,0.01)),[e+1.34*npits**(-0.5) for e in np.arange(0,1,0.01)],'b-',label='5 percent critical values',linestyle='dashed')
axpit.plot(list(np.arange(0,1,0.01)),[e-1.34*npits**(-0.5) for e in np.arange(0,1,0.01)],'b-',linestyle='dashed')
axpit.plot(list(np.arange(0,1,0.01)),[e+1.21*npits**(-0.5) for e in np.arange(0,1,0.01)],'c-',label='10 percent critical values',linestyle='dashed')
axpit.plot(list(np.arange(0,1,0.01)),[e-1.21*npits**(-0.5) for e in np.arange(0,1,0.01)],'c-',linestyle='dashed')
axpit.plot(list(np.arange(0,1,0.01)),[e+1.61*npits**(-0.5) for e in np.arange(0,1,0.01)],'m-',label='1 percent critical values',linestyle='dashed')
axpit.plot(list(np.arange(0,1,0.01)),[e-1.61*npits**(-0.5) for e in np.arange(0,1,0.01)],'m-',linestyle='dashed')
plt.xlim(0,1)
plt.ylim(0,1)
plt.legend(loc=4)
plt.title('PIT test for '+cname+' horizon '+str(hz))
figpit.savefig('./Figures/PITtest_'+cname+'_horizon_'+str(hz)+'.png')