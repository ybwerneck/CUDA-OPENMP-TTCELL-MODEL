# -*- coding: utf-8 -*-
"""
Created on Sat Oct 15 22:22:48 2022

@author: yanbw
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 16:43:30 2022

@author: yanbw
"""

###Base e modelo 
import subprocess 
import sys
import numpy as np
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mpld3
from scipy.integrate import odeint
import lmfit
from lmfit.lineshapes import gaussian, lorentzian
import chaospy as cp
from scipy.integrate import odeint
from lmfit import minimize, Parameters, Parameter, report_fit
from SALib.sample import saltelli
from SALib.analyze import sobol
import timeit
import re
import collections
import os
import six
from modelTT import TTCellModel

class TTCellModelChannel(TTCellModel):
   

   
    @staticmethod
    def cofs(ps):

        params=[
               (1-0.25*ps[1])*TTCellModel.g_Na_default,
               
               (1-0.25*ps[2])*TTCellModel.g_CaL_default,
                
               TTCellModel.K_i_default - 13.3*ps[3],
               
               TTCellModel.K_o_default + 4.6*ps[4],
               
               TTCellModel.atp_default - 3 * ps[0], ##Atp
               
               TTCellModel.g_K1_defaults *(1 -  ps[5] * 0.4), 
               TTCellModel.g_Kr_defaults *(1 -  ps[6] * 0.7), 
               TTCellModel.g_Ks_defaults *(1  -  ps[6] * 0.8) ,                
               TTCellModel.g_to_defaults (1  -  ps[6] )  ,
               
                
         ]
              
            
          
     
        return np.array(params)
    
    @staticmethod
    def getDist():

        gna=cp.Uniform(0,1.25)
        gcal=cp.Uniform(0,1.25)    
        ki=cp.Uniform(0,1.25) 
        ko=cp.Uniform(0,1.25)    
        atp=cp.Uniform(0,1.25) 
        
        gk1=cp.Uniform(0,1.25)    
        gkr=cp.Uniform(0,1.25) 
        gks=cp.Uniform(0,1.25)    
        gto=cp.Uniform(0,1.25) 
        dist = cp.J(gna,gcal,ki,ko,atp,gk1,gkr,gks,gto)
        return dist


    @staticmethod
    def getNPar():
        return 9
         