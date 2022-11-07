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
        print(ps)
        params=[
               (1-0.25*ps[0])*TTCellModel.g_Na_default,               
               (1-0.25*ps[1])*TTCellModel.g_CaL_default,                
               TTCellModel.K_i_default,               
               TTCellModel.K_o_default ,               
               TTCellModel.atp_default, ##Atp
               TTCellModel.g_K1_defaults *(1 -  ps[2] * 0.4), 
               TTCellModel.g_Kr_defaults *(1 -  ps[3] * 0.7), 
               TTCellModel.g_Ks_defaults *(1  -  ps[4] * 0.8) ,                
               TTCellModel.g_to_defaults *(1  -  ps[5] )  ,
               TTCellModel.g_bca_defaults *(1  +  ps[6]*(0.33) )  ,
               
               
                
         ]
              
            
          
     
        return np.array(params)
    
    @staticmethod
    def getDist(low=0,high=1):
        
        gna=cp.Uniform(low,high)
        gcal=cp.Uniform(low,high)    
        ki=cp.Uniform(low,high) 
        ko=cp.Uniform(low,high)    
        atp=cp.Uniform(low,high) 
        
        gk1=cp.Uniform(low,high)    
        gkr=cp.Uniform(low,high) 
        gks=cp.Uniform(low,high)    
        gto=cp.Uniform(low,high) 
        gbca=cp.Uniform(low,high)
        dist = cp.J(gna,gcal,gk1,gkr,gks,gto,gbca)
        return dist

    @staticmethod
    def run(P="",use_gpu=False, regen=True,name="out.txt"):  

         return TTCellModel.run(P,use_gpu=use_gpu,regen=regen,name=name,cofsF=TTCellModelChannel.cofs)


    @staticmethod
    def getNPar():
        return 7
         