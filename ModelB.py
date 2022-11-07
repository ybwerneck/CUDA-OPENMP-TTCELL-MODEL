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

class TTCellModelExt(TTCellModel):
   

   
    @staticmethod
    def cofs(ps):
        print(ps)
        params=[
               (1-0.25*ps[0])*TTCellModel.g_Na_default,
               
               (1-0.25*ps[1])*TTCellModel.g_CaL_default,
                
               TTCellModel.K_i_default - 13.3*ps[2],
               
               TTCellModel.K_o_default + 4.6*ps[3],
               
               TTCellModel.atp_default - 3 * ps[4], ##Atp
               
               TTCellModel.g_K1_defaults , 
               TTCellModel.g_Kr_defaults , 
               TTCellModel.g_Ks_defaults,                
               TTCellModel.g_to_defaults  ,
               TTCellModel.g_bca_defaults ,
               
               
                
         ]
              
            
          
     
        return np.array(params)
    
    @staticmethod
    def getDist(low=0,high=1):

        gna=cp.Uniform(low,high)
        gcal=cp.Uniform(low,high)    
        ki=cp.Uniform(low,high) 
        ko=cp.Uniform(low,high)    
        atp=cp.Uniform(low,high) 
        dist = cp.J(gna,gcal,ki,ko,atp)
        return dist


    @staticmethod
    def run(P="",use_gpu=False, regen=True,name="out.txt"):  

         return TTCellModel.run(P,use_gpu=use_gpu,regen=regen,name=name,cofsF=TTCellModelExt.cofs)


    @staticmethod
    def getNPar():
        return 5
         