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
   
    K_o_default=5.40e+00
    g_CaL_default=1.750e-04
    g_Na_default=1.48380e+01
    K_i_default=138.3
    atp_default=2.6
   
    @staticmethod
    def cofs(ps):

        params=[
             (1-0.25*ps[0])*TTCellModel.g_Na_default, #gna
            (1-0.25*ps[1])*TTCellModel.g_CaL_default, #gcal
            
             138.3 - 13.3*ps[2], #KI
            5.4 + 4.6*ps[3],#KO
            5.6 - 3 * ps[4]#ATP      
            
            ]
     
        return np.array(params)
    
    @staticmethod
    def getDist():

        gna=cp.Uniform(0,1.25)
        gcal=cp.Uniform(0,1.25)    
        ki=cp.Uniform(0,1.25) 
        ko=cp.Uniform(0,1.25)    
        atp=cp.Uniform(0,1.25) 
        dist = cp.J(gna,gcal,ki,ko,atp)
        return dist


    @staticmethod
    def getNPar():
        return 5
         