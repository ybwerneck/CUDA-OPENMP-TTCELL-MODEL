# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 01:28:15 2022

@author: yanbw
"""
import numpy as np
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.integrate import odeint
import lmfit
from lmfit.lineshapes import gaussian, lorentzian
import chaospy as cp
from scipy.integrate import odeint
from lmfit import minimize, Parameters, Parameter, report_fit
from SALib.sample import saltelli
from SALib.analyze import sobol
import timeit
import os

sys.path.append('repo/CUDA-OPENMP-TTCELL-MODEL/')
from modelTT  import TTCellModel
from strategycomparison import surrogatefromfile as strategycomparisonS

from generateDatasets import *

ti=10000
tf=10300
dt=0.01
dtS=1
TTCellModel.setSizeParameters(ti, tf, dt, dtS)


Timepoints=TTCellModel.getEvalPoints()
size=np.shape(Timepoints)[0]    

Ns,Nv=100000,200000  
 
hypox=cp.Uniform(0,1.25)
hyper=cp.Uniform(0,1.25)
acid=cp.Uniform(0,1.25) 
dist = cp.J(hypox,hyper,acid)

folder="./sobol/"
try:
    os.mkdir(folder)
 
except:
    print("folder ocuppied, rewriting")
       
#generateDataset(dist,folder,Ns,Nv,nx=False,ny=False,out=True,remove_closest=False,gpu=True)    
qois={"ADP50","tdV","Vrest","dVmax"}

models = {
 
     
     "OLS CP": None,

     
 
   
 }
sobol={}
strategycomparisonS(folder,Ns,out=False,qoi=qois,sobolR=sobol,models=models)
#PCEsurrogate(folder,Ns,out=False,qoi=qois,sobolR=sobol,models=models)

# plts=[]
# for model,aux in models.items():
#     fig, pts = plt.subplots(2,2)
#     for row in pts:
#         for f in row:
#             plts.append(f)
#     for qoi in qois:
#         print(sobol[qoi][model])
#         ax=plts.pop()  
        

#         lab = ['Hypox', 'Hyper', 'Acid']
#         counts = sobol[qoi][model][0].flatten(),sobol[qoi][model][1].flatten()
#         bar_colors = ['tab:red', 'tab:blue', 'tab:green']
#         w=0.3
#         x=np.arange(len(lab))
#         ax.bar(x-w/2, counts[0],w,color=bar_colors,alpha=0.9)
#         ax.bar(x+w/2, counts[1],w, color=bar_colors,alpha=0.8)
#         ax.set_ylabel('Sobol Index First order and Total')
        
#         ax.set_title('SA '+qoi+" "+model)
#         ax.set_xticks(x, lab)
#     for ax in fig.get_axes():
#         ax.label_outer()     
#     fig.tight_layout()
#     fig.savefig(folder+"results/"+model+qoi+".png")
#     plt.show()
   


print("./",folder,"  is ready with results!")