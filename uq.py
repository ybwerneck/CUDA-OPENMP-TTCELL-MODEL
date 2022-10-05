# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 16:37:32 2022

@author: yanbw
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 02:58:23 2021

@author: yanbw
"""

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
from modelTT import TTCellModel
import math

#Choosing Parameters of Interest

TTCellModel.setParametersOfInterest(["gK1","gKs","gKr","gto","gNa","gCal"])



#Simulation size parameteres


ti=0
ti=5000
tf=5400
dt=0.01
dtS=1
TTCellModel.setSizeParameters(ti, tf, dt, dtS)

Timepoints=TTCellModel.getEvalPoints()
size=np.shape(Timepoints)[0]
Ns = 5000

hypox=cp.Uniform(0,0.1)
hyper=cp.Uniform(0,0.1)
acid=cp.Uniform(0,1)

dist = cp.J(hypox,hyper,acid)
samples = dist.sample(Ns)

print("--Solving")

sols=TTCellModel.run(samples.T,use_gpu=True)

wfs=np.zeros((Ns,size))

qoi={'ADP90':np.zeros(Ns),'ADP50':np.zeros(Ns),'dVmax':np.zeros(Ns),'Vreps':np.zeros(Ns)}

for i in range(Ns):
    for label,v in sols[i].items():
       if(label!='Wf'):
           qoi[label][i]=v     
    for k in range(size):
        wfs[i,k]=sols[i]["Wf"][k]


mean=np.array([np.mean(wfs.T[i]) for i in range(size)])
std=np.array([np.std(wfs.T[i]) for i in range(size)])
minw=np.array([np.min(wfs.T[i]) for i in range(size)])
maxw=np.array([np.max(wfs.T[i]) for i in range(size)])


print("--Ploting Wf")

for waveform in wfs:
        
    plt.plot(Timepoints,waveform, label="mean")
plt.show()

print("--Ploting Wf UQ")

fig, (ax1,ax2) = plt.subplots(2,1,sharex=True)
ax1.plot(Timepoints,mean, label="mean")
ax1.fill_between(Timepoints,mean-std,mean+std, alpha=0.7,label="std")
ax2.plot(Timepoints,minw,label="min")
ax2.plot(Timepoints,maxw,label="max")
ax1.legend(loc='best')
ax2.legend(loc='best')
plt.show()


print("--QOI")
for label,arr in qoi.items():
    print("(",label,") = ",np.mean(arr)," +- ",np.std(arr), "   ranging from ",np.mean(arr),"to",np.max(arr))
    
