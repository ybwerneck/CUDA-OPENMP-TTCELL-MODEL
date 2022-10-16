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
import utils
sys.path.append('repo/CUDA-OPENMP-TTCELL-MODEL/')
from modelTT  import TTCellModel as modelA
from ModelB import TTCellModelExt as modelB
from ModelC import TTCellModelChannel as modelC

from strategycomparison import surrogatefromSet as strategycomparisonS
from surrogatefromfile import surrogatefromfile as PCEsurrogate
from Models import GPModel,PCEModel,NModel
from generateDatasets import *
from functools import partial
from sklearn import linear_model as lm
from copy import copy



def getPlots(a,b):
    fig,plotslot=plt.subplots(a,b,figsize=(8, 6),dpi=250)
    plotsaux=[]
    plots=[]
            
    try:
                   for row in plotslot:
                       for frame in row:
                           plotsaux.append(frame)
    except:
                   try :
                       for frame in plotslot:
                           plotsaux.append(frame)
                   except:
                           plotsaux.append(plotslot)
    
               
    for i in range(0,len(plotsaux)):
                plots.append(plotsaux.pop())
    
    return fig,plots
    




model=modelB

ti=00
tf=400
dt=0.01
dtS=1
model.setSizeParameters(ti, tf, dt, dtS)
dist=model.getDist()
nPar=model.getNPar()

Timepoints=model.getEvalPoints()
size=np.shape(Timepoints)[0]    


TrueModelPeformance=3
#utils.measureModelPerfomance(dist,partial(model.run,use_gpu=False, regen=True,name="out.txt"))

Ns,Nv=10000,20000  



folder="./compA/"
try:
    os.mkdir(folder)
 
except:
    print("folder ocuppied, rewriting")

#generateDataset(dist,folder,Ns,Nv,model,nx=False,ny=False,out=True,remove_closest=False,gpu=True)    

qois={"ADP50","ADP90","Vrest","dVmax"}


FullSet=utils.readSet(folder, Ns, nPar, qois)



utils.init()   
kws={"fit_intercept": False}
surrogates={
        
       "GP":partial(GPModel,dist=dist),
        "Neural Network regressor":partial(NModel,dist=dist),
        "PCE Lars D4":partial(PCEModel,P=4,regressor=lm.Lars(**kws,eps=0.75),dist=dist),      
       
      
        "PCE OLS D4":partial(PCEModel,P=4,regressor=None,dist=dist),
        "PCE OLS D2":partial(PCEModel,P=2,regressor=None,dist=dist),
        
        }






ns=np.append([20*i for i in range(1,2)],[250*i for i in range(1,2)])
erros={}
timefit={}
timesamp={}
for qoi in qois:
    erros[qoi]={}
    timefit[qoi]={}
    timesamp[qoi]={}
    for ml,m in surrogates.items():
        
         erros[qoi][ml]=np.zeros(np.shape(ns)[0])
         timefit[qoi][ml]=np.zeros(np.shape(ns)[0])
         timesamp[qoi][ml]=np.zeros(np.shape(ns)[0])

for i,n in enumerate(ns):
   s,q,sv,qv=utils.drawSubset(copy(FullSet), n,dist.sample(n,rule="latin_hypercube"))      
   print("\n Set of size",n)
   e=strategycomparisonS(s.T,q,sv.T,qv,n,dist,out=False,folder=folder,qoi=qois,models=surrogates,plot=False)
   
   
   for qoi in qois:
       for ml,m in surrogates.items():
           erros[qoi][ml][i],timefit[qoi][ml][i],timesamp[qoi][ml][i]=e[0][ml][qoi],e[1][ml][qoi],e[2][ml][qoi]
           
        

        
       
modelName="Modelo A"

fig,plots=getPlots(2,2)
fig.suptitle(modelName+ " Model Error")
plot=0

for qoi in qois:
    
    plot=plots.pop()
    a=[]
    for ml, m in surrogates.items():
            a.append(plot.plot(ns, erros[qoi][ml], label=ml))
    
    plot.set_yscale('log')
    plot.set_ylim(1e-5,1e1)
    plot.set_title(qoi)           
    plot.set(xlabel="N Samples",ylabel="Error")
    for ax in fig.get_axes():
  
        ax.label_outer() 
        
tuples_lohand_lolbl = (ax.get_legend_handles_labels() for ax in fig.axes)
tolohs = zip(*tuples_lohand_lolbl)
handles, labels = (sum(list_of_lists, []) for list_of_lists in tolohs)


plot.legend(handles[0:len(surrogates)], labels[0:len(surrogates)] , loc = 'lower center', bbox_to_anchor = (0, -0.2, 1, 1),
           bbox_transform = plt.gcf().transFigure)

plt.savefig(folder+"error.png",bbox_inches='tight')

plot
fig,plots=getPlots(2,2)
fig.suptitle(modelName+ " Model Time")

for qoi in qois:
    
    plot=plots.pop()
    a=[]
    for ml, m in surrogates.items():
            a.append(plot.plot(ns, timesamp[qoi][ml], label=ml))
    
    plot.set_yscale('log')
    plot.set_ylim(0,1e1)
    plot.set_title(qoi)           
    plot.set(xlabel="N Samples",ylabel="Model Time")
    for ax in fig.get_axes():
  
        ax.label_outer() 
        
tuples_lohand_lolbl = (ax.get_legend_handles_labels() for ax in fig.axes)
tolohs = zip(*tuples_lohand_lolbl)
handles, labels = (sum(list_of_lists, []) for list_of_lists in tolohs)


plt.legend(handles[0:len(surrogates)], labels[0:len(surrogates)] , loc = 'lower center', bbox_to_anchor = (0, -0.2, 1, 1),
           bbox_transform = plt.gcf().transFigure)

plt.savefig(folder+"timemodel.png",bbox_inches='tight')



fig,plots=getPlots(2,2)
fig.suptitle(modelName "+ Time Fitting")
plot
for qoi in qois:
    
    plot=plots.pop()
    a=[]
    for ml, m in surrogates.items():
            a.append(plot.plot(ns, timefit[qoi][ml], label=ml))
    
    plot.set_yscale('log')
    plot.set_ylim(0,1e1)
    plot.set_title(qoi)           
    plot.set(xlabel="N Samples",ylabel="Time Fitting")
    for ax in fig.get_axes():
  
        ax.label_outer() 
        
tuples_lohand_lolbl = (ax.get_legend_handles_labels() for ax in fig.axes)
tolohs = zip(*tuples_lohand_lolbl)
handles, labels = (sum(list_of_lists, []) for list_of_lists in tolohs)


plt.legend(handles[0:len(surrogates)], labels[0:len(surrogates)] , loc = 'lower center', bbox_to_anchor = (0, -0.2, 1, 1),
           bbox_transform = plt.gcf().transFigure)
plt.savefig(folder+"timefit.png",bbox_inches='tight')
