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

from sklearn.gaussian_process.kernels import *

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
                           plotsaux.append+++(plotslot)
    
               
    for i in range(0,len(plotsaux)):
                plots.append(plotsaux.pop())
    
    return fig,plots
    



model=modelA
folder="./compA/"
modelName="Modelo A"

ti=20000
tf=20500
dt=0.01
dtS=1
model.setSizeParameters(ti, tf, dt, dtS)
dist=model.getDist()
nPar=model.getNPar()

Timepoints=model.getEvalPoints()
size=np.shape(Timepoints)[0]    


TrueModelPeformance=3
tmodelpe= utils.measureModelPerfomance(dist,partial(model.run,use_gpu=False, regen=True,name="out.txt"))

Ns,Nv=5000,int(1e4)



try:
    os.mkdir(folder)
 
except:
    print("folder ocuppied, rewriting")

generateDataset(dist,folder,Ns,Nv,model,nx=False,ny=False,out=True,remove_closest=True,gpu=True)    

qois=["ADP90","ADP50","dVmax","Vrest"]


FullSet=utils.readSet(folder, Ns, nPar, qois)





utils.init()   
kws={"fit_intercept": False}
surrogates={      
      # "GP  Quadratic":partial(GPModel,dist=dist,kernel=RationalQuadratic(length_scale=2.0, alpha=1.5)), 
      # "GP  WhiteKernel":partial(GPModel,dist=dist,kernel=WhiteKernel()),
     
      " GP  Default":partial(GPModel,dist=dist,kernel=None),
      #" GP  2":partial(GPModel,dist=dist,kernel=(ConstantKernel(10.0, constant_value_bounds="fixed") * RBF(1.0, length_scale_bounds="fixed"))),
     #  "GP  Quadratic + Dp":partial(GPModel,dist=dist,kernel=(RationalQuadratic(length_scale=2.0, alpha=1.5) * DotProduct())), 
    # " GP  3":partial(GPModel,dist=dist,kernel=(ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(1.0, length_scale_bounds="fixed")+RationalQuadratic(length_scale=0.1, alpha=15))),
 
   #    " GP  4":partial(GPModel,dist=dist,kernel=( RBF(1.0, length_scale_bounds="fixed")*RationalQuadratic(length_scale=1, alpha=15))),

      
"Neural Network regressor":partial(NModel,dist=dist),
       
       #"PCE Lars D4":partial(PCEModel,P=4,regressor=lm.Lars(**kws,eps=0.75),dist=dist),      
       
       "PCE OLS D4":partial(PCEModel,P=4,regressor=None,dist=dist),
       "PCE OLS D2":partial(PCEModel,P=2,regressor=None,dist=dist),
        #ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(1.0, length_scale_bounds="fixed")
        }





ns=np.append([50*i for i in range(1,4)],[250*i for i in range(1,15)])
erros={}
timefit={}
timesamp={}
errosM={}
errosR={}
for qoi in qois:
    errosM[qoi]={}
    errosR[qoi]={}
    erros[qoi]={}
    timefit[qoi]={}
    timesamp[qoi]={}
    for ml,m in surrogates.items():
        
         erros[qoi][ml]=np.zeros(np.shape(ns)[0])
         errosR[qoi][ml]=np.zeros(np.shape(ns)[0])

         errosM[qoi][ml]=np.zeros(np.shape(ns)[0])
         timefit[qoi][ml]=np.zeros(np.shape(ns)[0])
         timesamp[qoi][ml]=np.zeros(np.shape(ns)[0])

for i,n in enumerate(ns):
   s,q,sv,qv=utils.drawSubset(copy(FullSet), n,dist.sample(n,rule="latin_hypercube"))

   print("\n Set of size",n)
   e=strategycomparisonS(s.T,q,sv.T,qv,n,dist,out=False,folder=folder,qoi=qois,models=surrogates,plot=False)
   
   
   for qoi in qois:
       for ml,m in surrogates.items():
           erros[qoi][ml][i],timefit[qoi][ml][i],timesamp[qoi][ml][i],errosM[qoi][ml][i],errosR[qoi][ml][i]=e[0][ml][qoi],e[1][ml][qoi],e[2][ml][qoi],e[3][ml][qoi],e[4][ml][qoi]
           
        

        
       


fig,plots=getPlots(2,2)
fig.suptitle(modelName+ " Model Chi Error")
plot=0

for qoi in qois:
    
    plot=plots.pop()
    a=[]
    for ml, m in surrogates.items():
            a.append(plot.plot(ns, erros[qoi][ml], label=ml))
    
    plot.set_yscale('log')
    plot.set_ylim(1e-5,1)
    plot.set_title(qoi)           
    
 
        
tuples_lohand_lolbl = (ax.get_legend_handles_labels() for ax in fig.axes)
tolohs = zip(*tuples_lohand_lolbl)
handles, labels = (sum(list_of_lists, []) for list_of_lists in tolohs)

fig.get_axes()[2].set(xlabel="N Samples")
fig.get_axes()[3].set(xlabel="N Samples")       
fig.get_axes()[0].set(ylabel="N Samples")
fig.get_axes()[2].set(ylabel="N Samples")


plot.legend(handles[0:len(surrogates)], labels[0:len(surrogates)] , loc = 'lower center', bbox_to_anchor = (0, -0.2, 1, 1),
           bbox_transform = plt.gcf().transFigure)

plt.savefig(folder+"error.png",bbox_inches='tight')

plot
fig,plots=getPlots(2,2)
fig.suptitle(modelName+ " Model Speed up")

for qoi in qois:
    
    plot=plots.pop()
    a=[]
    for ml, m in surrogates.items():
            a.append(plot.plot(ns, tmodelpe/timesamp[qoi][ml], label=ml))
    
    plot.set_yscale('log')
    plot.set_ylim(10,10e5)

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
fig.suptitle(modelName +" Time Fitting")
plot
for qoi in qois:
    
    plot=plots.pop()
    a=[]
    for ml, m in surrogates.items():
            a.append(plot.plot(ns, timefit[qoi][ml], label=ml))
    
    plot.set_yscale('log')
    plot.set_ylim(1e-4,5e1)
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


fig,plots=getPlots(2,2)
fig.suptitle(modelName +" Relative Error")



for qoi in qois:
    
    plot=plots.pop()
    plot.set_yscale('log')
    k=0
    cs=plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for ml, m in surrogates.items():
       #     plot.plot(ns, errosR[qoi][ml], label=ml,color=cs[k])
            k=k+1

    k=0
    for ml, m in surrogates.items():
          plot.plot(ns, errosM[qoi][ml],"--", label=ml,color=cs[k])
          k=k+1
  
    plot.set_title(qoi)           

plot.plot([],[],"b--",label="Max")


fig.get_axes()[2].set(xlabel="N Samples")
fig.get_axes()[3].set(xlabel="N Samples")       
fig.get_axes()[0].set(ylabel="N Samples")
fig.get_axes()[2].set(ylabel="N Samples")

tuples_lohand_lolbl = (ax.get_legend_handles_labels() for ax in fig.axes)
tolohs = zip(*tuples_lohand_lolbl)
handles, labels = (sum(list_of_lists, []) for list_of_lists in tolohs)



lines = plt.gca().get_lines()
legend1 = plt.legend(handles[0:1*len(surrogates)], labels[0:1*len(surrogates)] , loc = 'lower center', bbox_to_anchor = (-0.2, -0.15, 1, 1),
            bbox_transform = plt.gcf().transFigure)
legend2 =plt.legend(handles[-2:], labels[-2:] , loc = 'lower center', bbox_to_anchor = (+0.2, -0.15, 1, 1),
            bbox_transform = plt.gcf().transFigure)
plt.gca().add_artist(legend1)
plt.gca().add_artist(legend2)
plt.savefig(folder+"relerror.png",bbox_inches='tight')
