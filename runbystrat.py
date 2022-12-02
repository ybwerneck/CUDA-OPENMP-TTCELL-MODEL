# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 20:26:44 2022

@author: yanbw
"""

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
import csv
from openpyxl import Workbook
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
import lhsmdu
import utils
sys.path.append('repo/CUDA-OPENMP-TTCELL-MODEL/')
from modelTT  import TTCellModel as modelA
from ModelB import TTCellModelExt as modelB
from ModelC import TTCellModelChannel as modelC
from scipy.stats.qmc import LatinHypercube
from pyDOE import *
from strategycomparison import surrogatefromSet as strategycomparisonS
from surrogatefromfile import surrogatefromfile as PCEsurrogate
from Models import GPModel,PCEModel,NModel
from generateDatasets import *
from functools import partial
from sklearn import linear_model as lm
from copy import copy

from sklearn.gaussian_process.kernels import *

def getPlots(a,b):
    fig,plotslot=plt.subplots(a,b,figsize=(10, 8),dpi=250)
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
    


 
utils.init() 

ti=20000
tf=20500
dt=0.01

dtS=1

plt.ioff()
resultfolder="./result/"

try:
    os.mkdir(resultfolder)
except:
    print("Result folder occupied, rewriting")
  

models={
        
        
        
        "Model A":["./compA/", modelA],
        "Model B":["./compB/", modelB],
        "Model C":["./compC/", modelC],
        
        
        

        
}
erros={}
timefit={}
timesamp={}
errosM={}
errosR={}
plotModelsSresult=True
for modelName,mdata in models.items():

    
 
  
    
    erros[modelName]={}
    timefit[modelName]={}
    timesamp[modelName]={}
    errosM[modelName]={}
    errosR[modelName]={}

 
    print("Start ", modelName)
    folder,model=mdata
    model.setSizeParameters(ti, tf, dt, dtS)
    dist=model.getDist()
    nPar=model.getNPar()
    kws={"fit_intercept": False}
    surrogates={      
            # "GP  Quadratic":partial(GPModel,dist=dist,kernel=RationalQuadratic(length_scale=2.0, alpha=1.5)), 
            # "GP  WhiteKernel":partial(GPModel,dist=dist,kernel=WhiteKernel()),
           
       
      #   " GP  Default":partial(GPModel,dist=dist,kernel=None),
            #" GP  2":partial(GPModel,dist=dist,kernel=(ConstantKernel(10.0, constant_value_bounds="fixed") * RBF(1.0, length_scale_bounds="fixed"))),
            #  "GP  Quadratic + Dp":partial(GPModel,dist=dist,kernel=(RationalQuadratic(length_scale=2.0, alpha=1.5) * DotProduct())), 
            # " GP  3":partial(GPModel,dist=dist,kernel=(ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(1.0, length_scale_bounds="fixed")+RationalQuadratic(length_scale=0.1, alpha=15))),
       
            #    " GP  4":partial(GPModel,dist=dist,kernel=( RBF(1.0, length_scale_bounds="fixed")*RationalQuadratic(length_scale=1, alpha=15))),
      
            
            "Neural Network regressor":partial(NModel,dist=dist),
             
             #"PCE Lars D4":partial(PCEModel,P=4,regressor=lm.Lars(**kws,eps=0.75),dist=dist),      
             "PCE OLS D4":partial(PCEModel,P=4,regressor=None,dist=dist),
             "PCE OLS D2":partial(PCEModel,P=2,regressor=None,dist=dist),
             
          #   "PCE OLS D5":partial(PCEModel,P=5,regressor=None,dist=dist),
           #  "PCE OLS D6":partial(PCEModel,P=6,regressor=None,dist=dist),
              #ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(1.0, length_scale_bounds="fixed")
              }
    Timepoints=model.getEvalPoints()
    size=np.shape(Timepoints)[0]    
    
    
    TrueModelPeformance=utils.measureModelPerfomance(dist,partial(model.run,use_gpu=False, regen=True,name="out.txt"))
    tmodelpe= TrueModelPeformance
    
    Ns,Nv=int(1e4),int(1e5)
    
    
    
   
    
    
    
    
    qois=["ADP90","ADP50","dVmax","Vrest"]
    
    
    FullSet=utils.readSet(folder, nPar, qois)
    
    
    
    
     

  
    
    s,q,sv,qv=utils.drawSubset(copy(FullSet), 100,dist.sample(100,rule="latin_hypercube",seed=100))
    SAMPREF=utils.storeGmem(sv)
    
    ns=np.append([50*i for i in range(1,20)],[250*i for i in range(4,20)])

    for qoi in qois:
        errosM[modelName][qoi]={}
        errosR[modelName][qoi]={}
        erros[modelName][qoi]={}
        timefit[modelName][qoi]={}
        timesamp[modelName][qoi]={}
        for ml,m in surrogates.items():
            
             erros[modelName][qoi][ml]=np.zeros(np.shape(ns)[0])
             errosR[modelName][qoi][ml]=np.zeros(np.shape(ns)[0])
    
             errosM[modelName][qoi][ml]=np.zeros(np.shape(ns)[0])
             timefit[modelName][qoi][ml]=np.zeros(np.shape(ns)[0])
             timesamp[modelName][qoi][ml]=np.zeros(np.shape(ns)[0])
    
    for i,n in enumerate(ns):
       s,q,sv,qv=utils.drawSubset(copy(FullSet), n,LatinHypercube(d=nPar,seed=10).random(n).T )
    
       print("\n Set of size",n)
       e=strategycomparisonS(s.T,q,sv.T,qv,n,dist,out=False,folder=folder,qoi=qois,models=surrogates,plot=False,sampref=SAMPREF)
       
       
       for qoi in qois:
           for ml,m in surrogates.items():
               erros[modelName][qoi][ml][i],timefit[modelName][qoi][ml][i],timesamp[modelName][qoi][ml][i],errosM[modelName][qoi][ml][i],errosR[modelName][qoi][ml][i]=e[0][ml][qoi],e[1][ml][qoi],e[2][ml][qoi],e[3][ml][qoi],e[4][ml][qoi]
               
            
    
    if(plotModelsSresult):
        
        try:
            os.mkdir(resultfolder+modelName)
        except:
            print("")         
        folder=resultfolder+modelName+"/"      
        ## PLOTS
        
        
        ##CHI ERROR
        
        fig,plots=getPlots(2,2)
        fig.suptitle(modelName+ "  Chi Error")
        plot=0
        
        for qoi in qois:
            
            plot=plots.pop()
            a=[]
            for ml, m in surrogates.items():
                    a.append(plot.plot(ns, erros[modelName][qoi][ml], label=ml))
            
            plot.set_yscale('log')
         
            
            plot.set_title(qoi)           
            
         
                
        tuples_lohand_lolbl = (ax.get_legend_handles_labels() for ax in fig.axes)
        tolohs = zip(*tuples_lohand_lolbl)
        handles, labels = (sum(list_of_lists, []) for list_of_lists in tolohs)
        
        fig.get_axes()[2].set(xlabel="N Samples")
        fig.get_axes()[3].set(xlabel="N Samples")       
        fig.get_axes()[0].set(ylabel="Chi Error")
        fig.get_axes()[2].set(ylabel="Chi Error")
        
        
        plot.legend(handles[0:len(surrogates)], labels[0:len(surrogates)] , loc = 'lower center', bbox_to_anchor = (0, -0.1, 1, 1),
                   bbox_transform = plt.gcf().transFigure)
        
        plt.savefig(folder+"error.png",bbox_inches='tight')
        
        
        ##SPEED UP
        plot
        fig,plots=getPlots(2,2)
        fig.suptitle(modelName+ "  Speed up")
        
        for qoi in qois:
            
            plot=plots.pop()
            a=[]
            for ml, m in surrogates.items():
                    a.append(plot.plot(ns, tmodelpe/timesamp[modelName][qoi][ml], label=ml))
            
            plot.set_yscale('log')
            plot.set_ylim(10,10e5)
        
            plot.set_title(qoi)           
            plot.set(xlabel="N Samples",ylabel="Model Speed Up")
            for ax in fig.get_axes():
          
                ax.label_outer() 
                
        tuples_lohand_lolbl = (ax.get_legend_handles_labels() for ax in fig.axes)
        tolohs = zip(*tuples_lohand_lolbl)
        handles, labels = (sum(list_of_lists, []) for list_of_lists in tolohs)
        
        
        plt.legend(handles[0:len(surrogates)], labels[0:len(surrogates)] , loc = 'lower center', bbox_to_anchor = (0, -0.1, 1, 1),
                    bbox_transform = plt.gcf().transFigure)
        
        plt.savefig(folder+"timemodel.png",bbox_inches='tight')
        
        
        
        fig,plots=getPlots(2,2)
        fig.suptitle(modelName +" Time Fitting")
        plot
        for qoi in qois:
            
            plot=plots.pop()
            a=[]
            for ml, m in surrogates.items():
                    a.append(plot.plot(ns, timefit[modelName][qoi][ml], label=ml))
            
            plot.set_yscale('log')
            plot.set_ylim(1e-4,5e1)
            plot.set_title(qoi)           
            plot.set(xlabel="N Samples",ylabel="Time Fitting")
            for ax in fig.get_axes():
          
                ax.label_outer() 
                
        tuples_lohand_lolbl = (ax.get_legend_handles_labels() for ax in fig.axes)
        tolohs = zip(*tuples_lohand_lolbl)
        handles, labels = (sum(list_of_lists, []) for list_of_lists in tolohs)
        
        
        plt.legend(handles[0:len(surrogates)], labels[0:len(surrogates)] , loc = 'lower center', bbox_to_anchor = (0, -0.1, 1, 1),
                    bbox_transform = plt.gcf().transFigure)
        plt.savefig(folder+"timefit.png",bbox_inches='tight')
        
        
        fig,plots=getPlots(2,2)
        fig.suptitle(modelName +" Max Error %")
        
        
        
        for qoi in qois:
            
            plot=plots.pop()
            plot.set_yscale('log')
            k=0
            cs=plt.rcParams["axes.prop_cycle"].by_key()["color"]
            for ml, m in surrogates.items():
         #           plot.plot(ns, errosR[modelName][qoi][ml], label=ml,color=cs[k])
                    k=k+1
        
            k=0
            for ml, m in surrogates.items():
                  plot.plot(ns, errosM[modelName][qoi][ml],"--", label=ml,color=cs[k])
                  k=k+1
          
            plot.set_title(qoi)           
        
        
        fig.get_axes()[2].set(xlabel="N Samples")
        fig.get_axes()[3].set(xlabel="N Samples")       
        fig.get_axes()[0].set(ylabel="Max Error %")
        fig.get_axes()[2].set(ylabel="Max Error %")
        
        tuples_lohand_lolbl = (ax.get_legend_handles_labels() for ax in fig.axes)
        tolohs = zip(*tuples_lohand_lolbl)
        handles, labels = (sum(list_of_lists, []) for list_of_lists in tolohs)
        
        
        
        lines = plt.gca().get_lines()
        legend1 = plt.legend(handles[0:1*len(surrogates)], labels[0:1*len(surrogates)] , loc = 'lower center', bbox_to_anchor = (0.0, -0.15, 1, 1),
                    bbox_transform = plt.gcf().transFigure)
       
        plt.gca().add_artist(legend1)
        #plt.gca().add_artist(legend2)
        plt.savefig(folder+"relerror.png",bbox_inches='tight')
    
    
    


folder=resultfolder
    
for ml,em in surrogates.items():
    
    try:
        os.mkdir(resultfolder+ml)
    except:
        print("")
    fig,plots=getPlots(2,2)
    fig.suptitle(ml)
    for qoi in qois:
        
        plot=plots.pop()
   
        for modelName,m in models.items():
                plot.plot(ns, erros[modelName][qoi][ml], label=modelName)
        
        plot.set_yscale('log')
        fig.get_axes()[2].set(xlabel="N Samples")
        fig.get_axes()[3].set(xlabel="N Samples")       
        fig.get_axes()[0].set(ylabel="Chi Error")
        fig.get_axes()[2].set(ylabel="Chi Error")
        plot.set_title(qoi)           
   
            
    tuples_lohand_lolbl = (ax.get_legend_handles_labels() for ax in fig.axes)
    tolohs = zip(*tuples_lohand_lolbl)
    handles, labels = (sum(list_of_lists, []) for list_of_lists in tolohs)
    
    
    plt.legend(handles[0:len(models)], labels[0:len(models)] , loc = 'lower center', bbox_to_anchor = (0, -0.1, 1, 1),
                bbox_transform = plt.gcf().transFigure)
    
    plt.savefig(folder+ml+"/chierr.png",bbox_inches='tight')

for ml,em in surrogates.items():
    
    fig,plots=getPlots(2,2)
    fig.suptitle(ml)
    for qoi in qois:
        
        plot=plots.pop()
   
        for modelName,m in models.items():
                plot.plot(ns, errosM[modelName][qoi][ml], label=modelName)
        
        plot.set_yscale('log')
      
        fig.get_axes()[2].set(xlabel="N Samples")
        fig.get_axes()[3].set(xlabel="N Samples")       
        fig.get_axes()[0].set(ylabel="Max Err (%)")
        fig.get_axes()[2].set(ylabel="Max Err (%)")
        plot.set_title(qoi)  
            
    tuples_lohand_lolbl = (ax.get_legend_handles_labels() for ax in fig.axes)
    tolohs = zip(*tuples_lohand_lolbl)
    handles, labels = (sum(list_of_lists, []) for list_of_lists in tolohs)
    
    
    plt.legend(handles[0:len(models)], labels[0:len(models)] , loc = 'lower center', bbox_to_anchor = (0, -0.1, 1, 1),
                bbox_transform = plt.gcf().transFigure)
    
    plt.savefig(folder+ml+"/maxerr.png",bbox_inches='tight')    
    

for ml,em in surrogates.items():
    
    fig,plots=getPlots(2,2)
    fig.suptitle(ml)
    for qoi in qois:
        
        plot=plots.pop()
   
        for modelName,m in models.items():
                plot.plot(ns,  tmodelpe/timesamp[modelName][qoi][ml], label=modelName)
        
        plot.set_yscale('log')
      
        fig.get_axes()[2].set(xlabel="N Samples")
        fig.get_axes()[3].set(xlabel="N Samples")       
        fig.get_axes()[0].set(ylabel="Speed up")
        fig.get_axes()[2].set(ylabel="Speed up")
        plot.set_title(qoi)  
            
    tuples_lohand_lolbl = (ax.get_legend_handles_labels() for ax in fig.axes)
    tolohs = zip(*tuples_lohand_lolbl)
    handles, labels = (sum(list_of_lists, []) for list_of_lists in tolohs)
    
    
    plt.legend(handles[0:len(models)], labels[0:len(models)] , loc = 'lower center', bbox_to_anchor = (0, -0.1, 1, 1),
                bbox_transform = plt.gcf().transFigure)
    
    plt.savefig(folder+ml+"/timemodel.png",bbox_inches='tight')    
    

for ml,em in surrogates.items():
    
    fig,plots=getPlots(2,2)
    fig.suptitle(ml)
    for qoi in qois:
        
        plot=plots.pop()
   
        for modelName,m in models.items():
                plot.plot(ns, timefit[modelName][qoi][ml], label=modelName)
        
        plot.set_yscale('log')
      
        fig.get_axes()[2].set(xlabel="N Samples")
        fig.get_axes()[3].set(xlabel="N Samples")       
        fig.get_axes()[0].set(ylabel="Training Time")
        fig.get_axes()[2].set(ylabel="Training Time")
        plot.set_title(qoi)  
            
    tuples_lohand_lolbl = (ax.get_legend_handles_labels() for ax in fig.axes)
    tolohs = zip(*tuples_lohand_lolbl)
    handles, labels = (sum(list_of_lists, []) for list_of_lists in tolohs)
    
    
    plt.legend(handles[0:len(models)], labels[0:len(models)] , loc = 'lower center', bbox_to_anchor = (0,-0.1, 1, 1),
                bbox_transform = plt.gcf().transFigure)
    
    plt.savefig(folder+ml+"/timefit.png",bbox_inches='tight')    

porcent=[0.1,0.05,0.01]


for model,M in models.items():
    wb = Workbook()
    ws=wb.active
    print("Model",model)
    
    for qoi in qois:        
       
        ws.title = qoi

        
        
        print("QOI",qoi)
        
        for pc in porcent:
        
            ws.append(["For minimum accuracy of "+str(pc)])
            ws.append(['Strategy' ,'Training Samples' ,'Speed up' ,'Training time' ,'Chi Err'] )
            print("Cheapest emulator with ", pc," security")
           
           
            for ml,em in surrogates.items():
                print(" ")
                print("Min ",ml)
                try:
                    ind=[i for i in range(0,np.shape(ns)[0]) if errosM[model][qoi][ml][i]<pc]
                    print(ind)
                    ind=ind[0]
                    print("Training Samples ",ns[ind])
                    print("Speed up ",tmodelpe/timesamp[model][qoi][ml][ind])
                    print("Training Time", timefit[model][qoi][ml][ind])
                    print ("Chi err",erros[model][qoi][ml][ind])
                    ws.append([ml,ns[ind],tmodelpe/timesamp[model][qoi][ml][ind],timefit[model][qoi][ml][ind],erros[model][qoi][ml][ind]] )
                    
                except:
                    print("No Results")
                    ws.append([ml,"None"])
                print(" ")
                print(" ")
           
        ws = wb.create_sheet()            
    wb.save(folder+model+"/numerical.xlsx")
                   
