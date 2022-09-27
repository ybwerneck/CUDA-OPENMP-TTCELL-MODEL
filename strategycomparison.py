from sklearn.datasets import make_friedman2
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel


# -*- coding: utf-8 -*-
"""
Created on Mon May  9 12:40:26 2022

@author: yanbw
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 02:58:23 2021

@author: yanbw
"""
import os


import numpy as np
import matplotlib.pyplot as plt
import chaospy as cp
import timeit
from sklearn import linear_model as lm
import csv
from utils import runModelParallel as runModel
import utils
from functools import partial

import random
#MODEL WRAPPER -> PCE MODEL
def ModelPCE(exp):
    def Model(sample): 
        return cp.call(exp,sample)
    return Model


def surrogatefromfile(folder,Ns,qoi={"ADP50","ADP90","Vrest","dVmax","tdV"},out=False,sobolR=None,models=None,vali=True):

    utils.init()
    print("Start Surrogate from File")
    #Load validation files

    ##Reference Value for Sobol Indexes Error calculation
    
    pmin,pmax=2,4
    
    try:
        os.mkdir(folder+"results/")
    except:
        print("Updating results")
        
        
        
        
        
    #Parameters of interest X




    hypoxd=cp.Uniform(0,1.25)
    hyperd=cp.Uniform(0,1.25)
    acid=cp.Uniform(0,1.25)
    
    dist = cp.J(hypoxd,hyperd,acid)
    
    
    ##Load Result File 
    f = open(folder+'results/numeric.csv', 'a',newline='')
    updt=os.path.exists('numeric.csv')
    
    
    # create the csv writer
    writer = csv.writer(f)
    row=['QOI',	'Method', 'Degree','Val. error',' LOOERROR','Ns','Timeselected','Timemax','Timeselected G','TimemaxG','Time T']
    writer.writerow(row)
    
  
    #Load datasets
    
    #Training Samples
    nPar=6
    samples=np.empty((Ns,nPar))
    X=utils.readF(folder+"X.csv")
    samplesVal=np.zeros((len(X),6))
    for i,sample in enumerate(X):       ##must be matrix not list
        for k,y in enumerate(sample):
            samples[i][k]=y
            
    Y={}
    for qlabel in qoi:
        Y[qlabel]=utils.readF(folder+qlabel+".csv")
    
    
    
    #Validation Samples
    
    Xv=utils.readF(folder+"validation/"+"X.csv")
    samplesVal=np.zeros((len(Xv),6))
    for i,sample in enumerate(X):       ##must be matrix not list
        for k,y in enumerate(sample):
            samplesVal[i][k]=y
    
    Yval={}
    for qlabel in qoi:
        Yval[qlabel]=utils.readF(folder+"validation/"+qlabel+".csv")
    
    
    
    wfs=utils.readWF(folder+"validation/wfs.csv")
    
  
    
    #Sample the parameter distribution
         
          
       
       
       
    print('\n',"QOI: ", qlabel,'\n')      
   ##Adpative algorithm chooses best fit in deegree range
    timeL=0
    a,b=2,2   
    fig,plotslot=plt.subplots(a,b)
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

    
    def GPModel(x,y,dist):
        
        def model (gpr,X):
            return gpr.predict(np.array(X),return_std=False)
        gpr=GaussianProcessRegressor(
        random_state=0)
        gpr.fit(X, y)
        return partial(model,gpr)
        
    def PCEModel(x,y,dist,P=2,regressor=None) :
        def model(pce,X):
            return cp.call(pce,X)
        poly_exp = cp.generate_expansion(P, dist,rule="three_terms_recurrence")
        fp = cp.fit_regression (poly_exp,x.T,y,model=regressor)  
        return partial(model,fp)
        
    models={
        
       # "GP":GPModel,
        "PCE":partial(PCEModel,P=2,regressor=None)
        
        }
        
    
    for mlabel,model in models.items():
        for i in range(0,len(plotsaux)):
            plots.append(plotsaux.pop())
        pltidx=0
        fig.suptitle(mlabel)   
        for label in qoi:
            y = np.array(Y[label]).flatten()
            yv=np.array(Yval[label]).flatten()
            
          
            predictor=model(np.array(X),y,dist)
       
            ypred=runModel(np.array(Xv),predictor)     
            ypred=np.array([ypred[i] for i in ypred])
            
            
            
            p=plots.pop()   
            p.scatter(yv,ypred)
            p.set_title(label)           
            p.plot(yv,yv,"black",linewidth=2)
            p.set(xlabel="Y_true",ylabel="Y_pred")
            
            
import runSet