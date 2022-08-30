# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 18:01:49 2022

@author: yanbw
"""

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

import random

import numpy as np
import matplotlib.pyplot as plt
import chaospy as cp
import timeit
from sklearn import linear_model as lm
import csv
from utils import runModelParallel as runModel
import utils
#MODEL WRAPPER -> PCE MODEL
def ModelPCE(exp):
    def Model(sample): 
        return cp.call(exp,sample)
    return Model


def LOOEXP(folder,Ns,qoi={"ADP50"}):



#Load validation files

    ##Reference Value for Sobol Indexes Error calculation
    utils.init()
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
    
    
    #Training Samples
    nPar=6
    samples=np.empty((Ns,3))
    X=utils.readF(folder+"X.csv")
    samplesVal=np.zeros((len(X),3))
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
    
    
    
    
    

    
    
    alpha=1
    eps=0.75
    kws = {"fit_intercept": False,"normalize":False}
    models = {
    
        
        "OLS CP": None,    
    
      
    }
    
    
    
    
    ##
    pltxs=2
    pltys=0
    
    while(pltys*pltxs<len(models)):
        pltys=pltys+1
        
    P=2
    
    for qlabel,dataset in Y.items():
        print('\n',"QOI: ", qlabel,'\n')      
      
          
        for label, model in models.items():   
            print('\n--------------',"\n")
            print("Beggining ", label)
            
            
        
            start = timeit.default_timer()
            poly_exp = cp.generate_expansion(P, dist,rule="three_terms_recurrence")
            fp = cp.fit_regression (poly_exp,samples.T,dataset,model=model)  
            stop = timeit.default_timer()
            time=stop-start
            print('Time to generate exp: ',time) 
            
         
            Pontos=[50,100,200,300,500,750,1000,1250,1500,1750,2000]
            
            start = timeit.default_timer()
            ref=utils.calcula_loo(dataset,poly_exp,samples.T,model)
            stop = timeit.default_timer()
            timeRef=stop-start
            
            plt.axhline(y=ref, color='black', linestyle='--',label='Full Set Loo  Evaluation')
            
            mean=np.zeros(np.shape(Pontos)[0])
            std=np.zeros(np.shape(Pontos)[0])
            times=np.zeros(np.shape(Pontos)[0])
            
            n=20
            
            
            
            for k in range(np.shape(Pontos)[0]):
                loos=np.zeros(n)   
                time=0
                for i in range(n):
                    ind=random.sample(range(Ns), Pontos[k])
                    start = timeit.default_timer()
                    loos[i]=utils.calcula_loo( dataset ,poly_exp, samples.T ,model,I=ind)
                    stop = timeit.default_timer()
                    time+=(stop-start)/n
                mean[k]=np.mean(loos)
                std[k]=np.std(loos)
                times[k]=time
                print('Samples',Pontos[k],'Mean time LOO: ',time,'Mean : ',np.mean(loos),'STD LOO: ',np.std(loos)) 
        
    plt.plot(Pontos,mean)  
    plt.fill_between(Pontos,mean-std,mean+std, alpha=0.7,label="STD")
    plt.legend(loc='best')
    plt.xlabel('Number of points in Evalualtion', fontsize=18)
    plt.ylabel('Loo Error', fontsize=16)
    plt.show()
    
    plt.plot(Pontos,times)  
    plt.legend(loc='best')
    plt.axhline(y=timeRef, color='black', linestyle='--',label='Full Set Loo  Evaluation Time')
    plt.xlabel('Number of points in evalualtion', fontsize=14)
    plt.ylabel('Evaluation Avarage Time (s)', fontsize=16)
    plt.show()
    
    
    
    
    print(mean)
    print(std)
    print(times)           
         
from generateDatasets import *
from surrogatefromfile import *

import os    

ti=0
ti=000
tf=400
dt=0.01
dtS=1
TTCellModel.setSizeParameters(ti, tf, dt, dtS)

Ns,Nv=10000,100000

Timepoints=TTCellModel.getEvalPoints()
size=np.shape(Timepoints)[0]    


folder="./r/"
try:
    os.mkdir(folder)

except:
    print("folder ocuppied, rewriting")
       
#generateDataset(False, False, folder,Ns,Nv,False)    

LOOEXP(False,False,folder,Ns)