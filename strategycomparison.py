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
from utils import measureModelPerfomance as measureModel
import utils
from functools import partial
from scipy.spatial import KDTree as kd

import random



def surrogatefromfile(folder,Ns,qoi={"ADP50","ADP90","Vrest","dVmax","tdV"},out=False,sobolR=None,models=None,vali=True):

    utils.init()
    print("Start Surrogate from File")
    #Load validation files

    
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
    
  
    
  
          
       
       
       

       


   
        
    
    for mlabel,model in models.items():
        print('\n',"Model: ", mlabel,'\n')      
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

           
        for i in range(0,len(plotsaux)):
            plots.append(plotsaux.pop())
        pltidx=0
        fig.suptitle(mlabel)   
        
        
        crit={}
        Ypred={}
        for label in qoi:
            
            ##prepare input
            y = np.array(Y[label]).flatten()
            yv=np.array(Yval[label]).flatten()
            
            ##fit model
            predictor=model(np.array(X),y,dist)
       
        
            ##validate model
            ypred=runModel(np.array(Xv).T,predictor)     
            ypred=np.array([ypred[i] for i in ypred])
            
            errs=np.array(((ypred-yv)**2)/np.var(yv))
    

            ##store results
            crit[label]= np.where(errs>0.01) 
            Ypred[label]=ypred
            
            
            ##plot qoi
            p=plots.pop()   
            p.scatter(yv,ypred)
            p.set_title(label)           
            p.plot(yv,yv,"black",linewidth=2)
            p.set(xlabel="Y_true",ylabel="Y_pred")
            for ax in fig.get_axes():
                ax.label_outer() 
            p.get_figure().savefig(folder+"results/"+mlabel+"_validation_results.png")
        plt.show()
        
        
        
        ##post-process analasys of results
        for i in crit['dVmax'][0]:
                 
                    print(Ypred['dVmax'][i],Yval['dVmax'][i],'--',Ypred['tdV'][i],Yval['tdV'][i])
                    color='red'
                    plt.plot(wfs[i],color=color)
                    plt.show()
                    
        for i in range(0,np.shape(wfs)[0],int(np.shape(wfs)[0]/10)):
                    if(np.isin(i,crit['dVmax'][0])==False):
                        color='blue'
                        plt.plot(wfs[i],color=color)
        plt.show()
                      

def surrogatefromSet(X,Y,Xval,Yval,Ns,dist,folder="",qoi={"ADP50","ADP90","Vrest","dVmax","tdV"},out=False,sobolR=None,models=None,vali=True,plot=True,sampref=0):

   
    #Load validation files

    


    
    
    ##Load Result File 
 
    
    # create the csv writer

  
    #Load datasets
    
    #Training Samples
    nPar=6
    


        

    erros={}
    errosR={}
    errosMax={}
    timesamp={}
    timefit={}
    
    SAMPREF=sampref
    NSAMP=np.shape(np.array(Xval))[1]
    for ml,a in models.items():
        
        timefit[ml],timesamp[ml],erros[ml]=0,0,0
    for mlabel,model in models.items():
        if(out):
            print('\n',"Model: ", mlabel,'\n')      
        timeL=0
       
    
     
       
        crit={}
        Ypred={}
        
        erros[mlabel]={}
        errosR[mlabel]={}
        errosMax[mlabel]={}
        timefit[mlabel]={}
        timesamp[mlabel]={}
        for label in qoi:
            
            ##prepare input
            y = np.array(Y[label]).flatten()
            yv=np.array(Yval[label]).flatten()
            
            ##fit model
            start = timeit.default_timer()
            
            predictor=model(np.array(X).T,y)
       
            stop = timeit.default_timer()
            timefitting=stop-start
            timesample=measureModel(dist, predictor)
            if out:
                print(label)
                print("Time to fit",timefitting)
        
                
                print("Time to sample 10x resulting emulator",timesample)
                print("\n")
            ##validate model
            
            ypred=runModel(SAMPREF,predictor,NSAMP)     
            ypred=np.array([ypred[i] for i in ypred])
            
            errs=np.array(((ypred-yv)**2)**(1/2))
            
            
           
            errosMax[mlabel][label]=np.max(errs)/((yv[errs.argmax()]**2)**(1/2))
            errosR[mlabel][label]=np.mean((errs)/((yv**2)**(1/2)))
            errs=errs/np.var(yv)
            erros[mlabel][label]=np.mean(errs)
            timefit[mlabel][label]=timefitting
            timesamp[mlabel][label]=timesample
            
            
            ##store results
            crit[label]= np.where(errs>0.1) 
            Ypred[label]=ypred
            
            
            ##plot qoi
            if(plot==False):
                continue
          
                              
    return erros,timefit,timesamp,errosMax,errosR