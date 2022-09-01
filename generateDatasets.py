# -*- coding: utf-8 -*-
"""
Created on Sat May  7 00:17:43 2022

@author: yanbw
-"""

import os
import numpy as np
import chaospy as cp
import timeit
from modelTT import TTCellModel
import math
import csv
import ray
import copy
from scipy.spatial import KDTree as kd
from utils import *


def generateDataset(dist,folder,Ns,Nv,out=False,nx=False,ny=False,remove_closest=False):
    #PARAMETERS
    
    #Parameters of interest X
    
      
    
    samples = dist.sample(Ns)

    
    #Fitting parameters
    nPar=3
  
    #m * Np # number of samples
    
    #normalization options
    normalizarqoi=ny
    normalizarX=nx
    normalizationMethod='normal'  #normal or skt
    
    
    if(True):
        print('Generating Sets')
        print("Samples",Ns)
        print("Samples Validation ",Nv)
        

    
    f=folder
    
    
    #Sample the parameter distributio
    start = timeit.default_timer()
    samples = dist.sample(Ns,rule="latin_hypercube")
    samplesaux=copy.copy(samples)
    stop = timeit.default_timer()
    if(out):        
       print('Time to sample Dist: ',stop-start)
    
    
    start = timeit.default_timer()
    samplesV = dist.sample(Nv ,rule="latin_hypercube")
    samplesVaux=copy.copy(samplesV)
    stop = timeit.default_timer()
    if(out):
        print('Time to sample Dist V: ',stop-start)
    
    
    #Run training set
    
    start = timeit.default_timer()
    sols= TTCellModel.run(samples.T,use_gpu=True,regen=True,name="tS.txt")
    stop = timeit.default_timer()
   # print(sols)
    ads50=[sols[i]["ADP50"] for i in range(Ns)]
    ads90=[sols[i]["ADP90"] for i in range(Ns)]
    dVmaxs=[sols[i]["dVmax"] for i in range(Ns)]
    vrest= [sols[i]["Vreps"] for i in range(Ns)]
    
    qoi={
         "ADP50":ads50,
         "ADP90":ads90,
         "Vrest":vrest,
         "dVmax":dVmaxs,
         
         }
    
    
    print('\n Time to run Model training Set: ',stop-start)
    
    
    
    
    print("\n")
    #Run validation set
    
    start = timeit.default_timer()
    sols= TTCellModel.run(samplesV.T,use_gpu=True,regen=True,name="vS.txt")
    stop = timeit.default_timer()
    print('\n Time to run Model Validation set: ',stop-start)
    start = timeit.default_timer()
    ads50=[sols[i]["ADP50"] for i in range(Nv)]
    ads90=[sols[i]["ADP90"] for i in range(Nv)]
    dVmaxs=[sols[i]["dVmax"] for i in range(Nv)]
    vrest= [sols[i]["Vreps"] for i in range(Nv)]
    stop = timeit.default_timer()
    print('\n Time to unpack Validation set: ',stop-start)
    
    print("Treating Sets ")
    qoiV={
         "ADP50":ads50,
         "ADP90":ads90,
         "Vrest":vrest,
         "dVmax":dVmaxs,
         
         }
    
    
    
    
    start = timeit.default_timer()
    # #NORMALIZE
  
    if(out):    
        print("Normalizing")
    if(normalizarX):
        for i in range(nPar): #even if the output is a non-normalized array, a normalized array is used for treating the set
            samples[i],samplesV[i]=normalizeTwoArrays(samples[i], samplesV[i],0,1, normalizationMethod)
    elif(out):
        print("Warning: not normalizing X, for X not given in units, migth mess up closest points removal")
    if(normalizarqoi):
         for qlabel in qoi:  
            qoi[qlabel],qoiV[qlabel]=normalizeTwoArrays(qoi[qlabel],qoiV[qlabel], 0,1,normalizationMethod)
    
    
    

    #Treat validation set
    if(out):
        stop = timeit.default_timer()
        print('\n Time to normalize ',stop-start)  
        print("Treating validation dataset")
        print("Removing closest points")
        start = timeit.default_timer()
        
        
    
    
    #REMOVE CLOSES POINT IN VALIDATION SET TO EACH POINT IN TRAINING SET
    difs,i,retrys=np.zeros(Ns),0,0
    
    idtoremv=np.zeros(Ns,dtype=int)-1
    svt=samplesV.T
    kdt=kd(samplesV.T)
    if(remove_closest==True):
        for sample in samples.T:
            t=0 # trys
            flag=True
            while(flag):
                 hit,ii = kdt.query(sample,k=t+1) #find the closes point, if the point is already mark to be removed, select the next closest until find a point not marked
                 if(t>=1):
                     ii=ii[t]
                     difs[i]=hit[t]
                 else :
                     difs[i]=hit
                    
                 if(False==np.any(idtoremv==ii)):    
                     flag=False
                     break;
                 else:
                     retrys=retrys+1
                     t=t+1
            
           
            idtoremv[i]=ii
            i=i+1    
        
        
        if(normalizarX==False):
            svt=np.delete(samplesVaux.T,idtoremv,0)
        else:
            svt=np.delete(svt,idtoremv,0)
            
        for ql,dt in qoiV.items():
            qoiV[ql]=np.delete(dt,idtoremv,0)
            
        
        
        
    if(out and remove_closest==True):
            
        print("AVG DISTANCE",np.mean(difs))
        print("MAX DISTANCE",np.max(difs))
        print("Min DISTANCE",np.min(difs))
        print("EXACT MATCHES",np.count_nonzero(difs<=0.01)) ##0.01 tolerance
        print("Retrys",retrys)
    
    samplesV=svt.T
    
    
    
    stop = timeit.default_timer()
    print('\n Time to treat sets: ',stop-start)
    
    ##Write results
    
    
    try:   
        os.mkdir(f)
        
    except:
        if(out):
        
            print("Updating training set File")
    
    try:
        os.mkdir(f+"/validation")
    except:
        if(out):
        
            print("Updating validation set File")
    
    file=open(f+"X.csv","w", newline='') 
    writer = csv.writer(file)
    if(normalizarX==False):
        samples=samplesaux
        
    for row in samples.T:
        writer.writerow(row)
    file.close()
    
    file=open(f+"/validation/"+"X.csv","w",newline='')
    writer = csv.writer(file)
    for row in samplesV.T:
        writer.writerow(row)
    file.close()
    
    
    
    for qlabel in qoi:
        Y=qoi[qlabel]
        Yval=qoiV[qlabel]
        file=open(f+qlabel+".csv","w", newline='')
        np.savetxt(file,Y, fmt='%.8f')
        file.close()
        file=open(f+"/validation/"+qlabel+".csv","w", newline='')
        np.savetxt(file,Yval, fmt='%.8f')
        file.close()
    
    
    print("Done")
