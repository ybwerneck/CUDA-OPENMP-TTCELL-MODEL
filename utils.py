# -*- coding: utf-8 -*-
"""
Created on Thu May 12 01:43:13 2022

@author: yanbw
"""
import numpy as np
import chaospy as cp
from SALib.sample import saltelli
from SALib.analyze import sobol
from sklearn.preprocessing import normalize as normalizeSkt
import ray

PROCESSN=5

def init():
    try:
        ray.init(ignore_reinit_error=False,num_cpus=PROCESSN,log_to_driver=False)   
    except:
        ray.shutdown()
        ray.init(ignore_reinit_error=False,num_cpus=PROCESSN,log_to_driver=False)    


def normalizeTwoArrays(x,y,mmin,mmax,method='normal'): #append two arrays, normalize as a single array, re-separate and return
    
    nx=np.shape(x)[0]
    ny=np.shape(y)[0]
    
    temp=np.zeros(nx+ny)
    temp[0:nx]=x
    temp[-ny:]=y
    
    if(method=='normal'):
        temp=normalize(temp,mmin,mmax)
    elif(method=='skt' or method=="SKT"):
        if(mmin!=0 or mmax!=1):
            raise("SKT only accepts unit normalization")
        temp=normalizeSkt(temp.reshape(1,-1))[0]
    else:
        raise("Invalid method")
    
    return temp[0:nx],temp[-ny:]
    
    
def normalize(arr, t_min=0, t_max=1):
    norm_arr = []
    diff = t_max - t_min
    diff_arr = max(arr)-min(arr)    
    for i in arr:
        temp = (((i - min(arr))*diff)/diff_arr) + t_min
        norm_arr.append(temp)
    return norm_arr




def SobolPCE(problem,Nsobol,modelPCE):
    
    param_vals = saltelli.sample(problem,Nsobol,  calc_second_order=False)
    Ns = param_vals.shape[0]
    Y=np.empty([Ns])
    for i in range(Ns):
        Y[i]=modelPCE(param_vals[i])  
    sensitivity = sobol.analyze(problem,Y , calc_second_order=False)
    return sensitivity['S1']

def readF(fn):
    X=[]
    file = open(fn, 'r')
    for row in file:
        X.append([float(x) for x in row.split(',')])
    return X



      
@ray.remote
def looT(samps,y,idx,deltas,base,model):
    
    
     nsamp = np.shape(y)[0]
     indices = np.linspace(0,nsamp-1,nsamp, dtype=np.int32)
     indices = np.delete(indices,idx)
     subs_samples = samps[indices,:].copy()
     subs_y =[ y[i] for i in (indices)]

     subs_poly = cp.fit_regression (base,subs_samples.T,subs_y,model=model,retall=False) 
     
     yhat = cp.call(subs_poly, samps[idx,:])
     del(subs_y)
     del subs_samples
     del indices
     return ((y[idx] - yhat))**2
     


@ray.remote
def calcula_deltas(y,poly_exp,samps,model,ii,ifn,ind):
 nsamp = np.shape(y)[0] 
 blocoDeltas=np.zeros(ifn-ii + 1)   
 indices = np.linspace(0,nsamp-1,nsamp, dtype=np.int32)
 for k in range(ii,ifn):
        i=ind[k]  
        indicesa = np.delete(indices,i)
        subs_samples = samps[indicesa,:].copy()
        subs_y =[ y[i] for i in (indicesa)]
        subs_poly = cp.fit_regression (poly_exp,subs_samples.T,subs_y,model=model,retall=False) 
        yhat = cp.call(subs_poly, samps[i,:])
        blocoDeltas[k-ii] = ((y[i] - yhat))**2
 return blocoDeltas
        
        
def calcula_loo(y, poly_exp, samples,model,I=-1):

    
    #PARALALLE LOO CALC
    ind=I
    #SERIAL LOO CALC
    NS = np.shape(y)[0]
    if(ind==-1):
        ind=range(NS)  
    treads={};
    deltas = np.zeros(NS)  
    samps = samples.T
    nsamp=np.shape(ind)[0]
    


    blocksize=int(nsamp/PROCESSN)
    treads={};
          
    deltas=np.zeros(nsamp)
    k=0
    for i in range(PROCESSN):
        treads[i]=calcula_deltas.remote(y, poly_exp, samps,model,blocksize*i,blocksize*(i+1),ind)
    for x in range(PROCESSN): 
        blocSols=ray.get(treads[x])
        for s in range(blocksize) :
            deltas[k]=blocSols[s]
            k=k+1




   
    y_std = np.std(y)
    err = np.mean(deltas)/np.var(y)
    acc = 1.0 - np.mean(deltas)/np.var(y)
    return err
  
@ray.remote
def runModel(samples,nsamp,model):
    
   R={}
  
   i=0
   for i in range(nsamp):
        R[i]= model(samples[i])
  
   return R
   

def runModelParallel(samples,model):
      nsamp = np.shape(samples)[1]
      blocksize=int(nsamp/PROCESSN)
      treads={};
      
      Y={}
      k=0
      for i in range(PROCESSN):
          treads[i]=runModel.remote(samples.T[blocksize*i:blocksize*(i+1)],int(nsamp/PROCESSN),model)
      for x in range(PROCESSN): 
          blocSols=ray.get(treads[x])
          for s in range(blocksize) :
              Y[k]=blocSols[s]
              k=k+1
      return Y
def calcula_looSingle(y, poly_exp, samples,model,I=-1):
    
 
    ind=I
    #SERIAL LOO CALC
    nsamp = np.shape(y)[0]
    if(ind==-1):
        ind=range(nsamp)
    deltas = np.zeros(np.shape(ind)[0])
    samps = samples.T
    k=0
    for i in ind:
        indices = np.linspace(0,nsamp-1,nsamp, dtype=np.int32)
        indices = np.delete(indices,i)
        subs_samples = samps[indices,:].copy()
        subs_y =[ y[i] for i in (indices)]
        subs_poly = cp.fit_regression (poly_exp,subs_samples.T,subs_y,model=model,retall=False) 
        yhat = cp.call(subs_poly, samps[i,:])
        deltas[k] = ((y[i] - yhat))**2
        k+=1

    y_std = np.std(y)
    err = np.mean(deltas)/np.var(y)
    acc = 1.0 - np.mean(deltas)/np.var(y)
    
    return err

def getRefSobol():
    sensitivity={}
    sensitivity['S5']=[1.55538786e-02, 1.82850519e-01, 9.61438769e-02, 2.99173145e-03,
     1.56245611e-05, 7.60818660e-01]
    sensitivity['S9']=[ 5.31159406e-02 , 1.68227767e-01 , 1.13413410e-01, -3.96239115e-04,
      1.25679367e-03, 7.20783737e-01 ]
    sensitivity['SVM']=[3.98217241e-04 , 3.09025984e-05, -4.28701329e-05 , 1.58306471e-05,
      9.94885997e-01 , 3.35410661e-04 ]
    sensitivity['SVR']=[7.48449580e-01,3.65262866e-03, 2.01777153e-03 ,4.39363109e-04,
     1.82630159e-05, 2.43027627e-01]
    return sensitivity
