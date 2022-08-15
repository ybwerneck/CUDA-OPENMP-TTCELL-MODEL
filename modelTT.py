###Base e modelo 
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
import re
import collections
import os
class TTCellModel:
    tf=400
    ti=0
    dt=0.01
    dtS=1
    parametersN=["ki","ko","gna","gca","atp"]
    K_o_default=5.40e+00
    g_CaL_default=1.750e-04
    g_Na_default=1.48380e+01
    K_i_default=138.3
    atp_default=2.6
    
    
    @staticmethod
    def run(P,use_gpu=False):  
        parametersS=[TTCellModel.cofs(p) for p in P]
        with open('m.txt','wb') as f:
             np.savetxt(f,parametersS, fmt='%.8f')

        TTCellModel.callCppmodel(np.shape(P)[0],use_gpu)
        name="out.txt"        
        X=[]
        file = open(name, 'r')
        for row in file:
           aux=[]
           for x  in row.split(' '):
               try:
                   aux.append(float(x))
               except:
                   aux.append(-100)
           ads=TTCellModel.ads(aux,[0.5,0.9] )
           k={"Wf": aux[:-1],"dVmax":aux[-1],"ADP90":ads[1],"ADP50":ads[0],"Vrepos":aux[-2]}
          
           X.append(k)
   
        return X
        
    def printModel(self,printresult=False):
        print(self.parameters,TTCellModel.cofs(self.parameters))
    
    @staticmethod
    def setParametersOfInterest(parametersN):
        TTCellModel.parametersN=parametersN
        
    @staticmethod
    def cofs(ps):

        params=[
             (1-0.25*ps[2])*TTCellModel.g_Na_default,
            (1-0.25*ps[2])*TTCellModel.g_CaL_default,
            
             138.3 - 13.3*ps[2],
            5.4 + 4.6*ps[1],
            5.6 - 3 * ps[0]          
            
            ]
     
        return np.array(params)
    

    @staticmethod
    def getSimSize(): #Returns size of result vector for given simulation size parameters, usefull for knowing beforehand the number of datapoints to compare
        n=(tf-ti)/dt
        return n
    
    @staticmethod
    def setSizeParameters(ti,tf,dt,dtS):
        TTCellModel.ti=ti
        TTCellModel.tf=tf
        TTCellModel.dt=dt
        TTCellModel.dtS=dtS
        
    @staticmethod   #runs the model once for the given size parameters and returns the time points at wich there is evalution
    def getEvalPoints():
        
        t=0
        ts=0
        ep=[]
        while(t<TTCellModel.tf):
            
            if (ts >= TTCellModel.dtS) :
                ep.append(t)
               	ts = 0;
            t=t+TTCellModel.dtS
            ts=ts+TTCellModel.dtS
            
        return ep    
    
    @staticmethod      
    def ads(sol,repoCofs): ##calculo da velocidade de repolarização
        k=0
        i=0;
        out={}
        x=sol
        flag=0
  
        x=np.array(sol)
        index=0
        idxmax=0
        for value in x:
                
           index+=1  
           if(value==x.max()):
                        flag=1                
                        out[len(repoCofs)]=index  + TTCellModel.ti
                        idxmax=index
           if(flag==1):
                        k+=1
           if(flag==1 and repoCofs[i]*x.min() >= value):
                        out[i]= (k)
                        i+=1
           if(i>=len(repoCofs)):
               
                        break
             
   
        return out

    @staticmethod
    def callCppmodel(N,use_gpu=False):     
        
        name="C:/Faculdade/Novapasta/numeric-models/uriel-numeric/CudaRuntime/x64/Release/CudaRuntime.exe"
        if os.name == 'nt':
            name="C:/Faculdade/Novapasta/numeric-models/uriel-numeric/CudaRuntime/x64/Release/CudaRuntime.exe"
        args=name +" --tf="+str(TTCellModel.tf)+" --ti="+str(TTCellModel.ti)+" --dt="+str(TTCellModel.dt)+" --dt_save="+str(TTCellModel.dtS) +" --n="+str(N)+" "  
        if(use_gpu):
            args=args+"--use_gpu=1"
        print(args)
        output = subprocess.Popen(args,stdout=subprocess.PIPE,shell=True)
        print( output.stdout.read().decode("utf-8"))
        # matrix={}
  
        # try:
        #     string = output.stdout.read().decode("utf-8")
        #     matrix = np.matrix(string)
        #     print(matrix)
        
        # except:
        #     print(args)
        #     print(string)
        #     print("\n")



       
        # try: 
        #     ads=TTCellModel.ads(matrix[:-1,1],[0.5,0.9])
        #     return {"Wf": matrix[:-1],"dVmax":matrix[-1,1],"ADP90":ads[1],"ADP50":ads[0],"Vrepos":matrix[-2,1]}
        # except:
             
        #      print(args)
        #      print("out",string)
        #      print(params)
        #      return TTCellModel.callCppmodel(params)
  
  

   
    
    