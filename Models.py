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
from scipy.spatial import KDTree as kd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import random

    
def NModel(x,y,dist):
        
        def model (clf,X):
            return clf.predict(X)        
        
        clf =  MLPRegressor(random_state=1, max_iter=1500)
        clf.fit(x, y)
        
        
        return partial(model,clf)
    
def GPModel(x,y,dist):
        
        def model (gpr,X):
            return gpr.predict(np.array(X),return_std=False)
        
        
        
        gpr=GaussianProcessRegressor(
        random_state=0,copy_X_train=False)
        gpr.fit(x, y)
        
        
        
        
        return partial(model,gpr)
                
def PCEModel(x,y,dist,P=2,regressor=None) :
        def model(pce,X):
            return cp.call(pce,np.array(X).T)
        
        poly_exp = cp.generate_expansion(P, dist,rule="three_terms_recurrence")
        fp = cp.fit_regression (poly_exp,np.array(x).T,y,model=regressor)  
        
        
        return partial(model,fp)
    
    
    
    
    ##Available regressor
           # "LARS": lm.Lars(**kws,eps=eps),
           # "OLS SKT": lm.LinearRegression(**kws),
            #"ridge"+str(alpha): lm.Ridge(alpha=alpha, **kws),
          #  "OMP"+str(alpha):
           # lm.OrthogonalMatchingPursuit(n_nonzero_coefs=3, **kws),
            
          #  "bayesian ridge": lm.BayesianRidge(**kws),
          #  "elastic net "+str(alpha): lm.ElasticNet(alpha=alpha, **kws),
            #"lasso"+str(alpha): lm.Lasso(alpha=alpha, **kws),
            #"lasso lars"+str(alpha): lm.LassoLars(alpha=alpha, **kws),
            