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



import sys
from numbers import Number
from collections import deque
from collections.abc import Set, Mapping


ZERO_DEPTH_BASES = (str, bytes, Number, range, bytearray)


def getsize(obj_0):
    """Recursively iterate to sum size of object & members."""
    _seen_ids = set()
    def inner(obj):
        obj_id = id(obj)
        if obj_id in _seen_ids:
            return 0
        _seen_ids.add(obj_id)
        size = sys.getsizeof(obj)
        if isinstance(obj, ZERO_DEPTH_BASES):
            pass # bypass remaining control flow and return
        elif isinstance(obj, (tuple, list, Set, deque)):
            size += sum(inner(i) for i in obj)
        elif isinstance(obj, Mapping) or hasattr(obj, 'items'):
            size += sum(inner(k) + inner(v) for k, v in getattr(obj, 'items')())
        # Check for custom object instances - may subclass above too
        if hasattr(obj, '__dict__'):
            size += inner(vars(obj))
        if hasattr(obj, '__slots__'): # can have __slots__ with __dict__
            size += sum(inner(getattr(obj, s)) for s in obj.__slots__ if hasattr(obj, s))
        return size
    return inner(obj_0)

    
def NModel(x,y,dist):
        
        def model (clf,X):
            return clf.predict(X)        
        
        clf =  MLPRegressor(random_state=1, max_iter=3000,hidden_layer_sizes=(20))
        clf.fit(x, y)
        
        
        return partial(model,clf)
    
def GPModel(x,y,dist,kernel):
        
        def model (gpr,X):
            return gpr.predict(np.array(X),return_std=False)
        
        
        
        gpr=GaussianProcessRegressor(kernel,
        random_state=0,copy_X_train=False)
        gpr.fit(x, y)
        

        
        
        return partial(model,gpr)

    ##Kernels
    
                
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
            