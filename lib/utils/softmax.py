"""
Reference Model
Copyright (c) 2018 MobaiTech Inc 
Author: Abinash Mohanty
"""
import numpy as np

def softmax(x):
    """Compute softmax values for each sets of scores in x."""

    sh = x.shape
    
    y=np.zeros(sh,dtype=np.float64)
    for i in range(sh[0]):
        y[i] =  np.exp(x[i]) / np.sum(np.exp(x[i]))
    return y