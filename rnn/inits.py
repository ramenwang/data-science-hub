import tensorflow as tf 
import numpy as np 

def uniform(nrow, ncol):
    return np.random.uniform(-np.sqrt(1.0/ncol), np.sqrt(1.0/ncol), (nrow, ncol))

