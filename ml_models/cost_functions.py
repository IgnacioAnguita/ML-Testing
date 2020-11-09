import numpy as np
from math import sqrt

def RMSE(y,y_hat):
    '''
    args
    ----
        y: true values (np array)
        y_hat: predicted values (np array)
     
     returns
     ----
        the root mean swquared error
    
    '''
    return sqrt(np.mean(np.square(y-y_hat)))
