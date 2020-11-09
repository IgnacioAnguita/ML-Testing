import numpy as np
from math import sqrt

def RMSE(y,y_hat):
    return sqrt(np.mean(np.square(y-y_hat)))