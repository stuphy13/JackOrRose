import numpy as np
import pandas as pd

def scaled(matrix):
    mean = np.mean(matrix, axis=0)
    std = np.std(matrix, axis=0)
    std[std==0]=1e-8
    mat_scaled = (matrix - mean)/std
    return mat_scaled, mean, std




