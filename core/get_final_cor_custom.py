
import time
import numpy as np
from .reg_with_scale_custom import reg_with_scale_custom


def get_final_cor_custom(X, Y, distances, scale, noise_val=0.01):
    """
    Customizable final correspondence function
    
    Parameters:
    -----------
    noise_val : float
        Noise value in meters (default: 0.01 for KITTI)
    """
    distance_th = 1.0
    noise = noise_val
    bound = 0.05 * np.sqrt(3) * noise
    ky_point = 150
    
    thre = 3 * noise
    nTest = 1
    
    tic = time.time()
    
    if len(distances.shape) > 1:
        distances = distances[:, 1]
    
    print('Size of X:', X.shape)
    
    bestS, bestR, best_T = reg_with_scale_custom(
        X, Y, bound, 5*bound, 2, 0, distances, ky_point, scale, noise_val
    )
    
    buildGraphTime = time.time() - tic
    print('Running graph time %.3f\n' % buildGraphTime)
    
    return buildGraphTime, bestS, bestR, best_T

