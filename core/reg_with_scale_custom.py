
import numpy as np
from .scale_vector import scale_vector
from .scale_estimation import scale_estimation
from .lv_map_to_pt import lv_map_to_pt
from .build_pt_estimate import build_pt_estimate
from .irls_sa_cauchy_point import irls_sa_cauchy_point
from .get_inlier_sum import get_inlier_sum


def reg_with_scale_custom(X, Y, bound, lbound, interval, ifestimateScale, distances, ky_point, scale, noise_val=0.01):
    """
    Customizable registration with scale estimation
    
    Parameters:
    -----------
    noise_val : float
        Noise value in meters (default: 0.01 for KITTI)
    """
    distance_th = 100
    
    if ifestimateScale == 0:
        bound = 2 * np.sqrt(3) * noise_val
        lbound = 5 * bound
    
    Sxy, Snoise, _, _, map_arr = scale_vector(X, Y, bound, lbound)
    bestS, inliers, _ = scale_estimation(Sxy, Snoise, interval, ifestimateScale, bound, scale)
    
    Xin, Yin, distances_in, inlierPTS = lv_map_to_pt(X, Y, map_arr, inliers, 50, distances)
    
    S_result = bestS
    
    if ifestimateScale == 0:
        bound = 2 * np.sqrt(3) * noise_val
        lbound = 5 * bound
    else:
        bound = noise_val
        lbound = bound * 5
    
    Yin = Yin
    Xin = bestS * Xin
    X = bestS * X
    
    point_size = Xin.shape[1]
    if ky_point > point_size:
        ky_point = point_size
    
    P, T, pt_map = build_pt_estimate(Xin, Yin, ky_point)
    print('PT finally')
    
    bestS = 1
    bestR = np.eye(3)
    best_T = np.zeros(3)
    
    if Xin.shape[1] > 0:
        bound_local = 2 * np.sqrt(3) * noise_val
        oth_R, oth_T, _ = irls_sa_cauchy_point(Xin, Yin, 100, bound_local)
        
        inlier_cout = get_inlier_sum(X, Y, oth_R, oth_T)
        
        bestR = oth_R
        best_T = oth_T
    
    return S_result, bestR, best_T

