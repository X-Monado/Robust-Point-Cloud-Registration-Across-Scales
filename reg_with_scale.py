import numpy as np
from scale_vector import scale_vector
from scale_estimation import scale_estimation
from lv_map_to_pt import lv_map_to_pt
from build_pt_estimate import build_pt_estimate
from irls_sa_cauchy_point import irls_sa_cauchy_point
from get_inlier_sum import get_inlier_sum


def ransac_pre_filter(Xin, Yin, n_iterations=50, threshold=0.05):
    """
    RANSAC预筛选：快速剔除明显的外点
    
    Parameters:
    -----------
    Xin : numpy.ndarray (3, N)
    Yin : numpy.ndarray (3, N)
    n_iterations : int - RANSAC迭代次数
    threshold : float - 内点距离阈值
    
    Returns:
    --------
    inlier_mask : numpy.ndarray (N,) - 内点掩码
    best_R : numpy.ndarray - 最佳旋转矩阵
    best_t : numpy.ndarray - 最佳平移向量
    """
    n_points = Xin.shape[1]
    
    if n_points < 4:
        return np.ones(n_points, dtype=bool), np.eye(3), np.zeros(3)
    
    best_inliers = 0
    best_inlier_mask = np.ones(n_points, dtype=bool)
    best_R = np.eye(3)
    best_t = np.zeros(3)
    
    for _ in range(n_iterations):
        # 随机选择3个点
        idx = np.random.choice(n_points, 3, replace=False)
        X_sample = Xin[:, idx]
        Y_sample = Yin[:, idx]
        
        try:
            # 计算变换（使用SVD）
            X_centroid = np.mean(X_sample, axis=1).reshape(-1, 1)
            Y_centroid = np.mean(Y_sample, axis=1).reshape(-1, 1)
            
            X_centered = X_sample - X_centroid
            Y_centered = Y_sample - Y_centroid
            
            H = np.dot(X_centered, Y_centered.T)
            U, _, Vt = np.linalg.svd(H)
            
            d = np.linalg.det(np.dot(Vt.T, U.T))
            sign_matrix = np.diag([1, 1, np.sign(d)])
            R = np.dot(Vt.T, np.dot(sign_matrix, U.T))
            t = Y_centroid.flatten() - np.dot(R, X_centroid.flatten())
            
            # 计算内点
            Bfit = np.dot(R, Xin) + t.reshape(-1, 1)
            errors = np.sqrt(np.sum((Bfit - Yin) ** 2, axis=0))
            inlier_mask = errors < threshold
            
            n_inliers = np.sum(inlier_mask)
            if n_inliers > best_inliers:
                best_inliers = n_inliers
                best_inlier_mask = inlier_mask
                best_R = R.copy()
                best_t = t.copy()
                
                if n_inliers > 0.8 * n_points:
                    break
                    
        except Exception:
            continue
    
    return best_inlier_mask, best_R, best_t


def reg_with_scale(X, Y, distances, scale):
    """
    Registration with scale estimation (Optimized v3 for GeoTransformer)
    
    Key improvements for large-scale matching:
    - Adaptive parameters based on number of matches
    - Distance-based pre-filtering to remove outliers
    - Multi-stage refinement strategy
    - Enhanced robustness for small match sets
    
    Parameters:
    -----------
    X : numpy.ndarray - Source point cloud, shape (3, N)
    Y : numpy.ndarray - Target point cloud, shape (3, N)
    distances : numpy.ndarray - Distances for each correspondence, shape (N,)
    scale : float - Given scale factor
    
    Returns:
    --------
    S_result : float - Estimated scale
    bestR : numpy.ndarray - Estimated rotation matrix, shape (3, 3)
    best_T : numpy.ndarray - Estimated translation vector, shape (3,)
    """
    
    noise = 0.01
    bound = 2 * np.sqrt(3) * noise
    lbound = 5 * bound
    
    n_matches = X.shape[1]
    
    # Scale estimation
    Sxy, Snoise, _, _, map_arr = scale_vector(X, Y, bound, lbound)
    bestS, inliers, _ = scale_estimation(Sxy, Snoise, 2, 0, bound, scale)
    
    # Map back to points - keep more candidates initially
    Xin_raw, Yin_raw, distances_in, inlierPTS = lv_map_to_pt(X, Y, map_arr, inliers, 80, distances)
    
    S_result = bestS
    
    # Apply scale
    Xin_raw = bestS * Xin_raw
    X_scaled = bestS * X
    
    if Xin_raw.shape[1] < 5:
        return S_result, np.eye(3), np.zeros(3)
    
    n_points = Xin_raw.shape[1]
    
    # Adaptive parameter selection based on data size
    if n_points > 1500:
        # Large dataset: use more aggressive filtering
        ransac_iter = 80
        ransac_thresh = 0.02
        irls_iter = 150
        num_trials = 2
    elif n_points > 800:
        # Medium dataset: balanced approach
        ransac_iter = 120
        ransac_thresh = 0.025
        irls_iter = 250
        num_trials = 4
    else:
        # Small dataset: more iterations and trials for robustness
        ransac_iter = 200
        ransac_thresh = 0.03
        irls_iter = 300
        num_trials = 5
    
    # Step 1: RANSAC预筛选，快速剔除明显外点
    ransac_mask, R_ransac, t_ransac = ransac_pre_filter(
        Xin_raw, Yin_raw, 
        n_iterations=ransac_iter, 
        threshold=ransac_thresh
    )
    Xin_filtered = Xin_raw[:, ransac_mask]
    Yin_filtered = Yin_raw[:, ransac_mask]
    
    if Xin_filtered.shape[1] < 4:
        return S_result, R_ransac, t_ransc
    
    # Step 2: IRLS with multiple trials and adaptive bounds
    bestR = np.eye(3)
    best_T = np.zeros(3)
    best_score = -float('inf')
    
    trial_results = []
    
    for trial in range(num_trials):
        try:
            # Different noise bounds for robustness (tighter for larger datasets)
            if n_points > 1500:
                bounds = [0.006, 0.008, 0.010]
            else:
                bounds = [0.006, 0.008, 0.010, 0.012, 0.015]
            
            bound_local = 2 * np.sqrt(3) * bounds[trial % len(bounds)]
            
            oth_R, oth_T, W = irls_sa_cauchy_point(
                Xin_filtered, Yin_filtered, 
                iter_num=irls_iter, 
                sigma=bound_local / (2 * np.sqrt(3))
            )
            
            # Compute comprehensive score
            Bfit = np.dot(oth_R, Xin_filtered) + oth_T.reshape(-1, 1)
            errors = np.sqrt(np.sum((Bfit - Yin_filtered) ** 2, axis=0))
            
            # Score based on: inlier ratio, median error, weight concentration
            tight_threshold = 0.03 if n_points > 800 else 0.04
            inlier_ratio = np.mean(errors < tight_threshold)
            median_error = np.median(errors)
            mean_error = np.mean(errors)
            
            # Combined score (higher is better)
            score = inlier_ratio * 15 - median_error * 30 - mean_error * 10
            
            trial_results.append({
                'R': oth_R,
                'T': oth_T,
                'W': W,
                'score': score,
                'median_error': median_error,
                'mean_error': mean_error,
                'inlier_ratio': inlier_ratio,
                'errors': errors
            })
            
        except Exception as e:
            continue
    
    if len(trial_results) == 0:
        return S_result, R_ransac, t_ransac
    
    # Sort by score and select the best
    trial_results.sort(key=lambda x: x['score'], reverse=True)
    
    # Validate the best result
    best_trial = trial_results[0]
    
    # Check for abnormal results
    rot_diff_from_identity = np.arccos(np.clip((np.trace(best_trial['R']) - 1) / 2, -1, 1))
    
    # More strict validation for small datasets
    error_threshold = 0.3 if n_points < 1000 else 0.5
    rot_threshold = np.pi/3 if n_points < 1000 else np.pi/2
    
    if rot_diff_from_identity > rot_threshold and best_trial['median_error'] > error_threshold:
        # Abnormal result detected! Try alternatives
        found_better = False
        
        for alt_trial in trial_results[1:min(4, len(trial_results))]:
            alt_rot_diff = np.arccos(np.clip((np.trace(alt_trial['R']) - 1) / 2, -1, 1))
            if alt_rot_diff < rot_threshold and alt_trial['median_error'] < error_threshold:
                bestR = alt_trial['R']
                best_T = alt_trial['T']
                found_better = True
                break
        
        if not found_better:
            # Use RANSAC result as fallback
            bestR = R_ransac
            best_T = t_ransac
    else:
        bestR = best_trial['R']
        best_T = best_trial['T']
    
    return S_result, bestR, best_T
