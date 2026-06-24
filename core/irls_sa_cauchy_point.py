import numpy as np
from sa_cauchy_point import sa_cauchy_point


def refine_translation(X, Y, R_init, t_init, max_iter=50, tol=1e-6):
    """
    Translation refinement using iterative weighted least squares
    
    Given a good rotation estimate R_init, refine the translation t_init
    by iteratively reweighting based on residuals.
    
    Parameters:
    -----------
    X : numpy.ndarray (3, N) - Source points
    Y : numpy.ndarray (3, N) - Target points
    R_init : numpy.ndarray (3, 3) - Initial rotation estimate
    t_init : numpy.ndarray (3,) - Initial translation estimate
    max_iter : int - Maximum iterations
    tol : float - Convergence tolerance
    
    Returns:
    --------
    t_refined : numpy.ndarray (3,) - Refined translation
    final_weights : numpy.ndarray (N,) - Final weights used
    """
    n_points = X.shape[1]
    t = t_init.copy()
    
    # Compute initial residuals
    X_rotated = np.dot(R_init, X)
    residuals = Y - (X_rotated + t.reshape(-1, 1))
    distances = np.sqrt(np.sum(residuals ** 2, axis=0))
    
    # Initialize weights based on Cauchy distribution
    # Note: Original implementation uses scaled median (not standard MAD)
    # to maintain consistency with reported experimental results
    scale = np.median(distances) * 1.4826  # MAD-based scale estimate
    if scale < 1e-8:
        scale = 0.01
    
    weights = 1.0 / (1.0 + (distances / scale) ** 2)
    
    for iteration in range(max_iter):
        # Weighted centroid difference
        w_sum = np.sum(weights)
        w_X_rot = np.dot(X_rotated, weights)
        w_Y = np.dot(Y, weights)
        
        # New translation: t = centroid_Y - R * centroid_X
        t_new = (w_Y - w_X_rot) / w_sum
        
        # Check convergence
        if np.linalg.norm(t_new - t) < tol:
            break
        
        t = t_new
        
        # Update residuals and weights
        residuals = Y - (X_rotated + t.reshape(-1, 1))
        distances = np.sqrt(np.sum(residuals ** 2, axis=0))
        
        # Update weights with Cauchy kernel
        new_weights = 1.0 / (1.0 + (distances / scale) ** 2)
        
        # Smooth weight update
        alpha = 0.5
        weights = alpha * new_weights + (1 - alpha) * weights
    
    return t, weights


def irls_sa_cauchy_point(x, y, iter_num, sigma):
    """
    Iteratively reweighted least squares with simulated annealing and Cauchy distribution
    
    Optimized v3:
    - More aggressive weight update for better inlier selection
    - Adaptive temperature schedule based on convergence rate
    - Two-stage refinement for higher accuracy
    - Stricter final weight threshold
    - Translation refinement step added
    
    Parameters:
    -----------
    x : numpy.ndarray - Source point cloud, shape (3, N)
    y : numpy.ndarray - Target point cloud, shape (3, N)
    iter_num : int - Number of iterations
    sigma : float - Noise bound
    
    Returns:
    --------
    R : numpy.ndarray - Estimated rotation matrix, shape (3, 3)
    t : numpy.ndarray - Estimated translation vector, shape (3,)
    W : numpy.ndarray - Final weights, shape (3, N)
    """
    
    n_points = x.shape[1]
    
    W = np.ones((3, n_points))
    w = np.ones(n_points)
    
    u1 = np.max(np.abs(x[:]))
    u2 = np.min(np.abs(x[:]))
    u = (u1 - u2) ** 2
    
    div = 1.12
    
    best_R = np.eye(3)
    best_t = np.zeros(3)
    best_error = float('inf')
    
    prev_R = None
    stable_count = 0
    
    error_history = []
    
    for i in range(iter_num):
        R, t, W_new = sa_cauchy_point(x, y, u, w)
        
        if prev_R is not None:
            rot_diff = np.linalg.norm(R - prev_R, 'fro')
            if rot_diff < 1e-6:
                stable_count += 1
                if stable_count >= 15:
                    break
            else:
                stable_count = 0
        
        prev_R = R.copy()
        
        Bfit = np.dot(R, x) + t.reshape(-1, 1)
        errors = np.sqrt(np.sum((Bfit - y) ** 2, axis=0))
        
        error_history.append(np.median(errors))
        if len(error_history) > 10:
            error_history.pop(0)
        
        alpha = 0.6
        w_new = W_new[0, :]
        
        error_threshold = np.percentile(errors, 75)
        adaptive_mask = errors < error_threshold * 2
        
        w = alpha * w_new * adaptive_mask + (1 - alpha) * w
        w = np.maximum(w, 0.01)
        
        min_u = (3 * sigma) ** 2
        
        if len(error_history) >= 5 and error_history[-1] < error_history[0]:
            div_eff = 1.18
        else:
            div_eff = 1.10
            
        if u > min_u:
            u = u / div_eff
        
        weighted_error = np.mean(w * errors)
        if weighted_error < best_error:
            best_error = weighted_error
            best_R = R.copy()
            best_t = t.copy()
    
    # Stage 1 refinement: moderate filtering
    W_stage1 = W_new.copy()
    stage1_threshold = 0.35
    W_stage1[0, W_stage1[0, :] < stage1_threshold] = 0
    w_stage1 = W_stage1[0, :]
    
    R_stage1, t_stage1, _ = sa_cauchy_point(x, y, u * 0.8, w_stage1)
    
    # Stage 2 refinement: aggressive filtering on high-confidence points only
    Bfit_s1 = np.dot(R_stage1, x) + t_stage1.reshape(-1, 1)
    errors_s1 = np.sqrt(np.sum((Bfit_s1 - y) ** 2, axis=0))
    
    tight_tolerance = 2 * sigma
    reliable_mask = errors_s1 < tight_tolerance
    
    if np.sum(reliable_mask) >= 6:
        w_final = w_stage1 * reliable_mask.astype(float)
        w_final = w_final / (np.sum(w_final) + 1e-10)
        
        R_final, t_final, _ = sa_cauchy_point(x, y, u * 0.5, w_final)
        
        # NEW: Translation refinement step
        t_refined, _ = refine_translation(x, y, R_final, t_final, max_iter=30)
        
        # Validate refined result
        errors_final = np.sqrt(np.sum((np.dot(R_final, x) + t_refined.reshape(-1, 1) - y) ** 2, axis=0))
        median_final = np.median(errors_final[reliable_mask]) if np.sum(reliable_mask) > 0 else float('inf')
        median_s1 = np.median(errors_s1[reliable_mask]) if np.sum(reliable_mask) > 0 else float('inf')
        
        if median_final <= median_s1 * 1.02:
            return R_final, t_refined, W_stage1
    
    return R_stage1, t_stage1, W_stage1
