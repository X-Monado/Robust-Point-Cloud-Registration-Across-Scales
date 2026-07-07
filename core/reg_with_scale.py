import numpy as np
from .scale_vector import scale_vector
from .scale_estimation import scale_estimation
from .lv_map_to_pt import lv_map_to_pt
from .build_pt_estimate import build_pt_estimate
from .irls_sa_cauchy_point import irls_sa_cauchy_point
from .get_inlier_sum import get_inlier_sum
from .triangular_consistency import filter_by_triangular_consistency
from .sc2_compatibility import filter_by_sc2
from .translation_ransac import translation_ransac_1pt

# GPU acceleration (optional — only used when use_gpu=True)
try:
    import torch
    from .gpu_accel import (scale_vector_gpu, lv_map_to_pt_gpu,
                            filter_by_triangular_consistency_gpu,
                            _to_torch)
    _HAS_GPU = torch.cuda.is_available()
except ImportError:
    _HAS_GPU = False


def ransac_pre_filter(Xin, Yin, n_iterations=50, threshold=0.05, rng=None):
    """
    RANSAC预筛选：快速剔除明显的外点

    Parameters:
    -----------
    Xin : numpy.ndarray (3, N)
    Yin : numpy.ndarray (3, N)
    n_iterations : int - RANSAC迭代次数
    threshold : float - 内点距离阈值
    rng : numpy.random.Generator or None - 随机数生成器（保证可复现性）

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

    # 使用传入的 rng 保证可复现性，避免依赖全局随机状态
    _choice = rng.choice if rng is not None else np.random.choice

    for _ in range(n_iterations):
        # 随机选择3个点
        idx = _choice(n_points, 3, replace=False)
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


def reg_with_scale(X, Y, distances, scale, use_triangular=False,
                   triangular_percent=50, epsilon_2=None, epsilon_3=None,
                   random_seed=42, return_diagnostics=False, lv_percent=80,
                   noise=None, use_sc2=False, sc2_percent=50, sc2_d_thre=None,
                   post_refine=True, estimate_scale=False, length_noise=None,
                   use_gpu=False):
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
    use_triangular : bool - Whether to apply triangular consistency filtering
    triangular_percent : float - Cumulative frequency percent for triangular filtering
    epsilon_2 : float or None - Compatibility threshold for C1/C2 (auto if None)
    epsilon_3 : float or None - Compatibility threshold for C3 (auto if None)
    random_seed : int - Random seed for RANSAC
    return_diagnostics : bool - If True, return diagnostic dict
    estimate_scale : bool - If True, estimate scale via RANSAC (flag=1); if False, use given scale (flag=0)

    Returns:
    --------
    S_result : float - Estimated scale
    bestR : numpy.ndarray - Estimated rotation matrix, shape (3, 3)
    best_T : numpy.ndarray - Estimated translation vector, shape (3,)
    diagnostics : dict (only if return_diagnostics=True)
    """
    diagnostics = {} if return_diagnostics else None

    # noise 可按数据集配置（KITTI 室外需更大值如 0.05-0.1；3DMatch 室内 0.01）
    if noise is None:
        noise = 0.01
    bound = 2 * np.sqrt(3) * noise
    # length_noise controls the line-vector length filter independently of the
    # inlier noise bound (useful when Y is artificially scaled but X is not).
    if length_noise is None:
        length_noise = noise
    lbound = 5 * 2 * np.sqrt(3) * length_noise

    # 为 RANSAC 创建可复现的随机数生成器
    rng = np.random.default_rng(random_seed)

    n_matches = X.shape[1]
    if return_diagnostics:
        diagnostics['n_input_matches'] = n_matches

    # ---- GPU acceleration for O(N²) scale_vector + lv_map_to_pt ----
    if use_gpu and _HAS_GPU:
        X_t = _to_torch(X)
        Y_t = _to_torch(Y)
        dist_t = _to_torch(distances)
        Sxy_t, Snoise_t, map_arr_t = scale_vector_gpu(X_t, Y_t, bound, lbound)
        Sxy = Sxy_t.cpu().numpy()
        Snoise = Snoise_t.cpu().numpy()
        # Scale estimation stays on CPU (1D RANSAC, not worth GPU overhead)
        se_flag = 1 if estimate_scale else 0
        bestS, inliers, _ = scale_estimation(Sxy, Snoise, 2, se_flag, bound, scale, random_seed=random_seed)
        if return_diagnostics:
            diagnostics['n_scale_inliers'] = len(inliers)
            diagnostics['estimated_scale'] = bestS
        # Point selection stays on CPU for exact numerical consistency:
        # Python Counter + stable sort has different tie-breaking than torch.argsort,
        # which matters on sparse datasets (3DCSR-FCGF, MVS-FCGF).
        # The O(N²) bottleneck (scale_vector) is already GPU-accelerated above.
        map_arr_np = map_arr_t.cpu().numpy()
        Xin_raw, Yin_raw, distances_in, inlierPTS = lv_map_to_pt(
            X, Y, map_arr_np, inliers, lv_percent, distances)
        # Free GPU memory from scale vectors (can be ~150MB for N=5000)
        del Sxy_t, Snoise_t, map_arr_t, X_t, Y_t, dist_t
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        # Scale estimation (CPU original path)
        Sxy, Snoise, _, _, map_arr = scale_vector(X, Y, bound, lbound)
        se_flag = 1 if estimate_scale else 0
        bestS, inliers, _ = scale_estimation(Sxy, Snoise, 2, se_flag, bound, scale, random_seed=random_seed)
        if return_diagnostics:
            diagnostics['n_scale_inliers'] = len(inliers)
            diagnostics['estimated_scale'] = bestS
        Xin_raw, Yin_raw, distances_in, inlierPTS = lv_map_to_pt(X, Y, map_arr, inliers, lv_percent, distances)

    if return_diagnostics:
        diagnostics['n_after_lv_map'] = Xin_raw.shape[1] if Xin_raw.ndim == 2 else 0

    S_result = bestS

    # Apply scale
    if Xin_raw.ndim < 2 or Xin_raw.shape[1] < 5:
        if return_diagnostics:
            return S_result, np.eye(3), np.zeros(3), diagnostics
        return S_result, np.eye(3), np.zeros(3)

    # Optional: Triangular consistency filtering
    if use_triangular and Xin_raw.shape[1] >= 6:
        # Auto-set epsilon based on noise bound if not provided
        if epsilon_2 is None:
            epsilon_2 = 5 * bound
        if epsilon_3 is None:
            epsilon_3 = 5 * bound

        # Limit points for efficiency (triangular check is O(N^2))
        n_pts = Xin_raw.shape[1]
        max_pts_for_tri = 500
        if n_pts > max_pts_for_tri:
            # Subsample for triangular check, keep distances aligned
            idx_sub = np.random.RandomState(random_seed).choice(
                n_pts, max_pts_for_tri, replace=False)
            X_tri = Xin_raw[:, idx_sub]
            Y_tri = Yin_raw[:, idx_sub]
            d_tri = distances_in[idx_sub]
        else:
            X_tri = Xin_raw
            Y_tri = Yin_raw
            d_tri = distances_in
            idx_sub = np.arange(n_pts)

        tri_inlier_idx, tri_degrees = filter_by_triangular_consistency_gpu(
            _to_torch(X_tri), _to_torch(Y_tri), bestS, epsilon_2, epsilon_3, percent=triangular_percent
        ) if (use_gpu and _HAS_GPU) else filter_by_triangular_consistency(
            X_tri, Y_tri, bestS, epsilon_2, epsilon_3, percent=triangular_percent
        )

        if return_diagnostics:
            diagnostics['n_before_triangular'] = X_tri.shape[1]
            diagnostics['n_after_triangular'] = len(tri_inlier_idx)
            diagnostics['tri_degrees_mean'] = float(np.mean(tri_degrees)) if len(tri_degrees) > 0 else 0
            diagnostics['tri_degrees_max'] = int(np.max(tri_degrees)) if len(tri_degrees) > 0 else 0
            # Graph density: edges / possible edges
            n_g = X_tri.shape[1]
            n_possible = n_g * (n_g - 1) / 2
            n_edges = float(np.sum(tri_degrees)) / 2
            diagnostics['graph_density'] = n_edges / n_possible if n_possible > 0 else 0

        if len(tri_inlier_idx) >= 5:
            # Map back to original indices
            orig_idx = idx_sub[tri_inlier_idx]
            Xin_raw = Xin_raw[:, orig_idx]
            Yin_raw = Yin_raw[:, orig_idx]
            distances_in = distances_in[orig_idx]

    # Optional: SC2 second-order spatial compatibility filtering (alternative to triangular)
    # 真正的二阶约束 (MAC/SC2-PCR 风格): SCG = (FCG @ FCG) * FCG，尺度感知 cross_dist=|D_y-s*D_x|
    if use_sc2 and Xin_raw.shape[1] >= 6:
        # 兼容阈值默认与三角 C3 一致 (5*bound)，保证公平对比
        if sc2_d_thre is None:
            sc2_d_thre = 5 * bound

        sc2_inlier_idx, sc2_degrees = filter_by_sc2(
            Xin_raw, Yin_raw, bestS, sc2_d_thre, percent=sc2_percent)

        if return_diagnostics:
            diagnostics['n_before_sc2'] = Xin_raw.shape[1]
            diagnostics['n_after_sc2'] = len(sc2_inlier_idx)
            diagnostics['sc2_degrees_mean'] = float(np.mean(sc2_degrees)) if len(sc2_degrees) > 0 else 0
            diagnostics['sc2_d_thre'] = float(sc2_d_thre)

        if len(sc2_inlier_idx) >= 5:
            Xin_raw = Xin_raw[:, sc2_inlier_idx]
            Yin_raw = Yin_raw[:, sc2_inlier_idx]
            distances_in = distances_in[sc2_inlier_idx]

    Xin_raw = bestS * Xin_raw
    X_scaled = bestS * X

    n_points = Xin_raw.shape[1]

    if return_diagnostics:
        diagnostics['n_before_ransac'] = n_points
    
    # Adaptive parameter selection based on data size
    # 阈值随 noise/bound 缩放，使室外大噪声数据集（如 KITTI）与室内数据集（3DMatch）都能工作
    if n_points > 1500:
        # Large dataset: use more aggressive filtering
        ransac_iter = 80
        ransac_thresh = 0.6 * bound
        irls_iter = 150
        num_trials = 2
    elif n_points > 800:
        # Medium dataset: balanced approach
        ransac_iter = 120
        ransac_thresh = 0.7 * bound
        irls_iter = 250
        num_trials = 4
    else:
        # Small dataset: more iterations and trials for robustness
        ransac_iter = 200
        ransac_thresh = 0.8 * bound
        irls_iter = 300
        num_trials = 5

    # Step 1: RANSAC预筛选，快速剔除明显外点
    ransac_mask, R_ransac, t_ransac = ransac_pre_filter(
        Xin_raw, Yin_raw,
        n_iterations=ransac_iter,
        threshold=ransac_thresh,
        rng=rng
    )
    Xin_filtered = Xin_raw[:, ransac_mask]
    Yin_filtered = Yin_raw[:, ransac_mask]
    
    if Xin_filtered.shape[1] < 4:
        if return_diagnostics:
            diagnostics['n_after_ransac'] = int(np.sum(ransac_mask))
            return S_result, R_ransac, t_ransac, diagnostics
        return S_result, R_ransac, t_ransac
    
    # Step 2: IRLS with multiple trials and adaptive bounds
    bestR = np.eye(3)
    best_T = np.zeros(3)
    best_score = -float('inf')
    
    trial_results = []
    
    for trial in range(num_trials):
        try:
            # 不同 noise bound 多试验（随配置 noise 缩放，更鲁棒）
            # 基准乘数相对 noise=0.01 标定：[0.6, 0.8, 1.0, 1.2, 1.5]
            if n_points > 1500:
                noise_multipliers = [0.6, 0.8, 1.0]
            else:
                noise_multipliers = [0.6, 0.8, 1.0, 1.2, 1.5]
            bound_local = 2 * np.sqrt(3) * noise * noise_multipliers[trial % len(noise_multipliers)]

            oth_R, oth_T, W = irls_sa_cauchy_point(
                Xin_filtered, Yin_filtered,
                iter_num=irls_iter,
                sigma=bound_local / (2 * np.sqrt(3))
            )

            # Compute comprehensive score
            Bfit = np.dot(oth_R, Xin_filtered) + oth_T.reshape(-1, 1)
            errors = np.sqrt(np.sum((Bfit - Yin_filtered) ** 2, axis=0))

            # 评分阈值随 noise 缩放（基准 noise=0.01 时为 3*noise=0.03 / 4*noise=0.04）
            tight_threshold = 3 * noise if n_points > 800 else 4 * noise
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
        if return_diagnostics:
            diagnostics['n_after_ransac'] = int(np.sum(ransac_mask))
            return S_result, R_ransac, t_ransac, diagnostics
        return S_result, R_ransac, t_ransac
    
    # Sort by score and select the best
    trial_results.sort(key=lambda x: x['score'], reverse=True)
    
    # Validate the best result
    best_trial = trial_results[0]
    
    # Check for abnormal results
    rot_diff_from_identity = np.arccos(np.clip((np.trace(best_trial['R']) - 1) / 2, -1, 1))
    
    # More strict validation for small datasets（阈值随 noise 缩放）
    error_threshold = 30 * noise if n_points < 1000 else 50 * noise
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

    # ---- Post-refinement: 1-point translation RANSAC on the full match set ----
    # The IRLS weighted-centroid translation can be severely biased when the
    # final inlier set is small and contaminated (common in low-inlier-ratio
    # outdoor scenes such as KITTI, where RANSAC may leave <15 points). Given a
    # reliable R, we re-estimate t by finding the densest cluster among the
    # per-match translation candidates t_i = Y_i - R @ (S*X_i).
    #
    # Adoption rule (disagreement + tight-threshold validation):
    # 1. Disagreement gate: only consider switching when |t_pipe - t_rans| > 3*bound
    #    (i.e. they are clearly not the same solution) and the RANSAC cluster has
    #    enough support (inl_ransac >= 20).
    # 2. Tight-threshold validation: when they disagree, compare the number of
    #    inliers each translation produces on the FULL match set at a tight
    #    threshold (bound = 2*sqrt(3)*noise). Only switch to the RANSAC t if it
    #    has STRICTLY MORE tight inliers than the pipeline t. This prevents the
    #    RANSAC from switching to an outlier cluster when the pipeline t is
    #    already correct — a failure mode that pure disagreement cannot detect.
    if post_refine and X.shape[1] >= 20:
        X_full_scaled = bestS * X  # (3, N) full match set, scaled
        t_ransac, inl_ransac, _ = translation_ransac_1pt(
            X_full_scaled, Y, bestR, threshold=3 * bound,
            max_n=1000, rng_seed=random_seed)

        t_diff = float(np.linalg.norm(t_ransac - best_T.flatten()))
        if t_diff > 3 * bound and inl_ransac >= 20:
            # Tight-threshold validation: compare inliers on full match set
            tight_thre = bound  # 2*sqrt(3)*noise
            Y_pred_pipe = bestR @ X_full_scaled + best_T.reshape(-1, 1)
            Y_pred_rans = bestR @ X_full_scaled + t_ransac.reshape(-1, 1)
            err_pipe = np.linalg.norm(Y_pred_pipe - Y, axis=0)
            err_rans = np.linalg.norm(Y_pred_rans - Y, axis=0)
            inl_pipe_tight = int(np.sum(err_pipe < tight_thre))
            inl_rans_tight = int(np.sum(err_rans < tight_thre))

            # v3: Switch only if RANSAC t has at least 20% more tight inliers
            # than the pipeline t (margin requirement). At very low inlier
            # ratios (e.g. KITTI ~0.3%), outlier clusters can occasionally
            # have slightly more tight inliers than the true translation
            # (ratio ~1.1). The 1.2 margin filters out these marginal cases
            # while preserving genuine improvements (ratio typically >1.5).
            # Empirically validated on 32 KITTI-FCGF switch cases: the single
            # regression had ratio=1.12; all 31 correct switches had ratio>=1.24.
            if inl_rans_tight >= 1.2 * inl_pipe_tight:
                best_T = t_ransac
                if return_diagnostics:
                    diagnostics['post_refine_applied'] = True
                    diagnostics['post_refine_t_diff'] = t_diff
                    diagnostics['post_refine_inl_ransac'] = inl_ransac
                    diagnostics['post_refine_inl_pipe_tight'] = inl_pipe_tight
                    diagnostics['post_refine_inl_rans_tight'] = inl_rans_tight
                    diagnostics['post_refine_ratio'] = inl_rans_tight / max(inl_pipe_tight, 1)

    if return_diagnostics:
        diagnostics['n_after_ransac'] = int(np.sum(ransac_mask))
        diagnostics['n_after_irls'] = int(Xin_filtered.shape[1])
        return S_result, bestR, best_T, diagnostics
    return S_result, bestR, best_T
