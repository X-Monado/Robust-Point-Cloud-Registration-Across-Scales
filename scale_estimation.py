import numpy as np


def scale_estimation(Sxy, Snoise, interval, flag, bound, scale):
    """
    Estimate scale from scale vectors
    
    Parameters:
    -----------
    Sxy : numpy.ndarray
        Scale ratios, shape (M,)
    Snoise : numpy.ndarray
        Noise scales, shape (M,)
    interval : int
        Interval for subsampling
    flag : int
        If 1, estimate scale; if 0, use given scale
    bound : float
        Bound for noise
    scale : float
        Given scale (only used if flag == 0)
    
    Returns:
    --------
    bestS : float
        Estimated scale
    inliers : numpy.ndarray
        Indices of inliers
    bound : float
        Updated bound
    """
    # Local optimization parameters
    local_iter = 1
    local_scale = 5
    bound = 1
    
    # Subsample
    Sxy_sub = Sxy[::interval]
    Snoise_sub = Snoise[::interval]
    all_point = np.ones(len(Snoise_sub), dtype=bool)
    Snoise_t = Snoise
    
    s_list = np.zeros(40)
    ninliers_list = np.zeros(40)
    
    if flag == 1:
        iter_num = 0
        p = 0.99
        Niter = 10**5
        bestscore = 0
        bestS = 1.0
        
        while iter_num < Niter and np.sum(all_point) > 0:
            # Pick a random sample
            all_point_f = np.where(all_point)[0]
            all_size_f = len(all_point_f)
            
            random_ind = 0
            id = all_point_f[random_ind]
            all_point[id] = False
            
            s = Sxy_sub[id]
            
            # Find inliers for this scale
            res = Snoise_sub - np.abs(Sxy_sub - s)
            inliers = np.where(res > 0)[0]
            ninliers = len(inliers)
            
            all_point[inliers] = False
            
            if ninliers > ninliers_list[0]:
                # Refine scale using weighted least squares
                Sxy_in = Sxy_sub[inliers]
                Snoise_bin2 = (Snoise_sub[inliers] / 5) ** 2
                s_f = np.sum(Sxy_in / Snoise_bin2) / np.sum(1 / Snoise_bin2)
                
                # Check refined scale
                res_f = Snoise_sub - np.abs(Sxy_sub - s_f)
                inliers_f = np.where(res_f > 0)[0]
                ninliers_f = len(inliers_f)
                
                if ninliers_f >= ninliers:
                    ninliers_list[0] = ninliers_f
                    s_list[0] = s_f
                    
                    # Sort lists
                    sort_ind = np.argsort(ninliers_list)
                    ninliers_list = ninliers_list[sort_ind]
                    s_list = s_list[sort_ind]
                    
                    bestscore = ninliers_f
                    bestS = s_f
                else:
                    bestscore = ninliers
                    bestS = s
                    ninliers_list[0] = ninliers
                    s_list[0] = s
                    
                    sort_ind = np.argsort(ninliers_list)
                    ninliers_list = ninliers_list[sort_ind]
                    s_list = s_list[sort_ind]
            
            # Update number of iterations
            fracinliers = bestscore / len(Sxy_sub)
            pNoOutliers = 1 - fracinliers
            pNoOutliers = max(np.finfo(float).eps, pNoOutliers)
            pNoOutliers = min(1 - np.finfo(float).eps, pNoOutliers)
            Niter = np.log(1 - p) / np.log(pNoOutliers)
            Niter = max(Niter, 1000)
            
            iter_num += 1
            all_point_f = np.where(all_point)[0]
            all_size_f = len(all_point_f)
        
        # Select best scale from candidates
        inliers_max = 0
        noise = (0.05 / 0.01)
        
        for sum_s in range(len(s_list)):
            if s_list[sum_s] == 0:
                continue
            res = Snoise_t / noise - np.abs(Sxy - s_list[sum_s])
            inliers = np.where(res > 0)[0]
            ninliers = len(inliers)
            
            if inliers_max < ninliers:
                bestS = s_list[sum_s]
                inliers_max = ninliers
        
        # Refine best scale
        res = Snoise_t / noise - np.abs(Sxy - bestS)
        inliers = np.where(res > 0)[0]
        Sxy_in = Sxy[inliers]
        Snoise_bin2 = (Snoise_t[inliers] / noise) ** 2
        bestS = np.sum(Sxy_in / Snoise_bin2) / np.sum(1 / Snoise_bin2)
        
        # Local optimization
        s_list = np.zeros(50)
        ninliers_list = np.zeros(50)
        
        for bound_ind in range(local_iter):
            Sxy_sub = Sxy[inliers]
            bound = bound * local_scale
            Snoise_sub = Snoise[inliers] / bound
            Snoise = Snoise / bound
            all_point = np.ones(len(Snoise_sub), dtype=bool)
            
            iter_num = 0
            p = 0.99
            Niter = 10**5
            bestscore = 0
            
            while iter_num < Niter and np.sum(all_point) > 0:
                all_point_f = np.where(all_point)[0]
                all_size_f = len(all_point_f)
                
                random_ind = 0
                id = all_point_f[random_ind]
                all_point[id] = False
                
                s = Sxy_sub[id]
                
                res = Snoise_sub - np.abs(Sxy_sub - s)
                inliers_local = np.where(res > 0)[0]
                ninliers = len(inliers_local)
                
                all_point[inliers_local] = False
                
                if ninliers > ninliers_list[0]:
                    Sxy_in = Sxy_sub[inliers_local]
                    Snoise_bin2 = Snoise_sub[inliers_local] ** 2
                    s_f = np.sum(Sxy_in / Snoise_bin2) / np.sum(1 / Snoise_bin2)
                    
                    res_f = Snoise_sub - np.abs(Sxy_sub - s_f)
                    inliers_f = np.where(res_f > 0)[0]
                    ninliers_f = len(inliers_f)
                    
                    if ninliers_f >= ninliers:
                        bestscore = ninliers_f
                        ninliers_list[0] = ninliers_f
                        s_list[0] = s_f
                        
                        sort_ind = np.argsort(ninliers_list)
                        ninliers_list = ninliers_list[sort_ind]
                        s_list = s_list[sort_ind]
                        
                        bestS = s_f
                    else:
                        bestscore = ninliers
                        bestS = s
                        ninliers_list[0] = ninliers
                        s_list[0] = s
                        
                        sort_ind = np.argsort(ninliers_list)
                        ninliers_list = ninliers_list[sort_ind]
                        s_list = s_list[sort_ind]
                
                fracinliers = bestscore / len(Sxy_sub)
                pNoOutliers = 1 - fracinliers
                pNoOutliers = max(np.finfo(float).eps, pNoOutliers)
                pNoOutliers = min(1 - np.finfo(float).eps, pNoOutliers)
                Niter = np.log(1 - p) / np.log(pNoOutliers)
                Niter = max(Niter, 1000)
                
                iter_num += 1
                all_point_f = np.where(all_point)[0]
                all_size_f = len(all_point_f)
            
            # Select best scale from candidates
            inliers_max = 0
            noise = (0.05 / 0.01)
            
            for sum_s in range(len(s_list)):
                if s_list[sum_s] == 0:
                    continue
                res = Snoise_t / noise - np.abs(Sxy - s_list[sum_s])
                inliers_local = np.where(res > 0)[0]
                ninliers = len(inliers_local)
                
                if inliers_max < ninliers:
                    bestS = s_list[sum_s]
                    inliers_max = ninliers
        
        # Final refinement
        bound = bound / (local_scale ** (local_iter * 2))
        res = Snoise - np.abs(Sxy - bestS)
        inliers = np.where(res > 0)[0]
        Sxy_in = Sxy[inliers]
        Snoise_bin2 = Snoise[inliers] ** 2
        bestS = np.sum(Sxy_in / Snoise_bin2) / np.sum(1 / Snoise_bin2)
    
    else:
        bestS = scale
        print('scale estimate final start')
        res = Snoise * 4 - np.abs(Sxy - bestS)
        print('scale estimate final')
        inliers = np.where(res > 0)[0]
    
    return bestS, inliers, bound
