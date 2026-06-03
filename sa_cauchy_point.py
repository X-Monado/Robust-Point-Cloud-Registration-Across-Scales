import numpy as np


def sa_cauchy_point(A, B, u, w):
    """
    Simulated annealing with Cauchy distribution for point cloud registration
    
    Parameters:
    -----------
    A : numpy.ndarray
        Source point cloud, shape (3, N)
    B : numpy.ndarray
        Target point cloud, shape (3, N)
    u : float
        Scale parameter for Cauchy distribution
    w : numpy.ndarray
        Initial weights, shape (N,)
    
    Returns:
    --------
    R : numpy.ndarray
        Estimated rotation matrix, shape (3, 3)
    t : numpy.ndarray
        Estimated translation vector, shape (3,)
    W : numpy.ndarray
        Updated weights, shape (3, N)
    """
    sw = np.sum(w)
    if sw == 0:
        w = np.ones(A.shape[1])
        sw = np.sum(w)
    
    w = w / sw
    lc = np.dot(A, w.T)
    rc = np.dot(B, w.T)
    
    w2 = np.sqrt(w)
    
    # Weighted coordinates
    left = A - lc.reshape(-1, 1)
    left = left * w2
    right = B - rc.reshape(-1, 1)
    right = right * w2
    
    M = np.dot(right, left.T)
    
    # Extract components
    Sxx, Syx, Szx, Sxy, Syy, Szy, Sxz, Syz, Szz = M.flatten()
    
    # Build N matrix
    N = np.array([
        [Sxx + Syy + Szz, Syz - Szy, Szx - Sxz, Sxy - Syx],
        [Syz - Szy, Sxx - Syy - Szz, Sxy + Syx, Szx + Sxz],
        [Szx - Sxz, Sxy + Syx, -Sxx + Syy - Szz, Syz + Szy],
        [Sxy - Syx, Szx + Sxz, Syz + Szy, -Sxx - Syy + Szz]
    ])
    
    # Eigen decomposition
    D, V = np.linalg.eig(N)
    D = np.real(D)
    V = np.real(V)
    
    emax = np.argmax(D)
    q = V[:, emax]
    
    # Sign ambiguity
    ii = np.argmax(np.abs(q))
    sgn = np.sign(q[ii])
    q = q * sgn
    
    # Quaternion to rotation matrix
    quat = q / np.linalg.norm(q)
    q0 = quat[0]
    qx = quat[1]
    qy = quat[2]
    qz = quat[3]
    v = quat[1:4]
    
    # Build Z matrix
    Z = np.array([
        [q0, -qz, qy],
        [qz, q0, -qx],
        [-qy, qx, q0]
    ])
    
    R = np.outer(v, v) + np.dot(Z, Z)
    
    # Estimate translation
    center_A = np.dot(A, w.T) / np.sum(w)
    center_B = np.dot(B, w.T) / np.sum(w)
    t = center_B - np.dot(R, center_A)
    
    # Compute error and update weights
    Bfit = np.dot(R, A) + t.reshape(-1, 1)
    E = np.sqrt(np.sum((Bfit - B) ** 2, axis=0))
    
    w_new = u / (u + E ** 2)
    W = np.vstack([w_new, w_new, w_new])
    
    return R, t, W
