import numpy as np
from scipy.spatial import KDTree
import open3d as o3d
import math


orgin_pcd =o3d.io.read_point_cloud(r'C:\Users\peog\Desktop\test\ptCloud_x_2.pcd')
target_pcd =o3d.io.read_point_cloud(r'C:\Users\peog\Desktop\test\ptCloud_y_2.pcd')

#T1 = icp_corr.test_icp(target_pcd,orgin_pcd)
#print('T1:',T1)
import numpy as np
from scipy.spatial.transform import Rotation as R

# source = np.asarray(orgin_pcd.points)
# target = np.asarray(target_pcd.points)
# data = np.load(r'D:\fpfh_test\pair_19.npz')
# source = source + data['gt_trans'][0:3,3]#np.array([-9.16446204,-0.01698902,-0.10969446])
# orgin_pcd.points = o3d.utility.Vector3dVector(source)
# orgin_pcd.paint_uniform_color([0, 0, 1])
# target_pcd.paint_uniform_color([1, 0, 0])
# o3d.visualization.draw_geometries([orgin_pcd,target_pcd])
# T = icp_ransac(source, target, max_iterations=100, distance_threshold=0.15)
# print('T:',T)

import numpy as np

import numpy as np
def get_angular_error(R_gt, R_est):
    """
    Get angular error
    """
    try:
        A = (np.trace(np.dot(R_gt.T, R_est))-1) / 2.0
        print("A",A)
        if A < -1:
            A = -1
        if A > 1:
            A = 1
        
        rotError = math.fabs(math.acos(A));
        return math.degrees(rotError)
    except ValueError:
        import pdb; pdb.set_trace()
        return 99999

def huber_loss(x, delta):
    """
    Huber loss function.
    """
    x = np.abs(x)
    mask = x <= delta
    inlier_term = 0.5 * x**2
    outlier_term = delta * (x - 0.5 * delta)
    return np.where(mask, inlier_term, outlier_term)

def irls_rotation_estimation(A, B, max_iterations=100, threshold=1e-8):
    """
    IRLS algorithm for rotation estimation.
    :param A: (n, 3) numpy array, source point cloud
    :param B: (n, 3) numpy array, target point cloud
    :param max_iterations: maximum number of iterations
    :param threshold: threshold for convergence
    :return: (3, 3) numpy array, rotation matrix
    """
    # Initialize weights as 1/n
    n = A.shape[0]
    w = np.ones(n) / n
    
    u_max = np.max(A)
    u_min = np.min(A)
    u = (u_max - u_min)**2
    # Iterate until convergence or maximum iterations reached
    for i in range(max_iterations):
        # Solve weighted SVD problem
        W = np.diag(np.sqrt(w))
        C = np.dot(W, B)
        At = np.dot(W, A).T
        U, _, Vt = np.linalg.svd(np.dot(At, C))
        R = np.dot(Vt.T, np.dot(np.diag([1, 1, np.sign(np.linalg.det(np.dot(Vt.T, np.dot(At, C))))]), U.T))
        
        
        # centroid_A = np.mean(A, axis=0)
        # centroid_B = np.mean(B, axis=0)
        # Compute weighted centroids
        centroid_A = np.sum(A * w[:, None], axis=0) / np.sum(w)
        centroid_B = np.sum(B * w[:, None], axis=0) / np.sum(w)

        t = centroid_B - np.dot(centroid_A, R.T)
        # Calculate residuals and update weights
        # residuals = np.linalg.norm(np.dot(A, R.T) - B, axis=1)

        residuals = np.dot(A, R.T) + t - B
        normalized_residuals = np.linalg.norm(residuals, axis=1) / np.median(np.linalg.norm(residuals, axis=1))
        # normalized_residuals = huber_loss(normalized_residuals, 10)
        w= u/(u+(normalized_residuals**2))
        
        # w= 1-normalized_residuals**2/u
        # w = (1-normalized_residuals**2/(u+normalized_residuals**2))**2
        # w = 1-(1-normalized_residuals**2)**3
        
        # w = 1 / (normalized_residuals**2 + u+1e-5)
        # w /= np.sum(w)
        # Check convergence
        if np.max(np.abs(w - 1 / n)) < threshold:
            break
        if u>(3*0.01)**2:
            u = u/1.2
        # if i > max_iterations-10:
        #     w[w<0.7]=0
    
    return R,t

# R = irls_rotation_estimation(source,target)
# print('R:',R)
# error_R = get_angular_error(data['gt_trans'][0:3,0:3],R)
# print('error_R:',error_R)