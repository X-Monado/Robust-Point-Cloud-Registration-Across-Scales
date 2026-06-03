"""
集成FCGF特征到现有测试流程
支持在3DCSR数据集上使用FCGF进行跨源点云配准
"""

import os
import sys
import numpy as np
import open3d as o3d
import time
from datetime import datetime

# 添加路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入现有模块
import estimate_scale_LM
import estimate_R

# 导入FCGF特征提取器
try:
    from fcgf_feature_extractor import FCGFFeatureExtractor, fcgf_feature_matching
    FCGF_AVAILABLE = True
except ImportError:
    print("警告：无法导入FCGF模块，将使用FPFH")
    FCGF_AVAILABLE = False


class CrossSourceRegistration:
    """
    跨源点云配准类
    支持FPFH和FCGF特征
    """
    
    def __init__(self, 
                 feature_type='fcgf',
                 voxel_size=0.05,
                 device='cuda',
                 use_fcgf=True):
        """
        初始化配准器
        
        Args:
            feature_type: 'fpfh' 或 'fcgf'
            voxel_size: 体素大小
            device: 'cuda' 或 'cpu'
            use_fcgf: 是否使用FCGF（如果可用）
        """
        self.feature_type = feature_type
        self.voxel_size = voxel_size
        self.device = device
        
        # 初始化FCGF提取器
        if use_fcgf and FCGF_AVAILABLE and feature_type == 'fcgf':
            self.fcgf_extractor = FCGFFeatureExtractor(
                voxel_size=voxel_size,
                device=device
            )
            self.use_fcgf = True
            print("使用FCGF特征")
        else:
            self.fcgf_extractor = None
            self.use_fcgf = False
            print("使用FPFH特征")
    
    def extract_features(self, pcd):
        """
        提取特征（FPFH或FCGF）
        
        Args:
            pcd: Open3D点云
            
        Returns:
            pcd_down: 下采样点云
            features: 特征
            coords: 点坐标
        """
        if self.use_fcgf:
            # 使用FCGF
            features, coords = self.fcgf_extractor.extract_features_open3d(pcd)
            
            # 创建下采样点云
            pcd_down = o3d.geometry.PointCloud()
            pcd_down.points = o3d.utility.Vector3dVector(coords)
            
            return pcd_down, features, coords
        else:
            # 使用FPFH
            pcd_down = pcd.voxel_down_sample(self.voxel_size)
            
            # 估计法向量
            pcd_down.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(
                    radius=self.voxel_size * 2,
                    max_nn=30
                )
            )
            
            # 计算FPFH
            fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                pcd_down,
                o3d.geometry.KDTreeSearchParamHybrid(
                    radius=self.voxel_size * 5,
                    max_nn=100
                )
            )
            
            coords = np.asarray(pcd_down.points)
            features = fpfh.data.T  # (N, 33)
            
            return pcd_down, features, coords
    
    def feature_matching(self, feat0, feat1, ratio_threshold=0.9):
        """
        特征匹配
        
        Args:
            feat0: 源特征
            feat1: 目标特征
            ratio_threshold: 比例测试阈值
            
        Returns:
            matches: 匹配对
            distances: 匹配距离
        """
        if self.use_fcgf:
            return fcgf_feature_matching(feat0, feat1, ratio_threshold)
        else:
            # FPFH匹配
            from scipy.spatial.distance import cdist
            
            # 计算距离矩阵
            distances = cdist(feat0, feat1, metric='euclidean')
            
            # 找到最近邻
            min_dist0 = np.min(distances, axis=1)
            min_idx0 = np.argmin(distances, axis=1)
            
            # 比例测试
            sorted_dist = np.sort(distances, axis=1)
            ratio = sorted_dist[:, 0] / (sorted_dist[:, 1] + 1e-8)
            
            valid_mask = ratio < (1.0 / ratio_threshold)
            
            src_indices = np.where(valid_mask)[0]
            tgt_indices = min_idx0[valid_mask]
            
            matches = np.stack([src_indices, tgt_indices], axis=1)
            match_distances = min_dist0[valid_mask]
            
            return matches, match_distances
    
    def register(self, 
                 pcd0, 
                 pcd1, 
                 gt_trans=None,
                 verbose=True):
        """
        执行配准
        
        Args:
            pcd0: 源点云
            pcd1: 目标点云
            gt_trans: 真值变换矩阵（可选）
            verbose: 是否打印详细信息
            
        Returns:
            result: 配准结果字典
        """
        start_time = time.time()
        
        # 1. 特征提取
        if verbose:
            print("步骤1：特征提取...")
        
        t0 = time.time()
        pcd0_down, feat0, coords0 = self.extract_features(pcd0)
        pcd1_down, feat1, coords1 = self.extract_features(pcd1)
        feature_time = time.time() - t0
        
        if verbose:
            print(f"  源点云：{len(coords0)} 点")
            print(f"  目标点云：{len(coords1)} 点")
            print(f"  特征提取时间：{feature_time:.2f}s")
        
        # 2. 特征匹配
        if verbose:
            print("步骤2：特征匹配...")
        
        t0 = time.time()
        matches, distances = self.feature_matching(feat0, feat1)
        matching_time = time.time() - t0
        
        if verbose:
            print(f"  匹配数量：{len(matches)}")
            print(f"  匹配时间：{matching_time:.2f}s")
        
        # 3. 尺度估计
        if verbose:
            print("步骤3：尺度估计...")
        
        t0 = time.time()
        
        # 获取匹配点对
        src_points = coords0[matches[:, 0]]
        tgt_points = coords1[matches[:, 1]]
        
        # 使用线向量估计尺度
        scale, scale_inliers = estimate_scale_LM.estimate_scale_with_inliers(
            src_points.T,
            tgt_points.T,
            pcd0,
            pcd1,
            bound=0.05
        )
        
        scale_time = time.time() - t0
        
        if verbose:
            print(f"  估计尺度：{scale:.4f}")
            print(f"  尺度估计时间：{scale_time:.2f}s")
        
        # 4. 图构建与筛选
        if verbose:
            print("步骤4：图构建与筛选...")
        
        t0 = time.time()
        
        # 应用尺度
        scaled_src_points = scale * src_points
        
        # 构建图（使用现有的buildGraph逻辑）
        # 这里需要调用你的图构建代码
        # graph, degrees = build_graph(scaled_src_points, tgt_points)
        
        # 度数筛选
        # inlier_mask = degree_screening(degrees, threshold=5)
        
        # 暂时使用简单的距离筛选作为占位
        distances_3d = np.linalg.norm(scaled_src_points - tgt_points, axis=1)
        inlier_mask = distances_3d < np.percentile(distances_3d, 50)
        
        graph_time = time.time() - t0
        
        if verbose:
            print(f"  筛选后匹配数：{np.sum(inlier_mask)}")
            print(f"  图构建时间：{graph_time:.2f}s")
        
        # 5. IRLS精化
        if verbose:
            print("步骤5：IRLS精化...")
        
        t0 = time.time()
        
        # 使用筛选后的点
        final_src = scaled_src_points[inlier_mask]
        final_tgt = tgt_points[inlier_mask]
        
        # 估计R和t
        R, t = estimate_R.estimate_R_t(final_src.T, final_tgt.T)
        
        irls_time = time.time() - t0
        
        if verbose:
            print(f"  IRLS时间：{irls_time:.2f}s")
        
        # 6. 计算误差
        total_time = time.time() - start_time
        
        result = {
            'scale': scale,
            'rotation': R,
            'translation': t,
            'matches': len(matches),
            'inliers': np.sum(inlier_mask),
            'total_time': total_time,
            'feature_time': feature_time,
            'matching_time': matching_time,
            'scale_time': scale_time,
            'graph_time': graph_time,
            'irls_time': irls_time
        }
        
        if gt_trans is not None:
            # 计算误差
            rot_error = self.compute_rotation_error(R, gt_trans[:3, :3])
            trans_error = np.linalg.norm(t - gt_trans[:3, 3])
            scale_error = abs(scale - 1.0)  # 假设真值尺度为1
            
            result['rotation_error'] = rot_error
            result['translation_error'] = trans_error
            result['scale_error'] = scale_error
            result['success'] = rot_error < 5.0 and trans_error < 0.5
            
            if verbose:
                print(f"\n配准结果：")
                print(f"  旋转误差：{rot_error:.2f}°")
                print(f"  平移误差：{trans_error:.4f}m")
                print(f"  尺度误差：{scale_error:.4f}")
                print(f"  总时间：{total_time:.2f}s")
        
        return result
    
    @staticmethod
    def compute_rotation_error(R_est, R_gt):
        """
        计算旋转误差（度）
        """
        import math
        R_error = np.dot(R_gt.T, R_est)
        trace = np.trace(R_error)
        trace = np.clip(trace, -1.0, 3.0)
        angle = math.acos((trace - 1) / 2)
        return math.degrees(abs(angle))


def test_on_3dcsr_data(data_path, feature_type='fcgf'):
    """
    在3DCSR数据集上测试
    
    Args:
        data_path: 数据路径
        feature_type: 'fpfh' 或 'fcgf'
    """
    print(f"在3DCSR数据集上测试（特征：{feature_type}）")
    print("=" * 60)
    
    # 创建配准器
    registrar = CrossSourceRegistration(
        feature_type=feature_type,
        voxel_size=0.05
    )
    
    # TODO: 加载3DCSR数据
    # 这里需要根据你的数据格式进行修改
    
    # 示例：假设数据格式
    # pcd0 = o3d.io.read_point_cloud(os.path.join(data_path, 'lidar.ply'))
    # pcd1 = o3d.io.read_point_cloud(os.path.join(data_path, 'kinect.ply'))
    # gt_trans = np.loadtxt(os.path.join(data_path, 'gt_trans.txt'))
    
    # result = registrar.register(pcd0, pcd1, gt_trans)
    
    print("测试完成")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='跨源点云配准测试')
    parser.add_argument('--data_path', type=str, default='./data/3dcsr',
                        help='数据路径')
    parser.add_argument('--feature', type=str, default='fcgf',
                        choices=['fpfh', 'fcgf'],
                        help='特征类型')
    
    args = parser.parse_args()
    
    test_on_3dcsr_data(args.data_path, args.feature)
