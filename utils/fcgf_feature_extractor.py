"""
FCGF特征提取模块
用于替代FPFH在跨源点云配准中的特征提取

论文：Choy et al., "Fully Convolutional Geometric Features", ICCV 2019
代码：https://github.com/chrischoy/FCGF
"""

import numpy as np
import torch
import open3d as o3d


class FCGFFeatureExtractor:
    """
    FCGF特征提取器
    支持GPU加速，可复用模型实例
    """
    
    def __init__(self, 
                 model_path=None, 
                 voxel_size=0.05,
                 device='cuda',
                 use_pretrained=True):
        """
        初始化FCGF特征提取器
        
        Args:
            model_path: 预训练模型路径，如果为None则使用默认路径
            voxel_size: 体素大小
            device: 'cuda' 或 'cpu'
            use_pretrained: 是否使用预训练模型
        """
        self.voxel_size = voxel_size
        self.device = device
        self.model = None
        self.model_path = model_path
        
        # 延迟加载模型（第一次使用时才加载）
        self._model_loaded = False
        
    def _load_model(self):
        """
        延迟加载FCGF模型
        """
        if self._model_loaded:
            return
            
        try:
            import MinkowskiEngine as ME
            from fcgf.model import FCGFNet
        except ImportError as e:
            raise ImportError(
                "FCGF需要安装MinkowskiEngine。请运行以下命令安装：\n"
                "pip install MinkowskiEngine\n"
                "或参考：https://github.com/NVIDIA/MinkowskiEngine"
            )
        
        print("正在加载FCGF模型...")
        
        # 创建模型
        self.model = FCGFNet(
            conv1_kernel_size=5,
            conv1_out_channels=32,
            conv2_out_channels=64,
            conv3_out_channels=128,
            conv4_out_channels=256,
            conv5_out_channels=512,
            conv6_out_channels=1024,
            conv7_out_channels=2048,
            conv8_out_channels=4096,
            bn_momentum=0.05,
            normalize_feature=True
        )
        
        # 加载预训练权重
        if self.model_path is None:
            # 使用默认预训练模型路径
            self.model_path = self._get_default_model_path()
        
        if self.model_path and use_pretrained:
            try:
                checkpoint = torch.load(self.model_path, map_location=self.device)
                if 'state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
                print(f"成功加载预训练模型：{self.model_path}")
            except Exception as e:
                print(f"警告：无法加载预训练模型 {self.model_path}")
                print(f"错误信息：{e}")
                print("将使用随机初始化的模型")
        
        self.model = self.model.to(self.device)
        self.model.eval()
        self._model_loaded = True
        
    def _get_default_model_path(self):
        """
        获取默认预训练模型路径
        """
        import os
        
        # 尝试多个可能的路径
        possible_paths = [
            './weights/FCGF_3DMatch.pth',
            './FCGF_3DMatch.pth',
            '../weights/FCGF_3DMatch.pth',
            os.path.expanduser('~/.cache/fcgf/FCGF_3DMatch.pth'),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        print("警告：未找到预训练模型，请手动下载：")
        print("下载链接：https://github.com/chrischoy/FCGF#pre-trained-weights")
        return None
    
    def extract_features(self, points):
        """
        提取FCGF特征
        
        Args:
            points: 点云坐标，形状为 (N, 3) 或 (3, N)
            
        Returns:
            features: FCGF特征，形状为 (M, 32)，M是下采样后的点数
            coords: 下采样后的点坐标，形状为 (M, 3)
        """
        # 确保模型已加载
        self._load_model()
        
        # 转换点云格式
        if isinstance(points, np.ndarray):
            if points.shape[0] == 3 and points.shape[1] > points.shape[0]:
                # (3, N) -> (N, 3)
                points = points.T
            points_tensor = torch.from_numpy(points).float()
        elif isinstance(points, torch.Tensor):
            if points.shape[0] == 3 and points.shape[1] > points.shape[0]:
                points = points.T
            points_tensor = points.float()
        else:
            raise TypeError(f"不支持的点云类型：{type(points)}")
        
        points_tensor = points_tensor.to(self.device)
        
        # 体素化
        import MinkowskiEngine as ME
        
        # 计算体素坐标
        quantized_coords = torch.floor(points_tensor / self.voxel_size)
        
        # 去重并保留唯一坐标
        unique_coords, inverse_indices = torch.unique(
            quantized_coords, dim=0, return_inverse=True
        )
        
        # 创建稀疏张量
        coords = unique_coords.int()
        feats = torch.ones((coords.shape[0], 1), device=self.device)
        
        # 转换为MinkowskiEngine格式
        coords = torch.cat([
            torch.zeros((coords.shape[0], 1), device=self.device),  # batch index
            coords
        ], dim=1)
        
        # 创建稀疏张量
        input_sparse = ME.SparseTensor(
            features=feats,
            coordinates=coords,
            device=self.device
        )
        
        # 提取特征
        with torch.no_grad():
            output = self.model(input_sparse)
            fcgf_features = output.features
        
        # 获取对应的点坐标
        downsampled_points = unique_coords * self.voxel_size
        
        # 转换为numpy
        fcgf_features = fcgf_features.cpu().numpy()
        downsampled_points = downsampled_points.cpu().numpy()
        
        return fcgf_features, downsampled_points
    
    def extract_features_open3d(self, pcd):
        """
        从Open3D点云提取FCGF特征
        
        Args:
            pcd: Open3D点云对象
            
        Returns:
            features: FCGF特征
            coords: 下采样后的点坐标
        """
        points = np.asarray(pcd.points)
        return self.extract_features(points)


def fcgf_feature_matching(feat0, feat1, ratio_threshold=0.9):
    """
    使用FCGF特征进行特征匹配
    
    Args:
        feat0: 源点云特征，形状为 (N0, D)
        feat1: 目标点云特征，形状为 (N1, D)
        ratio_threshold: 比例测试阈值
        
    Returns:
        matches: 匹配对，形状为 (M, 2)，每行是 (src_idx, tgt_idx)
        distances: 匹配距离
    """
    import torch
    import torch.nn.functional as F
    
    # 转换为张量
    if isinstance(feat0, np.ndarray):
        feat0 = torch.from_numpy(feat0).float()
    if isinstance(feat1, np.ndarray):
        feat1 = torch.from_numpy(feat1).float()
    
    # 归一化特征
    feat0 = F.normalize(feat0, dim=1)
    feat1 = F.normalize(feat1, dim=1)
    
    # 计算相似度矩阵
    similarity = torch.mm(feat0, feat1.t())  # (N0, N1)
    
    # 找到每个点的最佳匹配和次佳匹配
    top2_sim, top2_idx = torch.topk(similarity, k=2, dim=1)
    
    # 比例测试
    ratio = top2_sim[:, 0] / (top2_sim[:, 1] + 1e-8)
    valid_mask = ratio > ratio_threshold
    
    # 构建匹配对
    src_indices = torch.where(valid_mask)[0]
    tgt_indices = top2_idx[valid_mask, 0]
    
    matches = torch.stack([src_indices, tgt_indices], dim=1)
    distances = 1 - top2_sim[valid_mask, 0]
    
    return matches.numpy(), distances.numpy()


def create_fcgf_extractor(voxel_size=0.05, device='cuda'):
    """
    创建FCGF特征提取器的便捷函数
    
    Args:
        voxel_size: 体素大小
        device: 'cuda' 或 'cpu'
        
    Returns:
        FCGFFeatureExtractor实例
    """
    return FCGFFeatureExtractor(voxel_size=voxel_size, device=device)


# 兼容性函数：与现有FPFH接口保持一致
def compute_fcgf_feature(pcd, voxel_size=0.05, device='cuda'):
    """
    计算FCGF特征（兼容Open3D的FPFH接口）
    
    Args:
        pcd: Open3D点云
        voxel_size: 体素大小
        device: 'cuda' 或 'cpu'
        
    Returns:
        pcd_down: 下采样后的点云
        fcgf_feature: FCGF特征对象（模拟Open3D的特征格式）
    """
    extractor = FCGFFeatureExtractor(voxel_size=voxel_size, device=device)
    features, coords = extractor.extract_features_open3d(pcd)
    
    # 创建下采样后的点云
    pcd_down = o3d.geometry.PointCloud()
    pcd_down.points = o3d.utility.Vector3dVector(coords)
    
    # 创建特征对象（模拟Open3D的FPFH特征格式）
    class FCGFFeature:
        def __init__(self, data):
            self.data = data
    
    fcgf_feature = FCGFFeature(features.T)  # 转置以匹配Open3D格式
    
    return pcd_down, fcgf_feature


if __name__ == "__main__":
    # 测试代码
    print("FCGF特征提取模块测试")
    
    # 创建测试点云
    import open3d as o3d
    
    # 生成随机点云
    np.random.seed(42)
    points = np.random.randn(1000, 3) * 0.5
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # 测试特征提取
    try:
        extractor = FCGFFeatureExtractor(voxel_size=0.05)
        features, coords = extractor.extract_features_open3d(pcd)
        print(f"原始点数：{len(points)}")
        print(f"下采样后点数：{len(coords)}")
        print(f"特征维度：{features.shape}")
        print("FCGF特征提取成功！")
    except Exception as e:
        print(f"FCGF特征提取失败：{e}")
        print("请确保已安装MinkowskiEngine和FCGF")
