# Robust-Point-Cloud-Registration-Across-Scales

**Point Cloud Registration with Scale Estimation** — A robust point cloud registration framework based on one-point RANSAC and scale-annealing biweight estimation, supporting cross-source and cross-scale scenarios.

---

## Features

- **Scale Estimation**: Robust scale estimation via line-vector ratio consensus with adaptive noise bounds
- **Rotation Estimation**: IRLS (Iteratively Reweighted Least Squares) with simulated annealing and Cauchy kernel
- **Translation Estimation**: Weighted centroid-based translation with iterative refinement
- **Cross-Source Support**: Works with heterogeneous point clouds (LiDAR-SFM, Kinect-SFM, etc.)
- **Cross-Scale Support**: Handles point cloud pairs with unknown relative scale differences
- **Multiple Feature Backends**: Supports both FPFH (Open3D) and FCGF (learned features)
- **RANSAC Filtering**: Built-in RANSAC-based outlier filtering for correspondence pruning

---

## Algorithm Pipeline

```
Input Point Clouds (Source, Target) + Feature Correspondences
        │
        ▼
┌─────────────────────┐
│  Scale Vector        │  Compute line-vector ratios between
│  Computation         │  all point pairs → scale candidates
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Scale Estimation    │  One-point RANSAC + weighted least
│  (scale_estimation)  │  squares refinement + local opt.
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Inlier Mapping      │  Map scale inliers back to point
│  (lv_map_to_pt)      │  correspondences, select top-%
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Point Estimate      │  Build keypoint-based point
│  Construction        │  difference structure
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  IRLS + SA Cauchy    │  Iteratively Reweighted Least Squares
│  Rotation/Translation│  with Simulated Annealing + Cauchy
└─────────┬───────────┘  kernel for robust R, t estimation
          │
          ▼
Output: Estimated Scale (s), Rotation (R), Translation (t)
```

---



## Installation

### Prerequisites

- Python 3.7+
- CUDA 10.2+ 
- CMake 3.10+ 

### Dependencies

```bash
pip install numpy open3d matplotlib
```

### FCGF Feature Support (Optional)

If you want to use FCGF learned features instead of FPFH:

```bash
# Install MinkowskiEngine
pip install MinkowskiEngine

# Install FCGF
git clone https://github.com/chrischoy/FCGF.git
cd FCGF && pip install -r requirements.txt && python setup.py develop

# Download pretrained weights
mkdir -p weights
wget https://github.com/chrischoy/FCGF/raw/master/weights/FCGF_3DMatch.pth -O weights/FCGF_3DMatch.pth
```

---

## Quick Start

### 1. Basic Registration on 3DCSR Dataset

```bash
python teaser_start_3dcsr_correct.py
```


### 2. Scale Estimation Only

```python
from scale_vector import scale_vector
from scale_estimation import scale_estimation
import numpy as np

# X, Y: source/target point clouds, shape (3, N)
noise_val = 0.05
bound = 0.05 * np.sqrt(3) * noise_val
lbound = 5 * bound

Sxy, Snoise, X_lv, Y_lv, map_arr = scale_vector(X, Y, bound, lbound)
bestS, inliers, _ = scale_estimation(Sxy, Snoise, interval=2, flag=1, bound=bound, scale=1.0)

print(f"Estimated scale: {bestS:.4f}")
print(f"Inlier count: {len(inliers)}")
```

### 3. Full Registration Pipeline

```python
from get_final_cor_custom import get_final_cor_custom
import numpy as np

# X, Y: source/target correspondences, shape (3, N)
# distances: feature distances, shape (N,)
scale = 1.0
noise_val = 0.01  # Adjust based on your data (0.001 for 3DCSR, 0.01 for KITTI)

buildGraphTime, bestS, bestR, bestT = get_final_cor_custom(
    X, Y, distances, scale, noise_val
)

print(f"Scale: {bestS:.4f}")
print(f"Rotation:\n{bestR}")
print(f"Translation: {bestT}")
```


---

## Datasets

### Supported Datasets

| Dataset | Type | Scale Variation | Feature |
|---------|------|----------------|---------|
| **3DCSR**  | Cross-source | Yes | FPFH / FCGF |
| **KITTI** | LiDAR sequential | Minimal | FPFH / FCGF |
| **3DMatch** | RGB-D | No | FPFH / FCGF |
| **MVS** | Multi-view | Controlled | FCGF |

### Data Format

For NPZ format datasets, each file should contain:

```python
{
    'xyz0': source_points,      # (N0, 3)
    'xyz1': target_points,      # (N1, 3)
    'matches': match_indices,   # (M, 2) int
    'distances': feat_distances,# (M,) float
    'gt_trans': gt_transform,   # (4, 4)
    'relative_scale': scale,    # (1,) float
}
```


## Acknowledgments

- [TEASER++](https://github.com/MIT-SPARK/TEASER-plusplus) — Truncated least squares Estimation And SEmidefinite Relaxation
- [FCGF](https://github.com/chrischoy/FCGF) — Fully Convolutional Geometric Features
- [Open3D](http://www.open3d.org/) — 3D data processing library
- [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine) — Sparse tensor library for FCGF

---

#

