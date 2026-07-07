# Data Directory

This directory contains correspondence data for point cloud registration experiments.
Each subdirectory corresponds to one dataset–feature combination.

## Directory Structure

```
data/
├── kitti_fpfh/        # KITTI + FPFH features (555 pairs)
├── kitti_fcgf/        # KITTI + FCGF features (555 pairs)
├── 3dmatch_fpfh/      # 3DMatch + FPFH features (1623 pairs)
├── 3dmatch_fcgf/      # 3DMatch + FCGF features (1623 pairs)
├── mvs_fcgf/          # MVS + FCGF features (64 pairs)
├── mvs_geotrans/      # MVS + GeoTransformer features (64 pairs)
├── 3dcsr_fcgf/        # 3DCSR + FCGF features (28 pairs)
└── 3dcsr_geotrans/    # 3DCSR + GeoTransformer features (32 pairs)
```

## Data Format (NPZ)

Each `.npz` file contains:

| Key          | Shape       | Description                              |
|--------------|-------------|------------------------------------------|
| `xyz0`       | (N0, 3)     | Source point cloud coordinates           |
| `xyz1`       | (N1, 3)     | Target point cloud coordinates           |
| `matches`    | (M, 2) int  | Correspondence indices: [source, target] |
| `distances`  | (M,) or (M, 2) | Feature matching distances            |
| `gt_trans`   | (4, 4)      | Ground truth transformation matrix       |

## Setup

### Option 1: Symlink (recommended for local development)

```bash
cd data
ln -s /path/to/your/kitti_fpfh_data kitti_fpfh
ln -s /path/to/your/3dmatch_fcgf_data 3dmatch_fcgf
# ... etc
```

### Option 2: Set DATA_ROOT environment variable

```bash
export DATA_ROOT=/path/to/your/data_root
# Inside $DATA_ROOT, create the same subdirectory structure:
#   $DATA_ROOT/kitti_fpfh/
#   $DATA_ROOT/kitti_fcgf/
#   ...
```

### Option 3: Copy files directly

```bash
cp /path/to/your/*.npz data/kitti_fpfh/
```

## Dataset Sources

| Dataset  | Source                                                                  | Type         |
|----------|-------------------------------------------------------------------------|--------------|
| KITTI    | [KITTI Odometry](http://www.cvlibs.net/datasets/kitti/odometry/)       | Outdoor LiDAR|
| 3DMatch  | [3DMatch Benchmark](http://3dmatch.cs.princeton.edu/)                  | Indoor RGB-D |
| MVS      | Cross-source MVS dataset (SFM vs. Kinect)                              | Cross-source  |
| 3DCSR    | [3DCSR Dataset](https://github.com/minhai-corner/3DCSR) (kinect_sfm)   | Cross-source  |

## Feature Extraction

- **FPFH**: Open3D `compute_fpfh_feature()`, radius=0.5m
- **FCGF**: [FCGF repository](https://github.com/chrischoy/FCGF), pretrained model
- **GeoTransformer**: [GeoTransformer repository](https://github.com/qinzheng93/GeoTransformer), pretrained model


