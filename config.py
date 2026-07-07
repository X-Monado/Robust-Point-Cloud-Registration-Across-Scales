# -*- coding: utf-8 -*-
"""
Centralized configuration for dataset paths and experiment parameters.

Usage:
    from config import DATASETS, get_dataset_path

    # Get path for a specific dataset
    path = get_dataset_path('kitti_fcgf')

    # Or access the full config dict
    cfg = DATASETS['kitti_fcgf']
    print(cfg['path'], cfg['re_th'], cfg['te_th'])

Path Resolution:
    By default, data is expected under ./data/<dataset_name>/.
    Set the DATA_ROOT environment variable to use a custom location:
        export DATA_ROOT=/path/to/your/data
"""

import os

# ============================================================
# Data root: env var > default ./data
# ============================================================
DATA_ROOT = os.environ.get('DATA_ROOT', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data'))

# ============================================================
# Dataset configurations
#   path         : directory containing .npz correspondence files
#   recursive    : whether to search subdirectories for .npz files
#   re_th / te_th: success thresholds (Rotation Error ° / Translation Error m)
#   noise        : noise bound (meters) for the registration algorithm
#   scale        : initial scale factor (1.0 for same-scale, estimated for cross-scale)
#   feature      : feature extractor name
#   type         : dataset description
# ============================================================
DATASETS = {
    # ---- KITTI (outdoor LiDAR, same-source) ----
    'kitti_fpfh': {
        'path': os.path.join(DATA_ROOT, 'kitti_fpfh'),
        'recursive': False, 're_th': 10, 'te_th': 0.3,
        'noise': 0.2, 'scale': 1.0,
        'feature': 'FPFH', 'type': 'KITTI (outdoor same-source)',
    },
    'kitti_fcgf': {
        'path': os.path.join(DATA_ROOT, 'kitti_fcgf'),
        'recursive': False, 're_th': 10, 'te_th': 0.3,
        'noise': 0.1, 'scale': 1.0,
        'feature': 'FCGF', 'type': 'KITTI (outdoor same-source)',
    },
    # ---- 3DMatch (indoor RGB-D, same-source) ----
    '3dmatch_fpfh': {
        'path': os.path.join(DATA_ROOT, '3dmatch_fpfh'),
        'recursive': False, 're_th': 10, 'te_th': 0.3,
        'noise': 0.01, 'scale': 1.0,
        'feature': 'FPFH', 'type': '3DMatch (indoor same-source)',
    },
    '3dmatch_fcgf': {
        'path': os.path.join(DATA_ROOT, '3dmatch_fcgf'),
        'recursive': False, 're_th': 10, 'te_th': 0.3,
        'noise': 0.01, 'scale': 1.0,
        'feature': 'FCGF', 'type': '3DMatch (indoor same-source)',
    },
    # ---- MVS (outdoor multi-view, cross-source) ----
    'mvs_fcgf': {
        'path': os.path.join(DATA_ROOT, 'mvs_fcgf'),
        'recursive': False, 're_th': 15, 'te_th': 0.3,
        'noise': 0.01, 'scale': 1.0,
        'feature': 'FCGF', 'type': 'MVS (outdoor cross-source)',
    },
    'mvs_geotrans': {
        'path': os.path.join(DATA_ROOT, 'mvs_geotrans'),
        'recursive': False, 're_th': 15, 'te_th': 0.3,
        'noise': 0.01, 'scale': 1.0,
        'feature': 'GeoTrans', 'type': 'MVS (outdoor cross-source)',
    },
    # ---- 3DCSR (indoor Kinect-SFM, cross-source) ----
    '3dcsr_fcgf': {
        'path': os.path.join(DATA_ROOT, '3dcsr_fcgf'),
        'recursive': False, 're_th': 30, 'te_th': 0.5,
        'noise': 0.01, 'scale': 1.0,
        'feature': 'FCGF', 'type': '3DCSR (indoor cross-source)',
    },
    '3dcsr_geotrans': {
        'path': os.path.join(DATA_ROOT, '3dcsr_geotrans'),
        'recursive': True, 're_th': 30, 'te_th': 0.5,
        'noise': 0.01, 'scale': 1.0,
        'feature': 'GeoTrans', 'type': '3DCSR (indoor cross-source)',
    },
}


def get_dataset_path(name):
    """Return the filesystem path for a dataset by name."""
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Available: {list(DATASETS.keys())}")
    return DATASETS[name]['path']


def get_dataset_config(name):
    """Return the full config dict for a dataset."""
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Available: {list(DATASETS.keys())}")
    return DATASETS[name]


def list_datasets():
    """Print all available datasets and their paths."""
    print(f"DATA_ROOT = {DATA_ROOT}\n")
    for name, cfg in DATASETS.items():
        exists = '✓' if os.path.exists(cfg['path']) else '✗'
        print(f"  {exists} {name:20s} → {cfg['path']}")
        print(f"    {cfg['type']} | {cfg['feature']} | RE<{cfg['re_th']}° TE<{cfg['te_th']}m")


if __name__ == '__main__':
    list_datasets()
