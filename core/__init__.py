# -*- coding: utf-8 -*-
"""
Robust Point Cloud Registration - Core Algorithm Modules

This package implements scale-aware point cloud registration using
line vector graphs and triangular consistency constraints.

Core API:
    get_final_cor      — Top-level registration orchestrator
    reg_with_scale     — Full registration pipeline with optional GPU support
    scale_vector       — Line-vector scale ratio computation
    scale_estimation   — RANSAC-based scale estimation
    irls_sa_cauchy_point — IRLS with SA-Cauchy for R, t estimation

Triangular Consistency:
    build_triangular_consistency_graph — Graph construction
    filter_by_triangular_consistency   — Degree-based inlier filtering
    compute_vertex_degrees             — Vertex degree computation
    select_anchor_by_degree            — Anchor selection

GPU Acceleration (optional, requires PyTorch + CUDA):
    gpu_accel.scale_vector_gpu
    gpu_accel.filter_by_triangular_consistency_gpu
    gpu_accel.line_vectors_gpu
    gpu_accel.compute_vertex_degrees_gpu
    gpu_accel.lv_map_to_pt_gpu
"""

from .line_vectors import line_vectors
from .scale_vector import scale_vector
from .scale_estimation import scale_estimation
from .sa_cauchy_point import sa_cauchy_point
from .irls_sa_cauchy_point import irls_sa_cauchy_point
from .lv_map_to_pt import lv_map_to_pt
from .build_pt_estimate import build_pt_estimate
from .get_inlier_sum import get_inlier_sum
from .reg_with_scale import reg_with_scale
from .reg_with_scale_custom import reg_with_scale_custom
from .get_final_cor import get_final_cor
from .get_final_cor_custom import get_final_cor_custom
from .triangular_consistency import (
    compute_vertex_degrees,
    select_anchor_by_degree,
    build_triangular_consistency_graph,
    filter_by_triangular_consistency,
)

# GPU acceleration module (optional — only importable when torch is installed)
try:
    from . import gpu_accel
    from .gpu_accel import (
        scale_vector_gpu,
        filter_by_triangular_consistency_gpu,
        line_vectors_gpu,
        compute_vertex_degrees_gpu,
        lv_map_to_pt_gpu,
    )
    _HAS_GPU = True
except ImportError:
    _HAS_GPU = False

# SC2 compatibility module (for ablation comparison)
try:
    from .sc2_compatibility import filter_by_sc2
except ImportError:
    pass

# Translation RANSAC (post-refinement)
try:
    from .translation_ransac import translation_ransac_1pt
except ImportError:
    pass
