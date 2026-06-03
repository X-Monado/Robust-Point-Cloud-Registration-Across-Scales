"""
Test DS-PCR on FCGF pre-matched pairs from 3DCSR dataset (v3)

This script tests the registration algorithm on FCGF features
that have already been extracted and matched.
"""

import os
import sys
import numpy as np
import math
import csv
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from different_scale_pcr import register_with_scale


def get_angular_error(R_gt, R_est):
    """
    Compute angular error between two rotation matrices
    """
    try:
        A = (np.trace(np.dot(R_gt.T, R_est)) - 1) / 2.0
        A = max(-1, min(1, A))
        return math.degrees(math.acos(A))
    except:
        return 99999


def test_fcgf_pair(pair_file, noise_bound=0.001):
    """
    Test a single FCGF pair
    """
    data = np.load(pair_file)
    
    xyz0 = data['xyz0']
    xyz1 = data['xyz1']
    matches = data['matches']
    distances = data['distances']
    gt_trans = data['gt_trans']
    
    print(f"  xyz0: {xyz0.shape}, xyz1: {xyz1.shape}")
    print(f"  matches: {matches.shape}, distances: {distances.shape}")
    
    if len(matches) < 3:
        print(f"  Too few matches ({len(matches)}), skipping...")
        return None
    
    X = xyz0[matches[:, 0]].T
    Y = xyz1[matches[:, 1]].T
    
    print(f"  Correspondences: {X.shape[1]} points")
    
    try:
        time_elapsed, scale, R, t = register_with_scale(
            X, Y, distances,
            scale=1.0,
            estimate_scale=False,
            noise_bound=noise_bound
        )
        
        R_error = get_angular_error(R, gt_trans[0:3, 0:3])
        t_error = np.linalg.norm(gt_trans[0:3, 3] - t.flatten())
        
        print(f"  Scale: {scale:.4f}")
        print(f"  R_error: {R_error:.4f} degrees")
        print(f"  T_error: {t_error:.4f}")
        print(f"  Time: {time_elapsed:.4f}s")
        
        return {
            'pair': os.path.basename(pair_file),
            'num_matches': len(matches),
            'scale': scale,
            'R_error': R_error,
            'T_error': t_error,
            'time': time_elapsed
        }
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    data_dir = "/home/zjy24/DeepLearning/FCGF-master/outputs/3dcsr_pairs_v3"
    
    pair_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.npz')])
    
    print(f"Found {len(pair_files)} FCGF pairs")
    print("=" * 60)
    
    results = []
    
    for i, pair_file in enumerate(pair_files):
        print(f"\n[{i+1}/{len(pair_files)}] Testing {pair_file}")
        print("-" * 60)
        
        pair_path = os.path.join(data_dir, pair_file)
        result = test_fcgf_pair(pair_path, noise_bound=0.001)
        
        if result is not None:
            results.append(result)
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if len(results) > 0:
        R_errors = [r['R_error'] for r in results]
        T_errors = [r['T_error'] for r in results]
        times = [r['time'] for r in results]
        num_matches = [r['num_matches'] for r in results]
        
        print(f"Total pairs tested: {len(results)}")
        print(f"Average R_error: {np.mean(R_errors):.4f} degrees")
        print(f"Average T_error: {np.mean(T_errors):.4f}")
        print(f"Average time: {np.mean(times):.4f}s")
        print(f"Average matches: {np.mean(num_matches):.1f}")
        print(f"Min matches: {np.min(num_matches)}, Max matches: {np.max(num_matches)}")
        
        R_errors_success = [r for r in R_errors if r < 30]
        T_errors_success = [t for t in T_errors if t < 0.5]
        
        print(f"\nSuccess rate (R_error < 30 deg): {len(R_errors_success)}/{len(results)} = {100*len(R_errors_success)/len(results):.1f}%")
        print(f"Success rate (T_error < 0.5): {len(T_errors_success)}/{len(results)} = {100*len(T_errors_success)/len(results):.1f}%")
        
        result_file = "log/result_fcgf_3dcsr_v3.csv"
        os.makedirs("log", exist_ok=True)
        
        with open(result_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(['pair', 'num_matches', 'scale', 'R_error', 'T_error', 'time'])
            for r in results:
                writer.writerow([r['pair'], r['num_matches'], r['scale'], r['R_error'], r['T_error'], r['time']])
        
        print(f"\nResults saved to {result_file}")
    else:
        print("No successful results!")


if __name__ == '__main__':
    main()
