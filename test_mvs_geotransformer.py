import os
import numpy as np
import math
import csv
import time
import sys

sys.path.append("/home/zjy24/DeepLearning/TEASER-plusplus-master/examples")
sys.path.append("/home/zjy24/DeepLearning/TEASER-plusplus-master/examples/code-doublegraph-exe-test-KITTI-ransac")

from get_final_cor import get_final_cor


def get_angular_error(R_gt, R_est):
    """
    Get angular error in degrees
    """
    try:
        A = (np.trace(np.dot(R_gt.T, R_est)) - 1) / 2.0
        if A < -1:
            A = -1
        if A > 1:
            A = 1
        rotError = math.fabs(math.acos(A))
        return math.degrees(rotError)
    except ValueError:
        return 99999


def test_geotransformer_pair(full_data_path, file_id):
    """
    Test a single GeoTransformer pair
    """
    print(f"Loading data from: {full_data_path}")
    data = np.load(full_data_path)
    
    xyz0 = data['xyz0']
    xyz1 = data['xyz1']
    matches = data['matches']
    distances = data['distances']
    gt_trans = data['gt_trans']
    
    print(f"  Source points: {xyz0.shape[0]}")
    print(f"  Target points: {xyz1.shape[0]}")
    print(f"  Matches: {matches.shape[0]}")
    
    if matches.shape[0] < 3:
        print("  WARNING: Not enough matches, skipping...")
        return None, None, None, None
    
    orgin_point = xyz0[matches[:, 0]].T
    target_point = xyz1[matches[:, 1]].T
    
    scale = 1.0
    
    buildGraphTime, S, best_R_np, best_T_np = get_final_cor(
        orgin_point, target_point, distances, scale
    )
    
    a = gt_trans
    
    SA_R_ERROR = get_angular_error(best_R_np, a[0:3, 0:3])
    trans_error = np.linalg.norm(a[0:3, 3] - best_T_np.flatten())
    
    print(f"  Scale: {S:.4f}")
    print(f"  Rotation Error: {SA_R_ERROR:.4f} degrees")
    print(f"  Translation Error: {trans_error:.4f}")
    print(f"  Time: {buildGraphTime:.4f}s")
    
    return S, SA_R_ERROR, trans_error, buildGraphTime


def main():
    MATCHES_DIR = '/home/zjy24/DeepLearning/GeoTransformer/mvs_results'
    
    all_files = sorted(
        [os.path.join(MATCHES_DIR, f) for f in os.listdir(MATCHES_DIR) if f.endswith('.npz')],
        key=lambda x: int(os.path.basename(x).replace('scan', '').replace('.npz', ''))
    )
    
    print(f"Found {len(all_files)} pairs to test\n")
    
    if not os.path.exists("log"):
        os.makedirs("log")
    
    result_file = "log/result_mvs_geotransformer.csv"
    file_exists = os.path.exists(result_file)
    
    with open(result_file, "a", encoding="utf-8", newline="") as f:
        csv_writer = csv.writer(f)
        if not file_exists:
            csv_writer.writerow(['File', 'Scale', 'R_error', 'T_error', 'Time'])
    
    S_list = []
    R_error_list = []
    T_error_list = []
    time_list = []
    success_count = 0
    
    for i, full_data_path in enumerate(all_files):
        file_id = os.path.basename(full_data_path).replace('.npz', '')
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(all_files)}] Testing: {file_id}")
        print('='*60)
        
        try:
            S, R_error, T_error, time_val = test_geotransformer_pair(full_data_path, file_id)
            
            if S is not None:
                S_list.append(S)
                R_error_list.append(R_error)
                T_error_list.append(T_error)
                time_list.append(time_val)
                success_count += 1
                
                with open(result_file, mode='a+', newline='', encoding='utf-8') as outfile:
                    writer = csv.writer(outfile)
                    writer.writerow([file_id + '.npz', f"{S:.6f}", f"{R_error:.4f}", f"{T_error:.4f}", f"{time_val:.4f}"])
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)
    print(f"Total pairs: {len(all_files)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {len(all_files) - success_count}")
    
    if success_count > 0:
        print(f"\n--- Statistics ---")
        print(f"Mean Rotation Error: {np.mean(R_error_list):.4f} degrees")
        print(f"Median Rotation Error: {np.median(R_error_list):.4f} degrees")
        print(f"Mean Translation Error: {np.mean(T_error_list):.4f}")
        print(f"Median Translation Error: {np.median(T_error_list):.4f}")
        print(f"Mean Time: {np.mean(time_list):.4f}s")
        
        rot_success = sum(1 for r in R_error_list if r < 5)
        trans_success = sum(1 for t in T_error_list if t < 0.2)
        both_success = sum(1 for r, t in zip(R_error_list, T_error_list) if r < 10 and t < 0.3)
        print(f"\nRotation Error < 5 deg: {rot_success}/{success_count} ({100*rot_success/success_count:.1f}%)")
        print(f"Translation Error < 0.2: {trans_success}/{success_count} ({100*trans_success/success_count:.1f}%)")
        print(f"Success (R<10° & T<0.3m): {both_success}/{success_count} ({100*both_success/success_count:.1f}%)")
    
    with open(result_file, mode='a+', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        writer.writerow([''])
        writer.writerow(['SUMMARY'])
        writer.writerow(['Mean R_error', f"{np.mean(R_error_list):.4f}" if R_error_list else 'N/A'])
        writer.writerow(['Mean T_error', f"{np.mean(T_error_list):.4f}" if T_error_list else 'N/A'])
        writer.writerow(['Mean Time', f"{np.mean(time_list):.4f}" if time_list else 'N/A'])


if __name__ == '__main__':
    main()
