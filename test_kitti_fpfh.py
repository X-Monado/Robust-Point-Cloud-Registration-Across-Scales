
import os
import numpy as np
import math
import csv
from get_final_cor import get_final_cor

def get_angular_error(R_gt, R_est):
    try:
        A = (np.trace(np.dot(R_gt.T, R_est))-1) / 2.0
        if A < -1:
            A = -1
        if A > 1:
            A = 1
        
        rotError = math.fabs(math.acos(A));
        return math.degrees(rotError)
    except ValueError:
        return 99999

def test_file(full_data_path):
    data = np.load(full_data_path)
    
    xyz0 = data['xyz0']
    xyz1 = data['xyz1']
    matches = data['matches']
    distances = data['distances']
    gt_trans = data['gt_trans']
    
    if len(distances.shape) == 2 and distances.shape[1] == 2:
        distances = distances[:, 1]
    
    orgin_point = xyz0[matches[:,0]].T
    target_point = xyz1[matches[:,1]].T
    
    scale = 1.0
    
    buildGraphTime, S, best_R_np, best_T_np = get_final_cor(orgin_point, target_point, distances, scale)
    
    SA_R_ERROR = get_angular_error(best_R_np, gt_trans[0:3,0:3])
    trans_error = np.linalg.norm(gt_trans[0:3,3] - best_T_np.flatten())
    
    return S, SA_R_ERROR, trans_error, buildGraphTime

def main():
    MATCHES_DIR = '/home/zjy24/DeepLearning/TEASER-plusplus-master/examples/fpfh_test'
    all_npz_files = [os.path.join(MATCHES_DIR, f) for f in sorted(os.listdir(MATCHES_DIR)) if f.endswith('.npz')]
    
    matlab_results = {}
    matlab_result_file = 'log/result_fcgf.csv'
    if os.path.exists(matlab_result_file):
        with open(matlab_result_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                matlab_results[row['Num']] = {
                    'R_error': float(row['R_error']),
                    'T_error': float(row['T_error'])
                }
    
    print(f"Testing KITTI FPFH dataset...")
    print("-" * 100)
    print(f"{'Filename':<20} {'Python R':<12} {'MATLAB R':<12} {'Python T':<12} {'MATLAB T':<12}")
    print("-" * 100)
    
    test_count = min(20, len(all_npz_files))
    python_R_errors = []
    python_T_errors = []
    
    for i in range(test_count):
        full_data_path = all_npz_files[i]
        file_id = os.path.basename(full_data_path)
        
        try:
            S, R_error, T_error, time_val = test_file(full_data_path)
            
            python_R_errors.append(R_error)
            python_T_errors.append(T_error)
            
            matlab_R = matlab_results.get(file_id, {}).get('R_error', float('nan'))
            matlab_T = matlab_results.get(file_id, {}).get('T_error', float('nan'))
            
            print(f"{file_id:<20} {R_error:<12.4f} {matlab_R:<12.4f} {T_error:<12.6f} {matlab_T:<12.6f}")
        except Exception as e:
            print(f"Error testing {file_id}: {e}")
            import traceback
            traceback.print_exc()
    
    print("-" * 100)
    if python_R_errors:
        print(f"\nPython results - Avg R_error: {np.mean(python_R_errors):.4f}deg, Avg T_error: {np.mean(python_T_errors):.6f}m")
        print(f"Python results - Median R_error: {np.median(python_R_errors):.4f}deg, Median T_error: {np.median(python_T_errors):.6f}m")

if __name__ == '__main__':
    main()

