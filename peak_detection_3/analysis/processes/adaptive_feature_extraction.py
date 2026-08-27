import numpy as np
import pandas as pd
import os
import re
import itertools
import time
from datetime import datetime
from scipy.signal import medfilt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Create timing results directory
results_dir = 'results'
timing_dir = os.path.join(results_dir, 'time')
os.makedirs(timing_dir, exist_ok=True)

# Global timing data collectors
file_timing_data = []
parameter_combination_timing_data = []
overall_timing_data = []

# CPU Detection Only
CPU_PARALLEL_AVAILABLE = False
try:
    from joblib import Parallel, delayed
    import multiprocessing
    CPU_PARALLEL_AVAILABLE = True
    print(f" CPU: {multiprocessing.cpu_count()} cores available")
except:
    print(" Joblib not available - using sequential processing")

from log_processor import extract_network_capacity
from peak_speed_detect import peak_speed_detect

def save_timing_data():
    """Save all timing data to CSV files"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save file-level timing data
    if file_timing_data:
        file_timing_df = pd.DataFrame(file_timing_data)
        file_timing_path = os.path.join(timing_dir, f'param_test_file_timing_{timestamp}.csv')
        file_timing_df.to_csv(file_timing_path, index=False)
        print(f"\nFile timing data saved to: {file_timing_path}")
        
        # Also save latest file timing
        latest_file_path = os.path.join(timing_dir, 'latest_param_test_file_timing.csv')
        file_timing_df.to_csv(latest_file_path, index=False)
    
    # Save parameter combination timing data (if collected)
    if parameter_combination_timing_data:
        param_timing_df = pd.DataFrame(parameter_combination_timing_data)
        param_timing_path = os.path.join(timing_dir, f'param_combination_timing_{timestamp}.csv')
        param_timing_df.to_csv(param_timing_path, index=False)
        print(f"Parameter combination timing data saved to: {param_timing_path}")
        
        # Also save latest param timing
        latest_param_path = os.path.join(timing_dir, 'latest_param_combination_timing.csv')
        param_timing_df.to_csv(latest_param_path, index=False)
    
    # Save overall timing summary
    if overall_timing_data:
        overall_timing_df = pd.DataFrame(overall_timing_data)
        overall_timing_path = os.path.join(timing_dir, f'param_test_overall_timing_{timestamp}.csv')
        overall_timing_df.to_csv(overall_timing_path, index=False)
        print(f"Overall timing summary saved to: {overall_timing_path}")
        
        # Also save latest overall timing
        latest_overall_path = os.path.join(timing_dir, 'latest_param_test_overall_timing.csv')
        overall_timing_df.to_csv(latest_overall_path, index=False)

def extract_network_limit(subfolder):
    return extract_network_capacity(subfolder)

def calculate_throughput_matlab_method(df, byte_col='tx_bytes', time_col='time'):
    """
    Calculate throughput using MATLAB-compatible method (consistent with Streamlit app and parameter testing)
    MATLAB: 8 * ds(2:end, 5) / ds_interval / 1e6
    
    This method treats byte values as rates per interval, not cumulative counters.
    """
    if byte_col not in df.columns:
        print(f"Warning: {byte_col} not in columns")
        return np.zeros(len(df)), df[time_col].values
    
    time_values = df[time_col].values
    if len(time_values) < 3:
        print(f"Warning: Insufficient time points ({len(time_values)})")
        return np.zeros(len(df)), time_values
    
    # MATLAB: t_ds = ds(2:end, 1) - ds(2, 1)  (relative time from second row)
    relative_time = time_values[1:] - time_values[1]  # Start from second row
    if len(relative_time) < 2:
        print(f"Warning: Cannot calculate interval from {len(relative_time)} points")
        return np.zeros(len(df)), time_values
    
    # MATLAB: ds_interval = mean (t_ds(2:end) - t_ds(1:end-1))
    interval = np.mean(np.diff(relative_time))
    if interval <= 0:
        print(f"Warning: Invalid interval {interval}, using default 0.02")
        interval = 0.02
    
    # MATLAB: 8 * ds(2:end, 5) / ds_interval / 1e6
    byte_values = df[byte_col].values[1:]  # Skip first row like MATLAB
    
    # Check for valid data
    if df[byte_col].nunique() <= 1 or df[byte_col].isna().all():
        print(f"Warning: {byte_col} is constant or all NaN")
        return np.zeros(len(df)), time_values
    
    # Handle NaN values
    if np.any(np.isnan(byte_values)):
        print(f"Warning: {byte_col} contains NaNs. Filling with 0")
        byte_values = np.nan_to_num(byte_values, nan=0.0)
    
    # Calculate Mbps: convert bytes to bits, normalize by interval, convert to Mbps
    bit_rate = 8 * byte_values / interval / 1e6
    
    # Handle negative values (counter resets) by setting to 0
    bit_rate = np.maximum(bit_rate, 0)
    
    # Pad with zero at the beginning to match original dataframe length
    bit_rate_full = np.concatenate([np.zeros(1), bit_rate])
    
    # Ensure no inf/nan values
    bit_rate_full = np.nan_to_num(bit_rate_full, nan=0.0, posinf=0.0, neginf=0.0)
    
    return bit_rate_full, time_values

def test_single_parameter_combination_timed(params, time_vals, throughput, ground_truth, file_info):
    """Wrapper for timing individual parameter combination tests"""
    start_time = time.time()
    
    result = test_single_parameter_combination(params, time_vals, throughput, ground_truth)
    
    end_time = time.time()
    duration = end_time - start_time
    
    # Store parameter combination timing data (only collect sample for analysis)
    if len(parameter_combination_timing_data) < 1000:  # Limit to prevent huge datasets
        timing_record = {
            'file_name': file_info.get('file_name', 'unknown'),
            'short_window': params[0],
            'long_window': params[1], 
            'peak_tolerance': params[2],
            'count_threshold': params[3],
            'duration_seconds': round(duration, 6),
            'status': result['status'],
            'data_points': len(ground_truth),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        parameter_combination_timing_data.append(timing_record)
    
    return result

def test_single_parameter_combination(params, time_vals, throughput, ground_truth):
    short_window, long_window, peak_tolerance, count_threshold = params
    
    if short_window >= long_window:
        return {
            'short_window': short_window, 'long_window': long_window,
            'peak_tolerance': peak_tolerance, 'count_threshold': count_threshold,
            'accuracy': None, 'f1_score': None, 'precision': None, 'recall': None,
            'true_positives': None, 'true_negatives': None, 
            'false_positives': None, 'false_negatives': None,
            'status': 'skipped - short_window >= long_window'
        }
    
    clipping_binary, long_maxima, filtered_tp, peak_ratio, short_counts, short_peaks = peak_speed_detect(
        throughput, time_vals,
        long_window_ms=long_window,
        short_window_ms=short_window,
        peak_tolerance=peak_tolerance,
        count_threshold=count_threshold
    )
    
    binary_predictions = clipping_binary.astype(int)
    min_len = min(len(binary_predictions), len(ground_truth))
    predictions = binary_predictions[:min_len]
    gt = ground_truth[:min_len]
    
    if len(np.unique(predictions)) < 2:
        return {
            'short_window': short_window, 'long_window': long_window,
            'peak_tolerance': peak_tolerance, 'count_threshold': count_threshold,
            'accuracy': None, 'f1_score': None, 'precision': None, 'recall': None,
            'true_positives': None, 'true_negatives': None, 
            'false_positives': None, 'false_negatives': None,
            'status': 'skipped - no discrimination'
        }
    
    accuracy = accuracy_score(gt, predictions)
    f1 = f1_score(gt, predictions, zero_division=0)
    precision = precision_score(gt, predictions, zero_division=0)
    recall = recall_score(gt, predictions, zero_division=0)
    
    tp = np.sum((gt == 1) & (predictions == 1))
    tn = np.sum((gt == 0) & (predictions == 0))
    fp = np.sum((gt == 0) & (predictions == 1))
    fn = np.sum((gt == 1) & (predictions == 0))
    
    return {
        'short_window': short_window, 'long_window': long_window,
        'peak_tolerance': peak_tolerance, 'count_threshold': count_threshold,
        'accuracy': accuracy, 'f1_score': f1, 'precision': precision, 'recall': recall,
        'true_positives': tp, 'true_negatives': tn, 
        'false_positives': fp, 'false_negatives': fn,
        'status': 'success'
    }

def test_parameter_combinations_optimized(time_vals, throughput, ground_truth, 
                                        short_window_range, long_window_range, 
                                        peak_tolerance_range, count_threshold_range,
                                        base_name, subfolder, network_limit, is_upstream, 
                                        param_tests_folder):
    
    # FIXED: Renamed 'time' parameter to 'time_vals' to avoid conflict with time module
    combinations_start_time = time.time()
    combinations_start_datetime = datetime.now()
    
    total_combinations = (len(short_window_range) * len(long_window_range) * 
                         len(peak_tolerance_range) * len(count_threshold_range))
    
    print(f"Testing {total_combinations} parameter combinations for {base_name}")
    print(f"Started at: {combinations_start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    
    param_combinations = list(itertools.product(
        short_window_range, long_window_range,
        peak_tolerance_range, count_threshold_range
    ))
    
    file_info = {'file_name': base_name, 'subfolder': subfolder, 'network_limit': network_limit}
    
    if CPU_PARALLEL_AVAILABLE:
        n_cores = multiprocessing.cpu_count()
        print(f"Using CPU parallelization (all {n_cores} cores)")
        test_records = Parallel(n_jobs=-1, verbose=1, backend='loky')(
            delayed(test_single_parameter_combination_timed)(
                params, time_vals, throughput, ground_truth, file_info
            ) for params in param_combinations
        )
        
    else:
        print(f"Using sequential processing")
        test_records = []
        for i, params in enumerate(param_combinations):
            result = test_single_parameter_combination_timed(params, time_vals, throughput, ground_truth, file_info)
            test_records.append(result)
            if (i + 1) % 100 == 0:
                print(f"  Progress: {i + 1}/{total_combinations}")
    
    # MODIFIED: Calculate timing statistics
    combinations_end_time = time.time()
    combinations_end_datetime = datetime.now()
    combinations_duration = combinations_end_time - combinations_start_time
    combinations_duration_minutes = combinations_duration / 60.0
    
    print(f"Completed parameter combinations for {base_name} at: {combinations_end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration: {combinations_duration_minutes:.2f} minutes ({combinations_duration:.2f} seconds)")
    
    if total_combinations > 0:
        avg_time_per_combination = combinations_duration / total_combinations
        print(f"Average time per combination: {avg_time_per_combination:.4f} seconds")
    else:
        avg_time_per_combination = 0
    
    # MODIFIED: Store file-level timing data
    file_timing_record = {
        'file_name': base_name,
        'subfolder': subfolder,
        'network_limit': network_limit,
        'direction': 'upstream' if is_upstream else 'downstream',
        'total_combinations': total_combinations,
        'successful_combinations': len([r for r in test_records if r.get('status') == 'success']),
        'start_time': combinations_start_datetime.strftime('%Y-%m-%d %H:%M:%S'),
        'end_time': combinations_end_datetime.strftime('%Y-%m-%d %H:%M:%S'),
        'duration_seconds': round(combinations_duration, 2),
        'duration_minutes': round(combinations_duration_minutes, 2),
        'avg_seconds_per_combination': round(avg_time_per_combination, 6),
        'data_points': len(ground_truth),
        'cpu_cores_available': multiprocessing.cpu_count() if CPU_PARALLEL_AVAILABLE else 1,
        'parallel_processing': CPU_PARALLEL_AVAILABLE
    }
    file_timing_data.append(file_timing_record)
    
    if test_records:
        test_records_df = pd.DataFrame(test_records)
        test_records_df['file_name'] = base_name
        test_records_df['direction'] = 'upstream' if is_upstream else 'downstream'
        test_records_df['subfolder'] = subfolder
        test_records_df['network_limit'] = network_limit
        
        rel_path = os.path.join(subfolder)
        param_tests_dir = os.path.join(param_tests_folder, rel_path)
        os.makedirs(param_tests_dir, exist_ok=True)
        
        param_tests_file = os.path.join(param_tests_dir, base_name.replace('.csv', '_param_tests.csv'))
        test_records_df.to_csv(param_tests_file, index=False, float_format='%.6f')
        
        print(f"Saved {len(test_records)} parameter combinations to: {param_tests_file}")
        return param_tests_file
    
    return None

def process_file_for_param_testing(csv_file, param_tests_folder,
                                  short_window_range=[200, 500, 1000], 
                                  long_window_range=[2000, 5000, 10000],
                                  peak_tolerance_range=[0.1, 0.15, 0.2],
                                  count_threshold_range=[5, 8, 10]):
    
    # MODIFIED: Track timing for individual file processing
    file_start_time = time.time()
    file_start_datetime = datetime.now()
    
    print(f"Processing: {csv_file}")
    print(f"Started at: {file_start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    
    df = pd.read_csv(csv_file)
    dir_name = os.path.dirname(csv_file)
    base_name = os.path.basename(csv_file)
    subfolder = os.path.basename(dir_name)
    network_limit = extract_network_limit(subfolder)
    is_upstream = 'upstream' in base_name
    
    print(f"  Loaded {len(df)} rows, direction: {'upstream' if is_upstream else 'downstream'}")
    
    # Find time columns (MATLAB-compatible method expects different time column handling)
    time_cols = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()]
    for col in time_cols:
        df[col] = df[col].astype(float)
        
    # Use time column preference that matches MATLAB method
    if 'time' in df.columns:
        time_col = 'time'
    elif time_cols:
        time_col = time_cols[0]
    else:
        df['index_time'] = df.index.astype(float)
        time_col = 'index_time'
        time_cols.append(time_col)
    
    print(f"  Using time column: {time_col}")
    
    # Calculate throughput using MATLAB-compatible method (same as Streamlit app and feature extraction)
    rx_tp, time_vals = calculate_throughput_matlab_method(df, 'rx_bytes', time_col)
    tx_tp, _ = calculate_throughput_matlab_method(df, 'tx_bytes', time_col)
    primary_tp = rx_tp if is_upstream else tx_tp
    
    print(f"  Calculated throughput: {len(primary_tp)} samples (MATLAB method)")
    print(f"  Throughput range: {primary_tp.min():.2f} - {primary_tp.max():.2f} Mbps")
    
    # Check for negative throughput (should not happen with MATLAB method)
    negative_count = np.sum(primary_tp < 0)
    if negative_count > 0:
        print(f"  Warning: Found {negative_count} negative throughput values (unexpected with MATLAB method)")
        primary_tp = np.maximum(primary_tp, 0)  # Fix any remaining negatives
    
    if len(primary_tp) < 100:
        print(f"  Skipping - insufficient data: {len(primary_tp)} samples")
        
        # MODIFIED: Still record timing for skipped files
        file_end_time = time.time()
        file_duration = file_end_time - file_start_time
        
        skipped_timing = {
            'file_name': base_name,
            'subfolder': subfolder,
            'network_limit': network_limit,
            'direction': 'upstream' if is_upstream else 'downstream',
            'status': 'skipped - insufficient data',
            'start_time': file_start_datetime.strftime('%Y-%m-%d %H:%M:%S'),
            'end_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'duration_seconds': round(file_duration, 2),
            'duration_minutes': round(file_duration / 60.0, 2),
            'data_points': len(primary_tp)
        }
        file_timing_data.append(skipped_timing)
        
        return None
    
    # Check for all-zero throughput
    if np.all(primary_tp == 0):
        print(f"  Warning: All throughput values are zero")
    
    # Get ground truth data (ensure same length as throughput)
    queue_exists = np.zeros(len(primary_tp))
    if 'queue_exists' in df.columns:
        queue_exists_full = df['queue_exists'].values
        min_len = min(len(queue_exists_full), len(primary_tp))
        queue_exists = queue_exists_full[:min_len]
        time_vals = time_vals[:min_len]
        primary_tp = primary_tp[:min_len]
        print(f"  Using queue_exists as ground truth: {np.sum(queue_exists)} positive samples out of {len(queue_exists)}")
    else:
        print(f"  No ground truth available - using zeros")
    
    param_tests_file = test_parameter_combinations_optimized(
        time_vals, primary_tp, queue_exists,
        short_window_range, long_window_range, 
        peak_tolerance_range, count_threshold_range,
        base_name, subfolder, network_limit, is_upstream, 
        param_tests_folder
    )
    
    # MODIFIED: Record final file processing time (includes parameter testing)
    file_end_time = time.time()
    file_end_datetime = datetime.now()
    file_duration = file_end_time - file_start_time
    file_duration_minutes = file_duration / 60.0
    
    print(f"Completed processing {base_name} at: {file_end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total file processing duration: {file_duration_minutes:.2f} minutes ({file_duration:.2f} seconds)")
    
    return param_tests_file

def process_all_for_param_testing(input_folder='extracted_data', 
                                 param_tests_folder='param_tests_data',
                                 short_window_range=[200, 500, 1000], 
                                 long_window_range=[2000, 5000, 10000],
                                 peak_tolerance_range=[0.05, 0.1, 0.15, 0.2],
                                 count_threshold_range=[3, 5, 8, 10, 15]):
    
    # MODIFIED: Track overall processing timing
    overall_start_time = time.time()
    overall_start_datetime = datetime.now()
    
    print(f"Starting overall parameter testing at: {overall_start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    
    os.makedirs(param_tests_folder, exist_ok=True)
    
    csv_files = []
    all_csv_files = []
    
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith('.csv'):
                all_csv_files.append(file)
                if 'downstream' in file.lower():
                    csv_files.append(os.path.join(root, file))
    
    print(f"Found {len(all_csv_files)} total CSV files")
    print(f"Found {len(csv_files)} DOWNSTREAM CSV files to process")
    print(f"🔧 Using MATLAB-compatible throughput calculation for consistency")
    
    if len(csv_files) == 0:
        print("⚠️  No downstream files found!")
        return []
    
    total_combinations = 0
    for short_window in short_window_range:
        for long_window in long_window_range:
            if short_window < long_window:
                total_combinations += len(peak_tolerance_range) * len(count_threshold_range)
    
    print(f"Testing {total_combinations} parameter combinations per DOWNSTREAM file")
    print(f"Total tests across all DOWNSTREAM files: {total_combinations * len(csv_files)}")
    
    output_files = []
    successful = 0
    failed = 0
    
    for i, csv_file in enumerate(csv_files):
        print(f"\n[{i+1}/{len(csv_files)}] Processing DOWNSTREAM file: {os.path.basename(csv_file)}")
        result = process_file_for_param_testing(
            csv_file, 
            param_tests_folder,
            short_window_range=short_window_range,
            long_window_range=long_window_range,
            peak_tolerance_range=peak_tolerance_range,
            count_threshold_range=count_threshold_range
        )
        
        if result:
            output_files.append(result)
            successful += 1
        else:
            failed += 1
    
    # MODIFIED: Calculate overall timing statistics
    overall_end_time = time.time()
    overall_end_datetime = datetime.now()
    overall_duration = overall_end_time - overall_start_time
    overall_duration_minutes = overall_duration / 60.0
    overall_duration_hours = overall_duration / 3600.0
    
    print(f"\nCompleted overall parameter testing at: {overall_end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total processing duration: {overall_duration_hours:.2f} hours ({overall_duration_minutes:.2f} minutes, {overall_duration:.2f} seconds)")
    
    # Calculate statistics
    total_parameter_tests = total_combinations * successful
    avg_time_per_file = overall_duration / len(csv_files) if len(csv_files) > 0 else 0
    avg_time_per_param_test = overall_duration / total_parameter_tests if total_parameter_tests > 0 else 0
    
    print(f"\nDOWNSTREAM parameter testing complete!")
    print(f"  Successful: {successful} files")
    print(f"  Failed: {failed} files")
    print(f"  Total output files: {len(output_files)}")
    print(f"  Total parameter tests: {total_parameter_tests}")
    print(f"  Average time per file: {avg_time_per_file / 60.0:.2f} minutes")
    print(f"  Average time per parameter test: {avg_time_per_param_test:.4f} seconds")
    
    # MODIFIED: Store overall timing summary
    overall_timing_record = {
        'process_name': 'parameter_testing_all_files',
        'start_time': overall_start_datetime.strftime('%Y-%m-%d %H:%M:%S'),
        'end_time': overall_end_datetime.strftime('%Y-%m-%d %H:%M:%S'),
        'duration_seconds': round(overall_duration, 2),
        'duration_minutes': round(overall_duration_minutes, 2),
        'duration_hours': round(overall_duration_hours, 2),
        'total_files_processed': len(csv_files),
        'successful_files': successful,
        'failed_files': failed,
        'total_parameter_tests': total_parameter_tests,
        'combinations_per_file': total_combinations,
        'avg_seconds_per_file': round(avg_time_per_file, 2),
        'avg_seconds_per_param_test': round(avg_time_per_param_test, 6),
        'cpu_cores_available': multiprocessing.cpu_count() if CPU_PARALLEL_AVAILABLE else 1,
        'parallel_processing_enabled': CPU_PARALLEL_AVAILABLE,
        'short_window_range_size': len(short_window_range),
        'long_window_range_size': len(long_window_range),
        'peak_tolerance_range_size': len(peak_tolerance_range),
        'count_threshold_range_size': len(count_threshold_range)
    }
    overall_timing_data.append(overall_timing_record)
    
    return output_files

def combine_param_test_records(param_tests_folder='param_tests_data', 
                              output_file='combined_param_tests.csv'):
    
    # MODIFIED: Track timing for combination process
    combine_start_time = time.time()
    combine_start_datetime = datetime.now()
    
    print(f"Starting parameter test record combination at: {combine_start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    
    test_files = []
    for root, _, files in os.walk(param_tests_folder):
        for file in files:
            if file.endswith('_param_tests.csv'):
                test_files.append(os.path.join(root, file))
    
    print(f"Combining {len(test_files)} parameter test files")
    
    if len(test_files) == 0:
        print("No parameter test files found!")
        return None
    
    dfs = []
    for file in test_files:
        df = pd.read_csv(file)
        dfs.append(df)
    
    if dfs:
        combined_df = pd.concat(dfs, ignore_index=True)
        combined_df.to_csv(output_file, index=False, float_format='%.6f')
        
        # MODIFIED: Track combination timing
        combine_end_time = time.time()
        combine_end_datetime = datetime.now()
        combine_duration = combine_end_time - combine_start_time
        
        print(f"Completed parameter test record combination at: {combine_end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Combination duration: {combine_duration:.2f} seconds")
        
        print(f"Combined parameter test records saved to {output_file}")
        print(f"Total test records: {len(combined_df)}")
        
        successful_tests = combined_df[combined_df['status'] == 'success']
        if len(successful_tests) > 0:
            print(f"Successful tests: {len(successful_tests)}")
            print(f"Best F1 score: {successful_tests['f1_score'].max():.4f}")
            print(f"Average F1 score: {successful_tests['f1_score'].mean():.4f}")
            
            best_row = successful_tests.loc[successful_tests['f1_score'].idxmax()]
            print(f"Best parameters:")
            print(f"  Short window: {best_row['short_window']}ms")
            print(f"  Long window: {best_row['long_window']}ms") 
            print(f"  Peak tolerance: {best_row['peak_tolerance']}")
            print(f"  Count threshold: {best_row['count_threshold']}")
            print(f"  F1 score: {best_row['f1_score']:.4f}")
        
        # MODIFIED: Store combination timing data
        combination_timing_record = {
            'process_name': 'combine_param_test_records',
            'start_time': combine_start_datetime.strftime('%Y-%m-%d %H:%M:%S'),
            'end_time': combine_end_datetime.strftime('%Y-%m-%d %H:%M:%S'),
            'duration_seconds': round(combine_duration, 2),
            'input_files': len(test_files),
            'total_records': len(combined_df),
            'successful_records': len(successful_tests) if len(successful_tests) > 0 else 0,
            'output_file': output_file
        }
        overall_timing_data.append(combination_timing_record)
        
        return output_file
    else:
        print("No valid parameter test files to combine")
        return None

def main():
    # MODIFIED: Track main execution timing
    main_start_time = time.time()
    main_start_datetime = datetime.now()
    
    print("Starting CPU-optimized parameter testing for DOWNSTREAM files...")
    print(f"Main execution started at: {main_start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print("🔧 Using MATLAB-compatible throughput calculation for consistency")
    
    # CONFIGURABLE PARAMETER RANGES
    short_window_range = list(range(100, 1001, 100))        # [100, 200, 300, ..., 1000] ms
    long_window_range = list(range(10000, 30501, 500))      # [10000, 10500, ..., 30500] ms  
    peak_tolerance_range = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]  # 5% to 30%
    count_threshold_range = [3, 5, 8, 10, 15]               # Sample count thresholds
    
    param_tests_folder = 'param_tests_data'
    
    print(f"Parameter search space:")
    print(f"  Short windows (ms): {len(short_window_range)} values from {min(short_window_range)} to {max(short_window_range)}")
    print(f"  Long windows (ms): {len(long_window_range)} values from {min(long_window_range)} to {max(long_window_range)}")
    print(f"  Peak tolerances: {len(peak_tolerance_range)} values {peak_tolerance_range}")
    print(f"  Count thresholds: {len(count_threshold_range)} values {count_threshold_range}")
    
    valid_combinations = 0
    for short_window in short_window_range:
        for long_window in long_window_range:
            if short_window < long_window:
                valid_combinations += len(peak_tolerance_range) * len(count_threshold_range)
    
    print(f"Total valid combinations per file: {valid_combinations}")
    
    processed_files = process_all_for_param_testing(
        input_folder='extracted_data',
        param_tests_folder=param_tests_folder,
        short_window_range=short_window_range,
        long_window_range=long_window_range, 
        peak_tolerance_range=peak_tolerance_range,
        count_threshold_range=count_threshold_range
    )
    
    output_file = combine_param_test_records(param_tests_folder=param_tests_folder)
    
    # MODIFIED: Calculate main execution timing
    main_end_time = time.time()
    main_end_datetime = datetime.now()
    main_duration = main_end_time - main_start_time
    main_duration_minutes = main_duration / 60.0
    main_duration_hours = main_duration / 3600.0
    
    print(f"\nMain execution completed at: {main_end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total main execution duration: {main_duration_hours:.2f} hours ({main_duration_minutes:.2f} minutes, {main_duration:.2f} seconds)")
    
    # MODIFIED: Store main execution timing
    main_execution_timing = {
        'process_name': 'main_execution_complete',
        'start_time': main_start_datetime.strftime('%Y-%m-%d %H:%M:%S'),
        'end_time': main_end_datetime.strftime('%Y-%m-%d %H:%M:%S'),
        'duration_seconds': round(main_duration, 2),
        'duration_minutes': round(main_duration_minutes, 2),
        'duration_hours': round(main_duration_hours, 2),
        'files_processed': len(processed_files) if processed_files else 0,
        'valid_combinations_per_file': valid_combinations,
        'cpu_cores': multiprocessing.cpu_count() if CPU_PARALLEL_AVAILABLE else 1,
        'parallel_enabled': CPU_PARALLEL_AVAILABLE
    }
    overall_timing_data.append(main_execution_timing)
    
    print("\nCPU-optimized parameter testing complete!")
    if output_file:
        print(f"Combined results: {output_file}")
    
    # MODIFIED: Save all timing data
    print("\nSaving comprehensive timing data...")
    save_timing_data()
    print(f"Timing data saved to: {timing_dir}/")

if __name__ == "__main__":
    main()