import numpy as np
from scipy.signal import medfilt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# def peak_speed_detect(throughput, time, bin_widths_ms=[50, 2000], threshold=0.9):
#     """Calculate peak speed metrics"""
#     mean_dt = np.mean(np.diff(time)) * 1000
#     short_step = round(bin_widths_ms[0] / mean_dt)
#     long_step = round(bin_widths_ms[1] / mean_dt)
    
#     # Add padding and initialize arrays
#     padding = np.zeros(long_step)
#     throughput = np.concatenate([padding, throughput])
#     time = np.concatenate([padding, time])
#     len_data = len(throughput)
    
#     # Initialize arrays
#     st_memfactor = 0.95
#     moving_avg = np.zeros(len_data)
#     peaks_st = np.zeros(len_data)
#     peaks_lt = np.zeros(len_data)
    
#     # Calculate moving average
#     for i in range(1, len_data):
#         moving_avg[i] = st_memfactor * moving_avg[i-1] + (1-st_memfactor) * throughput[i]
    
#     # Short-term peaks
#     for i in range(0, len_data - short_step, short_step):
#         peaks_st[i:i+short_step] = np.max(throughput[i:i+short_step])
    
#     # Apply median filter and calculate long-term peaks
#     filtered = medfilt(throughput, 15)
#     for i in range(long_step, len_data - long_step):
#         peaks_lt[i] = np.max(filtered[i-long_step:i+long_step])
    
#     # Calculate metrics
#     score = peaks_st > threshold * peaks_lt
#     ratio = peaks_st / peaks_lt
    
#     # Remove padding
#     return (score[long_step:], peaks_lt[long_step:], filtered[long_step:], 
#             ratio[long_step:], peaks_st[long_step:])
    
def peak_speed_detect(throughput, time, long_window_ms=25000, short_window_ms=200, 
                      peak_tolerance=0.3, count_threshold=3, use_median_filter=True):
    """
    Implement the Max algorithm for peak speed detection with binary output
    
    This algorithm follows the exact steps described in the Max:
    1. Calculate maximum throughput over 'long windows'
    2. Break long windows into smaller bins and count samples close to maximum
    3. Return bins where count > threshold as 'clipping' (binary classification)
    
    Parameters:
    - throughput: array of throughput measurements
    - time: array of time values (seconds)
    - long_window_ms: duration of long window in milliseconds (default: 2000ms)
    - short_window_ms: duration of short window/bin in milliseconds (default: 200ms)  
    - peak_tolerance: tolerance for being "close" to peak (as fraction, e.g. 0.1 = within 10% of peak)
    - count_threshold: minimum number of samples in short bin to classify as clipping
    - use_median_filter: whether to apply median filter to reduce noise (default: True)
    
    Returns (all 1D arrays of same length as input):
    - clipping_score: binary array (0 or 1) indicating clipping classification for each sample
    - long_term_peak: maximum values found in each long window
    - short_term_peaks: maximum values found in each long window
    - filtered_throughput: median filtered throughput (or original if filtering disabled)
    - peak_ratio: ratio of short-term peaks to long-term peaks
    - short_term_counts: count of near-peak samples for each sample
    """
    import numpy as np
    from scipy.signal import medfilt
    
    if len(throughput) != len(time):
        raise ValueError("throughput and time arrays must have the same length")
    
    if len(throughput) < 2:
        # Return zeros for insufficient data
        zeros = np.zeros(len(throughput))
        return zeros.astype(int), zeros, throughput.copy(), zeros, zeros
    
    # Calculate mean time step
    mean_dt = np.mean(np.diff(time))
    
    # Convert window sizes to number of samples
    long_window_samples = max(1, int(long_window_ms / 1000 / mean_dt))
    short_window_samples = max(1, int(short_window_ms / 1000 / mean_dt))
    
    n_samples = len(throughput)
    
    # Apply median filter conditionally
    if use_median_filter:
        filter_length = min(15, n_samples)
        if filter_length % 2 == 0:
            filter_length -= 1
        filtered_throughput = medfilt(throughput, max(1, filter_length))
    else:
        # No filtering - use raw data
        filtered_throughput = throughput.copy()
    
    # Initialize output arrays
    clipping_binary = np.zeros(n_samples, dtype=int)  # Binary integer array
    long_maxima = np.zeros(n_samples)
    short_counts = np.zeros(n_samples)
    short_peaks = np.zeros(n_samples)
    
    # Step 1: Process each long window
    long_window_starts = range(0, n_samples, long_window_samples)
    
    for long_start in long_window_starts:
        long_end = min(long_start + long_window_samples, n_samples)
        
        # Find maximum in this long window using filtered data
        long_window_data = filtered_throughput[long_start:long_end]
        if len(long_window_data) == 0:
            continue
            
        long_max = np.max(long_window_data)
        
        # Fill long_maxima array for this window
        long_maxima[long_start:long_end] = long_max
        
        # Step 2: Break this long window into short bins
        short_bin_starts = range(long_start, long_end, short_window_samples)
        
        for short_start in short_bin_starts:
            short_end = min(short_start + short_window_samples, long_end)
            
            # Get data in this short bin (use raw data for counting)
            short_bin_data = throughput[short_start:short_end]
            
            if len(short_bin_data) == 0:
                continue
            
            # Calculate short-term peak (maximum in this short window)
            short_peak = np.max(short_bin_data)
            short_peaks[short_start:short_end] = short_peak
            
            # Step 2: Count samples close to long maximum
            # "Close" means within peak_tolerance of the maximum
            tolerance_threshold = (1 - peak_tolerance) * long_max
            near_peak_mask = short_bin_data >= tolerance_threshold
            near_peak_count = np.sum(near_peak_mask)
            
            # Fill count array for this short bin
            short_counts[short_start:short_end] = near_peak_count
            
            # Step 3: Binary classification based on count threshold (Max Step 3)
            # If count > threshold, classify as clipping (1), otherwise not clipping (0)
            if near_peak_count > count_threshold:
                clipping_binary[short_start:short_end] = 1
            else:
                clipping_binary[short_start:short_end] = 0
    
    # Calculate peak ratio (short-term peaks / long-term peaks)
    peak_ratio = np.zeros(n_samples)
    nonzero_mask = long_maxima > 0
    peak_ratio[nonzero_mask] = short_peaks[nonzero_mask] / long_maxima[nonzero_mask]
    
    # Ensure no inf or nan values
    peak_ratio = np.nan_to_num(peak_ratio, nan=0.0, posinf=0.0, neginf=0.0)
    long_maxima = np.nan_to_num(long_maxima, nan=0.0, posinf=0.0, neginf=0.0)
    short_counts = np.nan_to_num(short_counts, nan=0.0, posinf=0.0, neginf=0.0)
    short_peaks = np.nan_to_num(short_peaks, nan=0.0, posinf=0.0, neginf=0.0)
    filtered_throughput = np.nan_to_num(filtered_throughput, nan=0.0, posinf=0.0, neginf=0.0)
    
    return clipping_binary, long_maxima, filtered_throughput, peak_ratio, short_counts, short_peaks

def peak_speed_detect_adaptive(throughput, time, ground_truth, 
                              short_window_range=[100, 200, 500, 1000], 
                              long_window_range=[1000, 2000, 5000, 10000],
                              peak_tolerance_range=[0.05, 0.1, 0.15, 0.2],
                              count_threshold_range=[3, 5, 8, 10, 15]):
    """
    Find optimal peak speed detection parameters using the Max algorithm by scanning 
    different parameter combinations and comparing against ground truth.
    
    This function implements the Max algorithm with adaptive parameter tuning:
    1. Tests different combinations of long/short windows, peak tolerance, and count thresholds
    2. Uses the Max algorithm's binary classification approach for each combination
    3. Evaluates performance against ground truth using multiple metrics
    4. Returns the best parameter set and corresponding results
    
    Args:
        throughput: array of throughput values
        time: array of time values  
        ground_truth: binary array of ground truth values (1=congestion, 0=no congestion)
        short_window_range: list of short window durations in ms to test
        long_window_range: list of long window durations in ms to test
        peak_tolerance_range: list of peak tolerance values (fraction) to test
        count_threshold_range: list of sample count thresholds to test
    
    Returns:
        tuple: (clipping_binary, long_maxima, filtered_throughput, peak_ratio, short_counts,
                best_short_window, best_long_window, best_peak_tolerance, 
                best_count_threshold, best_metrics)
    """
    
    best_f1 = 0
    best_accuracy = 0
    best_precision = 0
    best_recall = 0
    best_params = None
    best_results = None
    best_metrics = {}
    
    total_combinations = (len(short_window_range) * len(long_window_range) * 
                         len(peak_tolerance_range) * len(count_threshold_range))
    
    print("Starting adaptive Max algorithm parameter search...")
    print(f"Testing {total_combinations} parameter combinations")
    print("Parameters to optimize:")
    print(f"  Short windows: {short_window_range} ms")
    print(f"  Long windows: {long_window_range} ms") 
    print(f"  Peak tolerances: {peak_tolerance_range}")
    print(f"  Count thresholds: {count_threshold_range}")
    
    combination_count = 0
    
    # Grid search over all parameter combinations
    for short_window in short_window_range:
        for long_window in long_window_range:
            # Skip invalid combinations where short >= long window
            if short_window >= long_window:
                continue
                
            for peak_tolerance in peak_tolerance_range:
                for count_threshold in count_threshold_range:
                    combination_count += 1
                    
                    try:
                        # Run Max algorithm with current parameter set
                        clipping_binary, long_maxima, filtered_tp, peak_ratio, short_counts, short_peaks = peak_speed_detect(
                            throughput, time,
                            long_window_ms=long_window,
                            short_window_ms=short_window,
                            peak_tolerance=peak_tolerance,
                            count_threshold=count_threshold
                        )
                        
                        # clipping_binary is already binary (0 or 1) from the fixed function
                        # No need for threshold conversion anymore
                        binary_predictions = clipping_binary.astype(int)
                        
                        # Make sure prediction and ground truth have the same length
                        min_len = min(len(binary_predictions), len(ground_truth))
                        predictions = binary_predictions[:min_len]
                        gt = ground_truth[:min_len]
                        
                        # Skip if no positive predictions or all predictions are the same
                        if np.sum(predictions) == 0 or np.all(predictions == predictions[0]):
                            continue
                            
                        # Calculate performance metrics
                        accuracy = accuracy_score(gt, predictions)
                        f1 = f1_score(gt, predictions, zero_division=0)
                        precision = precision_score(gt, predictions, zero_division=0)
                        recall = recall_score(gt, predictions, zero_division=0)
                        
                        # Progress reporting
                        if combination_count % 50 == 0:
                            print(f"Progress: {combination_count}/{total_combinations} combinations tested...")
                        
                        # Update best parameters if better F1 score found
                        # Primary metric is F1, but we also track others
                        if f1 > best_f1:
                            best_f1 = f1
                            best_accuracy = accuracy
                            best_precision = precision
                            best_recall = recall
                            
                            best_params = {
                                'short_window': short_window,
                                'long_window': long_window,
                                'peak_tolerance': peak_tolerance,
                                'count_threshold': count_threshold
                            }
                            
                            best_results = (clipping_binary, long_maxima, filtered_tp, peak_ratio, short_counts, short_peaks)
                            
                            best_metrics = {
                                'accuracy': accuracy,
                                'f1_score': f1,
                                'precision': precision,
                                'recall': recall
                            }
                            
                            print(f"New best parameters found:")
                            print(f"  Short window: {short_window}ms, Long window: {long_window}ms")
                            print(f"  Peak tolerance: {peak_tolerance}, Count threshold: {count_threshold}")
                            print(f"  Metrics - Accuracy: {accuracy:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
                            
                    except Exception as e:
                        print(f"Error with parameters (short={short_window}, long={long_window}, tolerance={peak_tolerance}, threshold={count_threshold}): {str(e)}")
                        continue
    
    # Handle case where no valid combination found
    if best_results is None:
        print("No valid parameter combination found. Using default parameters.")
        # Use default parameters from the original function
        clipping_binary, long_maxima, filtered_tp, peak_ratio, short_counts, short_peaks = peak_speed_detect(
            throughput, time
        )
        
        best_params = {
            'short_window': 200,
            'long_window': 2000,
            'peak_tolerance': 0.1,
            'count_threshold': 5
        }
        
        best_metrics = {
            'accuracy': 0,
            'f1_score': 0,
            'precision': 0,
            'recall': 0
        }
        
        return (clipping_binary, long_maxima, filtered_tp, peak_ratio, short_counts, short_peaks,
                best_params['short_window'], best_params['long_window'], 
                best_params['peak_tolerance'], best_params['count_threshold'], best_metrics)
    
    print(f"\nParameter search complete!")
    print(f"Total combinations tested: {combination_count}")
    print(f"Best parameter set:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    print(f"Best performance metrics:")
    for metric, value in best_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    return (*best_results, 
            best_params['short_window'], best_params['long_window'], 
            best_params['peak_tolerance'], best_params['count_threshold'], 
            best_metrics)


def evaluate_peak_detection_performance(throughput, time, ground_truth, 
                                      short_window=1000, long_window=5000,
                                      peak_tolerance=0.1, count_threshold=8):
    """
    Evaluate peak detection performance with specific parameters
    
    Args:
        throughput: array of throughput values
        time: array of time values
        ground_truth: binary array of ground truth values
        short_window: short window duration in ms
        long_window: long window duration in ms
        peak_tolerance: peak tolerance fraction
        count_threshold: sample count threshold
        
    Returns:
        dict: performance metrics and predictions
    """
    
    # Run Max algorithm
    clipping_binary, long_maxima, filtered_tp, peak_ratio, short_counts, short_peaks = peak_speed_detect(
        throughput, time,
        long_window_ms=long_window,
        short_window_ms=short_window,
        peak_tolerance=peak_tolerance,
        count_threshold=count_threshold
    )
    
    # clipping_binary is already binary (0 or 1) from the fixed function
    binary_predictions = clipping_binary.astype(int)
    
    # Align lengths
    min_len = min(len(binary_predictions), len(ground_truth))
    predictions = binary_predictions[:min_len]
    gt = ground_truth[:min_len]
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(gt, predictions),
        'f1_score': f1_score(gt, predictions, zero_division=0),
        'precision': precision_score(gt, predictions, zero_division=0),
        'recall': recall_score(gt, predictions, zero_division=0),
        'true_positives': np.sum((gt == 1) & (predictions == 1)),
        'true_negatives': np.sum((gt == 0) & (predictions == 0)),
        'false_positives': np.sum((gt == 0) & (predictions == 1)),
        'false_negatives': np.sum((gt == 1) & (predictions == 0)),
        'total_samples': len(gt),
        'positive_samples': np.sum(gt),
        'negative_samples': np.sum(gt == 0)
    }
    
    return {
        'metrics': metrics,
        'predictions': predictions,
        'clipping_binary': clipping_binary[:min_len],
        'ground_truth': gt,
        'parameters': {
            'short_window': short_window,
            'long_window': long_window,
            'peak_tolerance': peak_tolerance,
            'count_threshold': count_threshold
        }
    }