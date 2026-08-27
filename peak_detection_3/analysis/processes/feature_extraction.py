import numpy as np
import pandas as pd
import os
import re
from scipy.signal import medfilt
from peak_speed_detect import peak_speed_detect
from log_processor import extract_network_capacity

def extract_network_limit(subfolder):
    """Extract network limit from subfolder name"""
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

def extract_features(csv_file, output_folder):
    """Extract features from CSV file using MATLAB-compatible throughput calculation"""
    try:
        print(f"Processing: {csv_file}")
        
        # Load data and get path info
        df = pd.read_csv(csv_file)
        print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
        
        dir_name = os.path.dirname(csv_file)
        base_name = os.path.basename(csv_file)
        subfolder = os.path.basename(dir_name)
        network_limit = extract_network_limit(subfolder)
        is_upstream = 'upstream' in base_name
        
        # Find time columns
        time_cols = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()]
        for col in time_cols:
            df[col] = df[col].astype(float)
            
        # Use relative_time or first time column
        if 'time' in df.columns:
            time_col = 'time'
        elif time_cols:
            time_col = time_cols[0]
        else:
            df['index_time'] = df.index.astype(float)
            time_col = 'index_time'
            time_cols.append(time_col)
        
        print(f"  Using time column: {time_col}")
        
        # Calculate throughput using MATLAB-compatible method (same as Streamlit app)
        rx_tp, time = calculate_throughput_matlab_method(df, 'rx_bytes', time_col)
        tx_tp, _ = calculate_throughput_matlab_method(df, 'tx_bytes', time_col)
        primary_tp = rx_tp if is_upstream else tx_tp
        
        print(f"  Calculated throughput: {len(primary_tp)} samples, direction: {'upstream' if is_upstream else 'downstream'}")
        print(f"  Throughput range: {primary_tp.min():.2f} - {primary_tp.max():.2f} Mbps (MATLAB method)")
        
        # Check for negative throughput (should not happen with MATLAB method)
        negative_count = np.sum(primary_tp < 0)
        if negative_count > 0:
            print(f"  Warning: Found {negative_count} negative throughput values (unexpected with MATLAB method)")
            primary_tp = np.maximum(primary_tp, 0)  # Fix any remaining negatives
        
        # Skip files with insufficient data
        if len(primary_tp) < 10:
            print(f"  Skipping file - insufficient data: {len(primary_tp)} samples")
            return None
        
        # Check for all-zero throughput
        if np.all(primary_tp == 0):
            print(f"  Warning: All throughput values are zero")
        
        # Calculate peak speed metrics using the Max algorithm
        try:
            clipping_score, long_maxima, filtered_tp, peak_ratio, short_counts, short_peaks = peak_speed_detect(
                primary_tp, time,
                long_window_ms=28500,   # 10 second long windows
                short_window_ms=200,    # 0.2 second short bins  
                peak_tolerance=0.3,     # Within 20% of peak
                count_threshold=3       # At least 5 samples near peak for clipping
            )
            print(f"  Peak detection successful: 6 return values")
            
        except Exception as e:
            print(f"  Peak detection failed: {e}")
            print("  Using default values")
            # Create default arrays if peak detection fails
            clipping_score = np.zeros(len(primary_tp))
            long_maxima = np.zeros(len(primary_tp))
            filtered_tp = primary_tp.copy()
            peak_ratio = np.zeros(len(primary_tp))
            short_counts = np.zeros(len(primary_tp))
            short_peaks = np.zeros(len(primary_tp))
        
        # Ensure all arrays are 1D and the same length
        target_length = len(primary_tp)
        
        def ensure_1d_array(arr, target_len, name):
            """Ensure array is 1D and correct length"""
            arr = np.asarray(arr).flatten()
            if len(arr) != target_len:
                if len(arr) > target_len:
                    arr = arr[:target_len]
                else:
                    # Pad with zeros or repeat last value
                    if len(arr) > 0:
                        pad_val = arr[-1] if name in ['long_maxima', 'filtered_tp'] else 0
                        arr = np.concatenate([arr, np.full(target_len - len(arr), pad_val)])
                    else:
                        arr = np.zeros(target_len)
            return arr
        
        clipping_score = ensure_1d_array(clipping_score, target_length, 'clipping_score')
        long_maxima = ensure_1d_array(long_maxima, target_length, 'long_maxima')
        filtered_tp = ensure_1d_array(filtered_tp, target_length, 'filtered_tp')
        peak_ratio = ensure_1d_array(peak_ratio, target_length, 'peak_ratio')
        short_counts = ensure_1d_array(short_counts, target_length, 'short_counts')
        short_peaks = ensure_1d_array(short_peaks, target_length, 'short_peaks')  # Fixed: was using short_counts
        
        
        # Ensure no inf or nan values
        clipping_score = np.nan_to_num(clipping_score, nan=0.0, posinf=0.0, neginf=0.0)
        long_maxima = np.nan_to_num(long_maxima, nan=0.0, posinf=0.0, neginf=0.0)
        filtered_tp = np.nan_to_num(filtered_tp, nan=0.0, posinf=0.0, neginf=0.0)
        peak_ratio = np.nan_to_num(peak_ratio, nan=0.0, posinf=0.0, neginf=0.0)
        short_counts = np.nan_to_num(short_counts, nan=0.0, posinf=0.0, neginf=0.0)
        short_peaks = np.nan_to_num(short_peaks, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Get queue data (ensure same length)
        if 'queue_size' in df.columns:
            queue_size = ensure_1d_array(df['queue_size'].values, target_length, 'queue_size')
        else:
            queue_size = np.zeros(target_length)
            
        if 'queue_exists' in df.columns:
            queue_exists = ensure_1d_array(df['queue_exists'].values, target_length, 'queue_exists')
        else:
            queue_exists = np.zeros(target_length)
        
        # Create features dataframe
        features_df = pd.DataFrame({
            'clipping_score': clipping_score.astype(float),       # Max algorithm clipping score (0-1)
            'long_term_peak': long_maxima.astype(float),          # Maximum values in long windows
            'short_term_counts': short_counts.astype(float),      # Count of samples near peak in short bins
            'short_term_peaks': short_peaks.astype(float),        # Peaks of short bins
            'filtered_throughput': filtered_tp.astype(float),     # Median filtered throughput
            'peak_ratio': peak_ratio.astype(float),              # Ratio of short peaks to long peaks
            'queue_size': queue_size.astype(float),
            'queue_exists': queue_exists.astype(float)
        })
        
        print(f"  Created features dataframe: {len(features_df)} rows")
        
        # Add metadata columns
        features_df['file_name'] = base_name
        features_df['direction'] = 'upstream' if is_upstream else 'downstream'
        features_df['subfolder'] = subfolder
        features_df['network_limit'] = network_limit
        
        # Add time columns (ensure proper alignment)
        len_orig = len(df)
        len_feat = len(features_df)
        
        for tc in time_cols:
            if tc in df.columns:
                time_data = df[tc].astype(float).values
                time_data = ensure_1d_array(time_data, len_feat, tc)
                features_df[tc] = time_data
        
        # Verify all columns are 1D
        for col in features_df.columns:
            if features_df[col].dtype == 'object':
                continue  # Skip string columns
            col_data = features_df[col].values
            if col_data.ndim > 1:
                print(f"  Warning: Column {col} is not 1D, flattening...")
                features_df[col] = col_data.flatten()
        
        # Verify data quality
        print(f"  Features quality check:")
        print(f"    Clipping score range: {features_df['clipping_score'].min():.3f} - {features_df['clipping_score'].max():.3f}")
        print(f"    Peak ratio range: {features_df['peak_ratio'].min():.3f} - {features_df['peak_ratio'].max():.3f}")
        print(f"    Filtered throughput range: {features_df['filtered_throughput'].min():.2f} - {features_df['filtered_throughput'].max():.2f}")
        
        # Create output path and save
        rel_path = os.path.relpath(dir_name, 'extracted_data')
        output_dir = os.path.join(output_folder, rel_path if rel_path != '.' else '')
        os.makedirs(output_dir, exist_ok=True)
        
        output_file = os.path.join(output_dir, base_name.replace('.csv', '_features.csv'))
        features_df.to_csv(output_file, index=False, float_format='%.6f')
        
        print(f"  Saved to: {output_file}")
        return output_file
        
    except Exception as e:
        print(f"Error processing {csv_file}: {str(e)}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        return None

def extract_all_features(input_folder='extracted_data', output_folder='feature_data'):
    """Process all CSV files in input folder"""
    os.makedirs(output_folder, exist_ok=True)
    
    # Find all CSV files
    csv_files = []
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    
    print(f"Found {len(csv_files)} CSV files to process")
    print(f"Using MATLAB-compatible throughput calculation for consistency")
    
    # Process each file
    output_files = []
    successful = 0
    failed = 0
    
    for i, csv_file in enumerate(csv_files, 1):
        print(f"\n[{i}/{len(csv_files)}] Processing file...")
        result = extract_features(csv_file, output_folder)
        if result:
            output_files.append(result)
            successful += 1
        else:
            failed += 1
    
    print(f"\nFeature extraction complete!")
    print(f"  Successful: {successful} files")
    print(f"  Failed: {failed} files")
    print(f"  Total output files: {len(output_files)}")
    
    return output_files

def combine_features(feature_folder='feature_data', output_file='combined_features.csv'):
    """Combine all feature files into single CSV"""
    # Find all feature files
    feature_files = []
    for root, _, files in os.walk(feature_folder):
        for file in files:
            if file.endswith('_features.csv'):
                feature_files.append(os.path.join(root, file))
    
    print(f"Combining {len(feature_files)} feature files")
    
    if len(feature_files) == 0:
        print("No feature files found!")
        # Create empty dataframe with expected columns
        empty_df = pd.DataFrame(columns=[
            'clipping_score', 'long_term_peak', 'short_term_counts', 'short_term_peaks',
            'filtered_throughput', 'peak_ratio', 'queue_size', 'queue_exists',
            'file_name', 'direction', 'subfolder', 'network_limit'
        ])
        empty_df.to_csv(output_file, index=False)
        return output_file
    
    # Read and process each file
    dfs = []
    for file in feature_files:
        try:
            df = pd.read_csv(file)
            
            # Add missing columns if needed
            if 'subfolder' not in df.columns:
                df['subfolder'] = os.path.basename(os.path.dirname(file))
                
            if 'network_limit' not in df.columns:
                df['network_limit'] = extract_network_limit(df['subfolder'].iloc[0])
            
            # Convert time columns to float
            for col in df.columns:
                if any(time_word in col.lower() for time_word in ['time', 'date', 'timestamp']):
                    df[col] = df[col].astype(float)
                    
            # Ensure numerical columns have no NaN or inf values
            numerical_cols = ['clipping_score', 'peak_ratio', 'long_term_peak', 'short_term_counts', 
                            'short_term_peaks', 'filtered_throughput', 'queue_size', 'queue_exists']
            for col in numerical_cols:
                if col in df.columns:
                    df[col] = df[col].replace([np.inf, -np.inf, np.nan], 0.0)
            
            # Handle legacy column names if they exist
            if 'ratio_st_lt' in df.columns and 'peak_ratio' not in df.columns:
                df['peak_ratio'] = df['ratio_st_lt'].replace([np.inf, -np.inf, np.nan], 0.0)
            
            dfs.append(df)
            
        except Exception as e:
            print(f"Error reading {file}: {e}")
            continue
    
    # Check if we have any dataframes to combine
    if not dfs:
        print("No valid feature files found. Creating empty combined file.")
        # Create empty dataframe with expected columns
        empty_df = pd.DataFrame(columns=[
            'clipping_score', 'long_term_peak', 'short_term_counts', 'short_term_peaks',
            'filtered_throughput', 'peak_ratio', 'queue_size', 'queue_exists',
            'file_name', 'direction', 'subfolder', 'network_limit'
        ])
        empty_df.to_csv(output_file, index=False)
        return output_file
    
    # Combine all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Final data quality checks
    print(f"\nData quality summary:")
    if len(combined_df) > 0:
        downstream_data = combined_df[combined_df['direction'] == 'downstream']
        upstream_data = combined_df[combined_df['direction'] == 'upstream']
        
        print(f"  Total records: {len(combined_df)}")
        print(f"  Downstream records: {len(downstream_data)}")
        print(f"  Upstream records: {len(upstream_data)}")
        
        if len(downstream_data) > 0:
            print(f"  Downstream throughput range: {downstream_data['filtered_throughput'].min():.2f} - {downstream_data['filtered_throughput'].max():.2f} Mbps")
            print(f"  Downstream clipping detection rate: {downstream_data['clipping_score'].mean()*100:.1f}%")
    
    # Save combined data
    combined_df.to_csv(output_file, index=False, float_format='%.6f')
    
    # Print summary
    print(f"\nCombined features saved to {output_file}")
    print(f"Found {len(combined_df['network_limit'].unique())} unique network limits")
    print(f"Found {len(combined_df['subfolder'].unique())} unique subfolders") 
    print(f"Found {len(combined_df['file_name'].unique())} unique files")
    print(f"Records: {len(combined_df)} total")
    
    return output_file

def analyze_structure(feature_folder='feature_data'):
    """Analyze feature_data folder structure"""
    if not os.path.exists(feature_folder):
        print(f"Folder {feature_folder} does not exist")
        return [], {}, set()
        
    # Get subfolders
    subfolders = [f for f in os.listdir(feature_folder) 
                 if os.path.isdir(os.path.join(feature_folder, f))]
    
    # Map files by subfolder
    file_map = {}
    for subfolder in subfolders:
        subfolder_path = os.path.join(feature_folder, subfolder)
        if os.path.exists(subfolder_path):
            file_map[subfolder] = [f for f in os.listdir(subfolder_path) 
                                  if f.endswith('.csv')]
    
    # Find common files
    if file_map:
        all_files = set().union(*[set(files) for files in file_map.values()])
        common_files = set.intersection(*[set(files) for files in file_map.values()])
    else:
        all_files, common_files = set(), set()
    
    print(f"\nStructure Analysis: {len(subfolders)} subfolders, {len(all_files)} unique files")
    print(f"Files common to all subfolders: {len(common_files)}")
    
    return subfolders, file_map, common_files

def main():
    """Run feature extraction pipeline"""
    print("Starting MATLAB-compatible feature extraction process...")
    print("🔧 Using MATLAB-compatible throughput calculation for consistency")
    print("Using Max-based peak speed detection algorithm")
    
    # Analyze existing structure if available
    if os.path.exists('feature_data'):
        analyze_structure()
    
    # Extract features from input files
    extracted_files = extract_all_features()
    
    # Analyze generated structure
    analyze_structure()
    
    # Combine all features into single file
    output_file = combine_features()
    
    print("\nMATLAB-compatible feature extraction complete!")
    print(f"Combined file saved: {output_file}")
    print("✅ Features should now be consistent with Streamlit app and parameter testing")

if __name__ == "__main__":
    main()