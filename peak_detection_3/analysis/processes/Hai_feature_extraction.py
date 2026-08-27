import numpy as np
import pandas as pd
import os
import re
from scipy.signal import medfilt
from peak_speed_detect import peak_speed_detect
from log_processor import extract_network_capacity

# Placeholder for importing student's feature extraction functions
# from student_features import calculate_additional_features

def extract_network_limit(subfolder):
    """Extract network limit from subfolder name"""
    return extract_network_capacity(subfolder)

def calculate_throughput(df, byte_col='rx_bytes', time_col='relative_time'):
    """Calculate throughput in Mbps"""
    byte_diff = df[byte_col].diff().fillna(0)
    time_diff = df[time_col].diff().fillna(0)
    throughput = (byte_diff * 8 / 1e6) / time_diff
    throughput = throughput.replace([np.inf, -np.inf, np.nan], 0)
    return throughput.values, df[time_col].values

# New function to calculate additional network metrics
def calculate_advanced_metrics(df, throughput, time):
    """Calculate advanced network metrics including student's features"""
    # Original metrics
    jitter = np.abs(np.diff(throughput))
    if len(jitter) > 0:
        jitter = np.append(jitter[0], jitter)
    else:
        jitter = np.zeros_like(throughput)
    
    window_size = min(5, len(throughput))
    if window_size > 0:
        moving_avg = np.convolve(throughput, np.ones(window_size)/window_size, mode='same')
    else:
        moving_avg = np.zeros_like(throughput)
    
    avg_throughput = np.mean(throughput) if len(throughput) > 0 else 0
    peak_throughput = np.max(throughput) if len(throughput) > 0 else 0
    burstiness = peak_throughput / avg_throughput if avg_throughput > 0 else 0
    
    # Student's features - calculated from the DataFrame
    # 1. Used bandwidth ratio (correcting 10^6 to 10**6 for proper exponentiation)
    used_bandwidth_ratio = np.zeros_like(throughput)
    if 'tx_bytes' in df.columns and 'network_capacity' in df.columns:
        used_bandwidth_ratio = (df['tx_bytes'].values / (df['network_capacity'].values * 10**6))
    
    # 2 & 3. Average packet sizes
    avg_rx_packet_size = np.zeros_like(throughput)
    avg_tx_packet_size = np.zeros_like(throughput)
    if 'rx_bytes' in df.columns and 'rx_packets' in df.columns:
        rx_packets_safe = df['rx_packets'].replace(0, np.nan).values
        avg_rx_packet_size = (df['rx_bytes'].values / rx_packets_safe)
        avg_rx_packet_size = np.nan_to_num(avg_rx_packet_size, nan=0.0)
    
    if 'tx_bytes' in df.columns and 'tx_packets' in df.columns:
        tx_packets_safe = df['tx_packets'].replace(0, np.nan).values
        avg_tx_packet_size = (df['tx_bytes'].values / tx_packets_safe)
        avg_tx_packet_size = np.nan_to_num(avg_tx_packet_size, nan=0.0)
    
    # 4. Network load factor
    network_load_factor = np.zeros_like(throughput)
    if 'tx_bytes' in df.columns and 'rx_bytes' in df.columns:
        network_load_factor = df['tx_bytes'].values - df['rx_bytes'].values
    
    # 5 & 6. Symmetric percentage changes
    rx_bytes_sym_pct_change = np.zeros_like(throughput)
    tx_bytes_sym_pct_change = np.zeros_like(throughput)
    
    if 'rx_bytes' in df.columns:
        prev_rx = df['rx_bytes'].shift(1).values
        rx_current = df['rx_bytes'].values
        denom_rx = (np.abs(rx_current) + np.abs(prev_rx)) / 2
        rx_bytes_sym_pct_change = ((rx_current - prev_rx) / denom_rx * 100)
        rx_bytes_sym_pct_change = np.nan_to_num(rx_bytes_sym_pct_change, nan=0.0)
    
    if 'tx_bytes' in df.columns:
        prev_tx = df['tx_bytes'].shift(1).values
        tx_current = df['tx_bytes'].values
        denom_tx = (np.abs(tx_current) + np.abs(prev_tx)) / 2
        tx_bytes_sym_pct_change = ((tx_current - prev_tx) / denom_tx * 100)
        tx_bytes_sym_pct_change = np.nan_to_num(tx_bytes_sym_pct_change, nan=0.0)
    
    # 7. Packet drop ratio
    packet_drop_ratio = np.zeros_like(throughput)
    if 'drops' in df.columns and 'tx_packets' in df.columns:
        tx_packets_safe = df['tx_packets'].replace(0, np.nan).values
        packet_drop_ratio = (df['drops'].values / tx_packets_safe) * 100
        packet_drop_ratio = np.nan_to_num(packet_drop_ratio, nan=0.0)
    
    return {
        # Original metrics
        'jitter': jitter,
        'moving_avg': moving_avg,
        'burstiness': burstiness,
        
        # Student's metrics
        'used_bandwidth_ratio': used_bandwidth_ratio,
        'avg_rx_packet_size': avg_rx_packet_size,
        'avg_tx_packet_size': avg_tx_packet_size,
        'network_load_factor': network_load_factor,
        'rx_bytes_sym_pct_change': rx_bytes_sym_pct_change,
        'tx_bytes_sym_pct_change': tx_bytes_sym_pct_change,
        # 'packet_drop_ratio': packet_drop_ratio
    }

def extract_features(csv_file, output_folder):
    """Extract features from CSV file"""
    try:
        # Load data and get path info
        df = pd.read_csv(csv_file)
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
        if 'relative_time' in df.columns:
            time_col = 'relative_time'
        elif time_cols:
            time_col = time_cols[0]
        else:
            df['index_time'] = df.index.astype(float)
            time_col = 'index_time'
            time_cols.append(time_col)
        
        # Calculate throughput
        rx_tp, time = calculate_throughput(df, 'rx_bytes', time_col)
        tx_tp, _ = calculate_throughput(df, 'tx_bytes', time_col)
        primary_tp = rx_tp if is_upstream else tx_tp
        
        # Calculate peak speed metrics
        score, peaks_lt, filtered_tp, ratio, peaks_st = peak_speed_detect(primary_tp, time)
        
        # Handle potential division by zero in ratio
        if isinstance(ratio, np.ndarray):
            ratio = np.nan_to_num(ratio, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            # If ratio is a single value
            if np.isnan(ratio) or np.isinf(ratio):
                ratio = 0.0
        
        # Get queue data
        queue_size = df['queue_size'].values if 'queue_size' in df.columns else np.zeros_like(time)
        queue_exists = df['queue_exists'].values if 'queue_exists' in df.columns else np.zeros_like(time)
        
        # Calculate additional metrics
        advanced_metrics = calculate_advanced_metrics(df, primary_tp, time)
        
        # Create features dataframe with original features
        features_df = pd.DataFrame({
            'peak_score': score.astype(float),
            'peak_lt': peaks_lt,
            'peak_st': peaks_st,
            'filtered_throughput': filtered_tp,
            'ratio_st_lt': ratio,
            'queue_size': queue_size,
            'queue_exists': queue_exists
        })
        
        # Add the new advanced metrics
        for metric_name, metric_values in advanced_metrics.items():
            if isinstance(metric_values, np.ndarray):
                if len(metric_values) == len(features_df):
                    features_df[metric_name] = metric_values
                else:
                    # Handle different lengths by extending or truncating
                    if len(metric_values) < len(features_df):
                        extended = np.concatenate([
                            metric_values, 
                            np.full(len(features_df) - len(metric_values), metric_values[-1] if len(metric_values) > 0 else 0)
                        ])
                        features_df[metric_name] = extended
                    else:
                        features_df[metric_name] = metric_values[:len(features_df)]
            else:
                # Handle scalar values
                features_df[metric_name] = metric_values
        
        # Add metadata columns
        features_df['file_name'] = base_name
        features_df['direction'] = 'upstream' if is_upstream else 'downstream'
        features_df['subfolder'] = subfolder
        features_df['network_limit'] = network_limit
        
        # Add time columns
        len_orig = len(df)
        len_feat = len(features_df)
        for tc in time_cols:
            if tc in df.columns:
                if len_feat <= len_orig:
                    offset = len_orig - len_feat
                    features_df[tc] = df[tc].astype(float).values[offset:]
                else:
                    extended = np.concatenate([
                        df[tc].values, 
                        np.full(len_feat - len_orig, df[tc].values[-1])
                    ])
                    features_df[tc] = extended
        
        # Create output path and save
        rel_path = os.path.relpath(dir_name, 'extracted_data')
        output_dir = os.path.join(output_folder, rel_path if rel_path != '.' else '')
        os.makedirs(output_dir, exist_ok=True)
        
        output_file = os.path.join(output_dir, base_name.replace('.csv', '_features.csv'))
        features_df.to_csv(output_file, index=False, float_format='%.6f')
        
        return output_file
    except Exception as e:
        print(f"Error processing {csv_file}: {str(e)}")
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
    
    # Process each file
    output_files = []
    for csv_file in csv_files:
        if output_file := extract_features(csv_file, output_folder):
            output_files.append(output_file)
    
    print(f"Feature extraction complete. {len(output_files)} files generated.")
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
    
    # Read and process each file
    dfs = []
    for file in feature_files:
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
                
        # Ensure ratio_st_lt has no NaN or inf values
        if 'ratio_st_lt' in df.columns:
            df['ratio_st_lt'] = df['ratio_st_lt'].replace([np.inf, -np.inf, np.nan], 0.0)
        
        # Ensure new metrics have no NaN or inf values
        for col in df.columns:
            if col in ['jitter', 'moving_avg', 'burstiness', 'used_bandwidth_ratio', 
                       'avg_rx_packet_size', 'avg_tx_packet_size', 'network_load_factor',
                       'rx_bytes_sym_pct_change', 'tx_bytes_sym_pct_change', 'packet_drop_ratio']:
                df[col] = df[col].replace([np.inf, -np.inf, np.nan], 0.0)
        
        dfs.append(df)
    
    # Combine all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Save combined data only
    combined_df.to_csv(output_file, index=False, float_format='%.6f')
    
    # Print summary
    print(f"Combined features saved to {output_file}")
    print(f"Found {len(combined_df['network_limit'].unique())} unique network limits")
    print(f"Found {len(combined_df['subfolder'].unique())} unique subfolders")
    print(f"Found {len(combined_df['file_name'].unique())} unique files")
    print(f"Records: {len(combined_df)} total")
    
    return output_file

def analyze_structure(feature_folder='feature_data'):
    """Analyze feature_data folder structure"""
    # Get subfolders
    subfolders = [f for f in os.listdir(feature_folder) 
                 if os.path.isdir(os.path.join(feature_folder, f))]
    
    # Map files by subfolder
    file_map = {}
    for subfolder in subfolders:
        subfolder_path = os.path.join(feature_folder, subfolder)
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
    print("Starting feature extraction process...")
    
    # Analyze existing structure if available
    if os.path.exists('feature_data'):
        analyze_structure()
    
    # Extract features from input files
    extracted_files = extract_all_features()
    
    # Analyze generated structure
    analyze_structure()
    
    # Combine all features into single file only
    output_file = combine_features()
    
    print("Feature extraction complete!")
    print(f"File saved: {output_file}")

if __name__ == "__main__":
    main()