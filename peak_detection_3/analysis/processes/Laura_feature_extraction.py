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

def calculate_throughput(df, byte_col='rx_bytes', time_col='relative_time'):
    """Calculate throughput in Mbps"""
    byte_diff = df[byte_col].diff().fillna(0)
    time_diff = df[time_col].diff().fillna(0)
    throughput = (byte_diff * 8 / 1e6) / time_diff
    throughput = throughput.replace([np.inf, -np.inf, np.nan], 0)
    return throughput.values, df[time_col].values

def add_laura_features(df, time_col='relative_time'):
    """Add additional features based on Laura's feature engineering work"""
    # Store original index for later
    orig_index = df.index.copy()
    
    # Define window sizes
    window_sizes = [40, 400, 2000]  # about 1 sec, 10 sec, 50 sec
    
    # 1. Calculate time differences
    df['time_diff'] = df[time_col].diff().fillna(0)
    
    # Note: Removed loss_ratio calculation as it uses drops column
    
    # 2. Calculate rates (excluding drops)
    df['tx_bytes_rate'] = 8 * df['tx_bytes'].diff() / (df['time_diff'] * 1e6)  # in Mbits/s
    df['rx_bytes_rate'] = 8 * df['rx_bytes'].diff() / (df['time_diff'] * 1e6)  # in Mbits/s
    df['tx_packets_rate'] = df['tx_packets'].diff() / df['time_diff']
    df['rx_packets_rate'] = df['rx_packets'].diff() / df['time_diff']
    
    # 3. Calculate acceleration (second derivatives)
    df['tx_bytes_accel'] = df['tx_bytes_rate'].diff() / df['time_diff']
    df['rx_bytes_accel'] = df['rx_bytes_rate'].diff() / df['time_diff']
    df['tx_packets_accel'] = df['tx_packets_rate'].diff() / df['time_diff']
    df['rx_packets_accel'] = df['rx_packets_rate'].diff() / df['time_diff']
    
    # 4. Moving standard deviation
    for w in window_sizes:
        df[f'tx_bytes_std_{w}'] = df['tx_bytes_rate'].rolling(window=w).std()
        df[f'rx_bytes_std_{w}'] = df['rx_bytes_rate'].rolling(window=w).std()
        df[f'tx_packets_std_{w}'] = df['tx_packets_rate'].rolling(window=w).std()
        df[f'rx_packets_std_{w}'] = df['rx_packets_rate'].rolling(window=w).std()
    
    # 5. Moving average
    for w in window_sizes:
        df[f'tx_bytes_ma_{w}'] = df['tx_bytes_rate'].rolling(window=w).mean()
        df[f'rx_bytes_ma_{w}'] = df['rx_bytes_rate'].rolling(window=w).mean()
        df[f'tx_packets_ma_{w}'] = df['tx_packets_rate'].rolling(window=w).mean()
        df[f'rx_packets_ma_{w}'] = df['rx_packets_rate'].rolling(window=w).mean()
    
    # 6. Rolling Min/Max over a Window
    for win in window_sizes:
        # MAX
        df[f'max_rx_bytes_rate_{win}'] = df["rx_bytes_rate"].rolling(window=win).max() 
        df[f'max_tx_bytes_rate_{win}'] = df["tx_bytes_rate"].rolling(window=win).max() 
        df[f'max_rx_packets_rate_{win}'] = df["rx_packets_rate"].rolling(window=win).max() 
        df[f'max_tx_packets_rate_{win}'] = df["tx_packets_rate"].rolling(window=win).max() 

        # MIN
        df[f'min_rx_bytes_rate_{win}'] = df["rx_bytes_rate"].rolling(window=win).min()
        df[f'min_tx_bytes_rate_{win}'] = df["tx_bytes_rate"].rolling(window=win).min()
        df[f'min_rx_packets_rate_{win}'] = df["rx_packets_rate"].rolling(window=win).min()
        df[f'min_tx_packets_rate_{win}'] = df["tx_packets_rate"].rolling(window=win).min()
    
    # 7. Peak-to-Average Ratio
    for win in window_sizes:
        df[f'rx_bytes_peak_to_avg_{win}'] = df[f'max_rx_bytes_rate_{win}'] / df[f'rx_bytes_ma_{win}']
        df[f'tx_bytes_peak_to_avg_{win}'] = df[f'max_tx_bytes_rate_{win}'] / df[f'tx_bytes_ma_{win}']
        df[f'rx_packets_peak_to_avg_{win}'] = df[f'max_rx_packets_rate_{win}'] / df[f'rx_packets_ma_{win}']
        df[f'tx_packets_peak_to_avg_{win}'] = df[f'max_tx_packets_rate_{win}'] / df[f'tx_packets_ma_{win}']
    
    # 8. Peak-to-Peak
    for win in window_sizes:
        df[f'rx_bytes_peak_to_peak_{win}'] = df[f'max_rx_bytes_rate_{win}'] - df[f'min_rx_bytes_rate_{win}']
        df[f'tx_bytes_peak_to_peak_{win}'] = df[f'max_tx_bytes_rate_{win}'] - df[f'min_tx_bytes_rate_{win}']
        df[f'rx_packets_peak_to_peak_{win}'] = df[f'max_rx_packets_rate_{win}'] - df[f'min_rx_packets_rate_{win}']
        df[f'tx_packets_peak_to_peak_{win}'] = df[f'max_tx_packets_rate_{win}'] - df[f'min_tx_packets_rate_{win}']
    
    # Clean up NaN, inf values
    df = df.replace([np.inf, -np.inf, np.nan], 0)
    
    # Remove time_diff column as done in Laura's notebook before saving
    if 'time_diff' in df.columns:
        df = df.drop('time_diff', axis=1)
    
    # Ensure index is preserved
    df.index = orig_index
    
    return df

def extract_features(csv_file, output_folder):
    """Extract features from CSV file"""
    try:
        # Load data and get path info
        df = pd.read_csv(csv_file)
        
        # Remove drops column if it exists to ensure no related features are calculated
        if 'drops' in df.columns:
            df = df.drop('drops', axis=1)
            
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
        rx_bytes = df['rx_bytes'].values if 'rx_bytes' in df.columns else np.zeros_like(time)
        tx_bytes = df['tx_bytes'].values if 'tx_bytes' in df.columns else np.zeros_like(time)
        rx_packets = df['rx_packets'].values if 'rx_packets' in df.columns else np.zeros_like(time)
        tx_packets = df['tx_packets'].values if 'tx_packets' in df.columns else np.zeros_like(time)
        
        # Create features dataframe with original metrics
        features_df = pd.DataFrame({
            # 'peak_score': score.astype(float),
            # 'peak_lt': peaks_lt,
            # 'peak_st': peaks_st,
            # 'filtered_throughput': filtered_tp,
            # 'ratio_st_lt': ratio,
            'rx_bytes': rx_bytes,
            'tx_bytes': tx_bytes,
            'rx_packets': rx_packets,
            'tx_packets': tx_packets,
            'queue_size': queue_size,
            'queue_exists': queue_exists
        })
        
        
        # Add Laura's features by creating a dataframe with all columns
        # so we can compute the new metrics properly
        full_df = df.copy()
        
        # Add Laura's features
        enhanced_df = add_laura_features(full_df, time_col)
        
        # Get columns that are in enhanced_df but not in the original df
        # and exclude any that contain "drops"
        laura_features = [col for col in enhanced_df.columns 
                          if col not in df.columns and "drops" not in col.lower()]
        
        # Add Laura's features to the features dataframe
        for feature in laura_features:
            features_df[feature] = enhanced_df[feature].values
        
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
        
        # Replace any remaining NaN or inf values with 0
        features_df = features_df.replace([np.inf, -np.inf, np.nan], 0)
        
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
        
        dfs.append(df)
    
    # Combine all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Remove any drops-related columns that might have slipped through
    drops_cols = [col for col in combined_df.columns if 'drops' in col.lower()]
    if drops_cols:
        combined_df = combined_df.drop(columns=drops_cols)
    
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