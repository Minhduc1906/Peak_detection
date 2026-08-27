import numpy as np
import pandas as pd
import os
import glob
import json
import re

# Define folder paths
LOG_FOLDER = 'raw_data'
OUTPUT_FOLDER = 'extracted_data'
TEST_CONFIGS_DIR = 'completed_experiments'  # Directory with test configuration files
VM_METRICS_FOLDER = 'vm_metrics'
PEAK_THRESHOLD = 100

def ensure_folder_exists(folder_path):
    """Create folder if it doesn't exist"""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def extract_test_name_from_log(log_file):
    """Extract test name from log file path"""
    # Example: raw_data/test_basic_web_browsing_1limited_0unlimited_normal_load_client1_s2-r1.log
    base_name = os.path.basename(log_file)
    # Extract the test name part before interface details
    match = re.match(r'(test_[^_]+(?:_[^_]+)*?)_(?:client|server|s2)', base_name)
    if match:
        return match.group(1)
    return None

# def extract_test_name_from_log(log_file):
#     """Extract test name from log file path"""
#     # Example: 50mbps_4k_s2-mgmt_s2-r1.log or 100mbps_4k_s2-mgmt_s2-r1.log
#     base_name = os.path.basename(log_file)
#     # Extract the test name part before _s2-mgmt or similar suffixes
#     match = re.match(r'([^_]+(?:_[^_]+)*?)_(?:s2-mgmt|s2-r1|s2)', base_name)
#     if match:
#         return match.group(1)
#     return None

def extract_network_capacity(folder_path):
    """Extract network capacity from folder name if it ends with a number"""
    # Extract the directory name from the path
    dir_name = os.path.basename(os.path.normpath(folder_path))
    
    # Check if the directory name ends with a number
    match = re.search(r'_(\d+)$', dir_name)
    if match:
        return int(match.group(1))
    # Default to 100 if no match is found
    return 100

def get_experiment_duration(test_name):
    """Get experiment duration from test configuration file"""
    default_duration = None  # No default duration - process all data
    
    # Try to find matching configuration file
    config_path = os.path.join(TEST_CONFIGS_DIR, f"{test_name}.json")
    if not os.path.exists(config_path):
        # Try searching for all config files to find a partial match
        for config_file in glob.glob(os.path.join(TEST_CONFIGS_DIR, "*.json")):
            if test_name in config_file:
                config_path = config_file
                break
        else:
            print(f"Warning: No configuration file found for {test_name}, processing all available data")
            return default_duration
    
    try:
        with open(config_path, 'r') as f:
            test_config = json.load(f)
        
        # Find the latest start time and its duration
        latest_start_time = 0
        latest_duration = 0
        
        for test in test_config:
            # Extract start time
            start_time = test.get('host1', {}).get('start_time', 0)
            
            # Extract duration from command string
            cmd = test.get('host1', {}).get('cmd', '')
            duration = 60  # Default test duration
            
            # Look for -t parameter in iperf3 commands
            if 'iperf3' in cmd:
                parts = cmd.split()
                if '-t' in parts:
                    try:
                        duration = int(parts[parts.index('-t') + 1])
                    except (ValueError, IndexError):
                        pass
            
            # For D-ITG commands (-t is in milliseconds)
            if 'ITGSend' in cmd:
                parts = cmd.split()
                if '-t' in parts:
                    try:
                        duration = int(parts[parts.index('-t') + 1]) / 1000  # Convert ms to seconds
                    except (ValueError, IndexError):
                        pass
            
            # Check if this is the latest ending test
            end_time = start_time + duration
            if end_time > latest_start_time + latest_duration:
                latest_start_time = start_time
                latest_duration = duration
        
        # Add buffer time (20 seconds)
        total_duration = latest_start_time + latest_duration + 20
        
        # Return the calculated duration as a guideline, but still process all data
        print(f"Calculated experiment duration: {total_duration} seconds")
        return None  # Still process all data
    
    except Exception as e:
        print(f"Error calculating experiment duration for {test_name}: {e}")
        return default_duration

def find_log_pairs(base_folder=LOG_FOLDER):
    """
    Find pairs of log files ending with -r1.log and -r2.log in all subfolders.
    Returns a dictionary of pairs with their subfolder and base names as keys.
    """
    pairs = {}
    
    # Walk through all subfolders
    for root, _, _ in os.walk(base_folder):
        r1_files = glob.glob(os.path.join(root, '*-r1.log'))
        r2_files = glob.glob(os.path.join(root, '*-r2.log'))
        
        for r1_file in r1_files:
            base_name = r1_file[:-7]
            r2_file = f"{base_name}-r2.log"
            if r2_file in r2_files:
                # Get relative path from raw_data folder
                rel_path = os.path.relpath(root, base_folder)
                if rel_path == '.':
                    rel_path = ''
                    
                # Create key that includes subfolder path
                key = os.path.join(rel_path, os.path.basename(base_name))
                pairs[key] = (r1_file, r2_file)
    
    return pairs

def process_log_to_df(input_log, network_capacity=None, time_threshold=None):
    """
    Process log file to DataFrame with format matching the first file.
    """
    # Read the data, skipping the first line (comments)
    try:
        df = pd.read_csv(input_log, delimiter=' ', skiprows=1, header=None)
    except pd.errors.EmptyDataError:
        print(f"Error: File {input_log} is empty or cannot be read")
        return pd.DataFrame()
        
    # Check if file has enough data
    if len(df) <= 1:
        print(f"Warning: Log file {input_log} has insufficient data (only {len(df)} rows)")
        return pd.DataFrame()
    
    # Define the column names
    columns = [
        'time',
        'rx_packets',
        'rx_bytes',
        'tx_packets',
        'tx_bytes',
        'qdisc',
        'bytes',
        'packets',
        'drops',
        'overlimits',
        'BACKLOG'
    ]
    
    # Assign column names to the DataFrame
    df.columns = columns
    
    # Calculate relative time starting from second row
    start_time = df['time'].iloc[1]
    df = df.iloc[1:].copy()  # Remove first row and make a copy
    df['relative_time'] = df['time'] - start_time
    
    # Calculate buffer metrics with the queue_threshold from the first file (10 instead of 150)
    df['queue_size'] = df['BACKLOG'].diff()
    df['queue_exists'] = (df['queue_size'] > PEAK_THRESHOLD).astype(int)
    
    # Add network capacity column if available
    if network_capacity is not None:
        df['network_capacity'] = network_capacity
    
    # Reorder columns to match the first file's format, plus the new network_capacity column
    columns = ['time', 'relative_time', 'rx_packets', 'rx_bytes', 'tx_packets', 'tx_bytes', 
             'bytes', 'packets', 'drops', 'overlimits', 'BACKLOG', 'queue_size', 'queue_exists']
    
    # Add network_capacity at the end if it exists
    if 'network_capacity' in df.columns:
        columns.append('network_capacity')
    
    df = df[columns]
    
    # Drop the 'qdisc' column since it's not in the first file
    if 'qdisc' in df.columns:
        df = df.drop('qdisc', axis=1)
    
    # Print summary statistics
    print(f"Processed {len(df)} rows, time range: 0 to {df['relative_time'].max():.1f} seconds")
    
    return df

def process_all_logs():
    """
    Process all log files from all subfolders and save results to corresponding
    subfolders in extracted_data with format matching the first file.
    """
    ensure_folder_exists(OUTPUT_FOLDER)
    ensure_folder_exists(TEST_CONFIGS_DIR)  # Ensure config dir exists
    log_pairs = find_log_pairs()
    
    processed_files = []
    for base_name, (downstream_log, upstream_log) in log_pairs.items():
        try:
            # Extract test name to find its config
            test_name = extract_test_name_from_log(downstream_log)
            
            # Get experiment duration from config (for information only)
            if test_name:
                _ = get_experiment_duration(test_name)
            
            # Extract network capacity from folder name
            log_folder = os.path.dirname(downstream_log)
            network_capacity = extract_network_capacity(log_folder)
            # Check if it's the default value or an extracted value
            match = re.search(r'_(\d+)$', os.path.basename(os.path.normpath(log_folder)))
            if match:
                print(f"Extracted network capacity: {network_capacity} from folder: {log_folder}")
            else:
                print(f"No network capacity found in folder name, using default value: {network_capacity}")
                
            # Create output subfolder if needed
            output_subfolder = os.path.join(OUTPUT_FOLDER, os.path.dirname(base_name))
            ensure_folder_exists(output_subfolder)
            
            # Process logs to DataFrames with format matching the first file
            ds_df = process_log_to_df(downstream_log, network_capacity)
            us_df = process_log_to_df(upstream_log, network_capacity)
            
            # Save processed data maintaining subfolder structure
            ds_output = os.path.join(OUTPUT_FOLDER, f'{base_name}_downstream.csv')
            us_output = os.path.join(OUTPUT_FOLDER, f'{base_name}_upstream.csv')
            
            ds_df.to_csv(ds_output, index=False)
            us_df.to_csv(us_output, index=False)
            
            processed_files.extend([ds_output, us_output])
            print(f"Processed {base_name} - Files saved: {ds_output}, {us_output}")
            print(f"  Downstream: {len(ds_df)} samples up to {ds_df['relative_time'].max():.1f}s")
            print(f"  Upstream: {len(us_df)} samples up to {us_df['relative_time'].max():.1f}s")
            
        except Exception as e:
            print(f"Error processing {base_name}: {str(e)}")
    
    return processed_files

if __name__ == "__main__":
    processed_files = process_all_logs()
    print(f"\nProcessing complete. {len(processed_files)} files generated in {OUTPUT_FOLDER}/")