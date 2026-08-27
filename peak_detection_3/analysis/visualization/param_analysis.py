import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
from scipy.signal import medfilt

def peak_speed_detect(throughput, time, long_window_ms=2000, short_window_ms=200, peak_tolerance=0.1, count_threshold=5):
    if len(throughput) != len(time):
        raise ValueError("throughput and time arrays must have same length")
    if len(throughput) < 2 or np.all(throughput == 0):
        print("Warning: Throughput is zero or too short. No clipping detected.")
        return np.zeros(len(throughput), dtype=int), np.zeros(len(throughput)), throughput.copy(), np.zeros(len(throughput)), np.zeros(len(throughput))
    
    mean_dt = np.mean(np.diff(time))
    long_window_samples = max(1, int(long_window_ms / 1000 / mean_dt))
    short_window_samples = max(1, int(short_window_ms / 1000 / mean_dt))
    
    n_samples = len(throughput)
    filter_length = min(15, n_samples // 2 * 2 + 1)  # Ensure odd
    filtered_throughput = medfilt(throughput, filter_length)
    
    clipping_binary = np.zeros(n_samples, dtype=int)
    long_maxima = np.zeros(n_samples)
    short_counts = np.zeros(n_samples)
    short_peaks = np.zeros(n_samples)
    
    for long_start in range(0, n_samples, long_window_samples):
        long_end = min(long_start + long_window_samples, n_samples)
        long_window_data = throughput[long_start:long_end]
        if len(long_window_data) == 0:
            continue
        long_max = np.max(long_window_data)
        long_maxima[long_start:long_end] = long_max
        
        for short_start in range(long_start, long_end, short_window_samples):
            short_end = min(short_start + short_window_samples, long_end)
            short_bin_data = throughput[short_start:short_end]
            if len(short_bin_data) == 0:
                continue
            short_peak = np.max(short_bin_data)
            short_peaks[short_start:short_end] = short_peak
            tolerance_threshold = (1 - peak_tolerance) * long_max if long_max > 0 else 0
            near_peak_count = np.sum(short_bin_data >= tolerance_threshold)
            short_counts[short_start:short_end] = near_peak_count
            if long_max > 0 and near_peak_count >= count_threshold:
                clipping_binary[short_start:short_end] = 1
                print(f"Clipping detected at t={time[short_start]:.2f}s: long_max={long_max:.2f}, count={near_peak_count}")
    
    peak_ratio = np.zeros(n_samples)
    nonzero_mask = long_maxima > 0
    peak_ratio[nonzero_mask] = short_peaks[nonzero_mask] / long_maxima[nonzero_mask]
    print(f"Peak throughput: {np.max(throughput):.2f} Mbps")
    return (clipping_binary, long_maxima, filtered_throughput, 
            np.nan_to_num(peak_ratio, nan=0.0, posinf=0.0, neginf=0.0), 
            np.nan_to_num(short_counts, nan=0.0, posinf=0.0, neginf=0.0),
            np.nan_to_num(short_peaks, nan=0.0, posinf=0.0, neginf=0.0))

def calculate_throughput_matlab_method(df, byte_col='tx_bytes', time_col='relative_time'):
    """
    Calculate throughput using MATLAB-compatible method (consistent with feature extraction)
    MATLAB: 8 * ds(2:end, 5) / ds_interval / 1e6
    
    This method treats byte values as rates per interval, not cumulative counters.
    """
    if byte_col not in df.columns:
        print(f"Warning: {byte_col} not in columns")
        return np.zeros(len(df)), df[time_col].values if time_col in df.columns else np.arange(len(df), dtype=float)
    
    time_values = df[time_col].values if time_col in df.columns else np.arange(len(df), dtype=float)
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

def calculate_throughput(df, byte_col='tx_bytes', fallback_byte_col='rx_bytes', time_col='relative_time'):
    """
    FIXED: Now uses MATLAB-compatible method for consistency with feature extraction
    """
    try:
        print(f"🔧 Using MATLAB-compatible throughput calculation for {byte_col}")
        
        # Try primary byte column first
        if byte_col in df.columns and df[byte_col].nunique() > 1 and not df[byte_col].isna().all():
            throughput, time_values = calculate_throughput_matlab_method(df, byte_col, time_col)
            print(f"Using column: {byte_col}")
        elif fallback_byte_col in df.columns and df[fallback_byte_col].nunique() > 1 and not df[fallback_byte_col].isna().all():
            print(f"Falling back to {fallback_byte_col}")
            throughput, time_values = calculate_throughput_matlab_method(df, fallback_byte_col, time_col)
            print(f"Using column: {fallback_byte_col}")
        else:
            print(f"Error: No valid byte column found")
            time_values = df[time_col].values if time_col in df.columns else np.arange(len(df), dtype=float)
            return np.zeros(len(df)), time_values
        
        # MATLAB method debugging info
        print(f"MATLAB method - Mean interval: {np.mean(np.diff(time_values)) if len(time_values) > 1 else 'N/A':.6f}")
        print(f"First 5 throughput values (Mbps): {throughput[:5]}")
        print(f"MATLAB method - Throughput range: {throughput.min():.2f} - {throughput.max():.2f} Mbps")
        
        if np.all(throughput == 0) or np.all(np.isnan(throughput)):
            print(f"Warning: Throughput is all zero or NaN with MATLAB method")
        
        return throughput, time_values
        
    except Exception as e:
        print(f"Error in calculate_throughput: {str(e)}")
        time_values = df[time_col].values if time_col in df.columns else np.arange(len(df), dtype=float)
        return np.zeros(len(df)), time_values

os.makedirs('parameter_analysis_results', exist_ok=True)
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("notebook", font_scale=1.2)

try:
    data = pd.read_csv('combined_param_tests.csv')
    print(f"✅ Loaded parameter test data: {len(data)} records")
except FileNotFoundError:
    print("Error: combined_param_tests.csv not found.")
    exit()

data = data[data['direction'] == 'downstream']
print(f"Downstream records: {len(data)}")
print("🔧 Using MATLAB-compatible throughput calculation for consistency")

success_data = data[(data['status'] == 'success') & (data['accuracy'].notna())].copy()
print(f"Successful tests: {len(success_data)}")
if len(success_data) == 0:
    print("No successful downstream tests found.")
    exit()

success_data['file_name_clean'] = success_data['file_name'].str.replace('.csv', '', regex=False)
success_data['test_case'] = (success_data['file_name_clean'] + '_' + success_data['direction'] + '_' + 
                             success_data['subfolder'] + '_' + success_data['network_limit'].astype(str))

scenario_data = success_data[
    (success_data['file_name_clean'].str.startswith('100mbps')) & (success_data['network_limit'] == 50)
]
print(f"Test cases matching '100mbps' filename and network_limit=50: {len(scenario_data)}")

grouped_data = success_data.groupby(['short_window', 'long_window', 'peak_tolerance', 'count_threshold']).agg({
    'accuracy': ['mean', 'std', 'min', 'max', 'count'],
    'f1_score': 'mean'
}).reset_index()
grouped_data.columns = ['_'.join(col).strip() if col[1] else col[0] for col in grouped_data.columns.values]
grouped_data['accuracy_range'] = grouped_data['accuracy_max'] - grouped_data['accuracy_min']
grouped_data.rename(columns={'accuracy_count': 'occurrence_count'}, inplace=True)

# Graph 1: Parameter Performance
plt.figure(figsize=(14, 8))
scatter = plt.scatter(grouped_data['long_window'], grouped_data['short_window'], 
                      s=grouped_data['occurrence_count']*3, c=grouped_data['accuracy_mean'], 
                      cmap='viridis', alpha=0.8, edgecolors='k')
plt.colorbar(scatter, label='Average Accuracy')
plt.xlabel('Long Window (ms)')
plt.ylabel('Short Window (ms)')
plt.title('Parameter Combinations: Accuracy and Frequency (Downstream) - MATLAB Compatible')
sizes = sorted(grouped_data['occurrence_count'].unique())
for size in [min(sizes), np.percentile(sizes, 50), max(sizes)]:
    plt.scatter([], [], s=size*3, c='gray', alpha=0.6, label=f'Count: {int(size)}')
plt.legend(title='Occurrence Frequency')
top_5 = grouped_data.sort_values('accuracy_mean', ascending=False).head(5)
for _, row in top_5.iterrows():
    plt.annotate(f"Acc: {row['accuracy_mean']:.3f}\nPT: {row['peak_tolerance']}\nCT: {row['count_threshold']}",
                 (row['long_window'], row['short_window']), xytext=(10, 5), textcoords='offset points',
                 bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7), fontsize=9)
    plt.plot(row['long_window'], row['short_window'], 'ro', ms=12, mfc='none')
plt.tight_layout()
plt.savefig('parameter_analysis_results/1_parameter_performance.png', dpi=300)
plt.close()

# Graph 2: Accuracy Heatmap (Short vs Long Window)
plt.figure(figsize=(12, 8))
pivot = grouped_data.pivot_table(index='short_window', columns='long_window', values='accuracy_mean')
sns.heatmap(pivot, annot=True, cmap='viridis', fmt='.3f')
plt.title('Accuracy by Short and Long Window (Downstream) - MATLAB Compatible')
plt.xlabel('Long Window (ms)')
plt.ylabel('Short Window (ms)')
plt.tight_layout()
plt.savefig('parameter_analysis_results/2_accuracy_heatmap_windows.png', dpi=300)
plt.close()

# Graph 3: Accuracy Heatmap (Short Window vs Peak Tolerance)
plt.figure(figsize=(10, 8))
pivot = grouped_data.pivot_table(index='short_window', columns='peak_tolerance', values='accuracy_mean')
sns.heatmap(pivot, annot=True, cmap='viridis', fmt='.3f')
plt.title('Accuracy by Short Window and Peak Tolerance (Downstream) - MATLAB Compatible')
plt.xlabel('Peak Tolerance')
plt.ylabel('Short Window (ms)')
plt.tight_layout()
plt.savefig('parameter_analysis_results/3_accuracy_heatmap_short_tolerance.png', dpi=300)
plt.close()

# Graph 4: Accuracy Heatmap (Long Window vs Count Threshold)
plt.figure(figsize=(12, 8))
pivot = grouped_data.pivot_table(index='count_threshold', columns='long_window', values='accuracy_mean')
sns.heatmap(pivot, annot=True, cmap='viridis', fmt='.3f')
plt.title('Accuracy by Long Window and Count Threshold (Downstream) - MATLAB Compatible')
plt.xlabel('Long Window (ms)')
plt.ylabel('Count Threshold')
plt.tight_layout()
plt.savefig('parameter_analysis_results/4_accuracy_heatmap_long_count.png', dpi=300)
plt.close()

# Graph 5: Accuracy Heatmap (Peak Tolerance vs Count Threshold)
plt.figure(figsize=(10, 8))
pivot = grouped_data.pivot_table(index='count_threshold', columns='peak_tolerance', values='accuracy_mean')
sns.heatmap(pivot, annot=True, cmap='viridis', fmt='.3f')
plt.title('Accuracy by Peak Tolerance and Count Threshold (Downstream) - MATLAB Compatible')
plt.xlabel('Peak Tolerance')
plt.ylabel('Count Threshold')
plt.tight_layout()
plt.savefig('parameter_analysis_results/5_accuracy_heatmap_tolerances.png', dpi=300)
plt.close()

# Graph 6: Best Parameter Combinations
print("Finding best parameter combinations...")
best_params = []
for test in success_data['test_case'].unique():
    test_data = success_data[success_data['test_case'] == test]
    top = test_data.sort_values('accuracy', ascending=False).iloc[0]
    best_params.append({
        'test_case': test, 'short_window': top['short_window'], 'long_window': top['long_window'],
        'peak_tolerance': top['peak_tolerance'], 'count_threshold': top['count_threshold'],
        'max_accuracy': top['accuracy'], 'file_name': top['file_name'], 'subfolder': top['subfolder'],
        'network_limit': top['network_limit']
    })
best_params_df = pd.DataFrame(best_params)
param_counts = best_params_df.groupby(['short_window', 'long_window', 'peak_tolerance', 'count_threshold']).size().reset_index(name='count')
param_counts = param_counts.sort_values('count', ascending=False)
avg_acc = best_params_df.groupby(['short_window', 'long_window', 'peak_tolerance', 'count_threshold'])['max_accuracy'].mean().reset_index()
param_counts = pd.merge(param_counts, avg_acc, on=['short_window', 'long_window', 'peak_tolerance', 'count_threshold'])

fig, ax = plt.subplots(figsize=(14, 8))
top_n = min(15, len(param_counts))
norm = plt.Normalize(param_counts['max_accuracy'].min(), param_counts['max_accuracy'].max())
colors = plt.cm.viridis(norm(param_counts['max_accuracy'].head(top_n)))
bars = ax.bar(range(top_n), param_counts['count'].head(top_n), color=colors)
sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
plt.colorbar(sm, ax=ax, label='Accuracy')
for i, row in param_counts.head(top_n).iterrows():
    ax.text(i, row['count'] + 0.1, f"Acc: {row['max_accuracy']:.3f}", ha='center', fontsize=9, 
            bbox=dict(facecolor='white', alpha=0.8))
ax.set_xticks(range(top_n))
ax.set_xticklabels([f"SW:{row['short_window']}\nLW:{row['long_window']}\nPT:{row['peak_tolerance']}\nCT:{row['count_threshold']}" 
                    for _, row in param_counts.head(top_n).iterrows()], rotation=0, fontsize=8)
ax.set_xlabel('Parameter Combination')
ax.set_ylabel('Number of Test Cases Where Best')
ax.set_title('Top Parameter Combinations (Downstream) - MATLAB Compatible')
plt.tight_layout()
plt.savefig('parameter_analysis_results/6_best_combinations.png', dpi=300)
plt.close()

# Graph 7: Best Window Pairs
print("Analyzing window pairs...")
window_pairs = best_params_df.groupby(['short_window', 'long_window']).size().reset_index(name='count')
window_pairs = window_pairs.sort_values('count', ascending=False)
window_pairs = pd.merge(window_pairs, best_params_df.groupby(['short_window', 'long_window'])['max_accuracy'].mean().reset_index(), 
                        on=['short_window', 'long_window'])

fig, ax = plt.subplots(figsize=(12, 8))
top_n_pairs = min(10, len(window_pairs))
norm = plt.Normalize(window_pairs['max_accuracy'].min(), window_pairs['max_accuracy'].max())
colors = plt.cm.viridis(norm(window_pairs['max_accuracy'].head(top_n_pairs)))
bars = ax.bar(range(top_n_pairs), window_pairs['count'].head(top_n_pairs), color=colors)
plt.colorbar(plt.cm.ScalarMappable(cmap='viridis', norm=norm), ax=ax, label='Accuracy')
for i, row in window_pairs.head(top_n_pairs).iterrows():
    ax.text(i, row['count'] + 0.1, f"Acc: {row['max_accuracy']:.3f}", ha='center', fontsize=9, 
            bbox=dict(facecolor='white', alpha=0.8))
ax.set_xticks(range(top_n_pairs))
ax.set_xticklabels([f"({row['short_window']}, {row['long_window']})" for _, row in window_pairs.head(top_n_pairs).iterrows()], 
                   rotation=45, ha='right')
ax.set_xlabel('Window Pair (Short, Long)')
ax.set_ylabel('Number of Test Cases Where Best')
ax.set_title('Best Window Pairs (Downstream) - MATLAB Compatible')
plt.tight_layout()
plt.savefig('parameter_analysis_results/7_best_window_pairs.png', dpi=300)
plt.close()

# Graph 8: Accuracy vs Long Window
print("Creating Accuracy vs Long Window...")
top_short = param_counts['short_window'].unique()[:4]
top_tolerance = param_counts['peak_tolerance'].unique()[:3]
plt.figure(figsize=(16, 12))
for i, (sw, pt) in enumerate([(sw, pt) for sw in top_short for pt in top_tolerance], 1):
    filtered = grouped_data[(grouped_data['short_window'] == sw) & (grouped_data['peak_tolerance'] == pt)]
    if not filtered.empty:
        avg = filtered.groupby('long_window')['accuracy_mean'].mean().reset_index().sort_values('long_window')
        plt.subplot(len(top_short), len(top_tolerance), i)
        plt.plot(avg['long_window'], avg['accuracy_mean'], marker='o')
        plt.title(f'Short: {sw}ms, PT: {pt}')
        plt.xlabel('Long Window (ms)')
        plt.ylabel('Accuracy')
        max_point = avg.loc[avg['accuracy_mean'].idxmax()]
        plt.scatter(max_point['long_window'], max_point['accuracy_mean'], color='red', s=100, 
                    label=f'Max: {max_point["accuracy_mean"]:.3f}')
        plt.legend()
plt.suptitle('Accuracy vs Long Window - MATLAB Compatible', fontsize=16)
plt.tight_layout()
plt.savefig('parameter_analysis_results/8_accuracy_vs_long_window.png', dpi=300)
plt.close()

# Graph 9: Accuracy vs Peak Tolerance
print("Creating Accuracy vs Peak Tolerance...")
top_pairs = window_pairs.head(6)
plt.figure(figsize=(16, 12))
for i, row in enumerate(top_pairs.itertuples(), 1):
    filtered = grouped_data[(grouped_data['short_window'] == row.short_window) & (grouped_data['long_window'] == row.long_window)]
    if not filtered.empty:
        avg = filtered.groupby('peak_tolerance')['accuracy_mean'].mean().reset_index().sort_values('peak_tolerance')
        plt.subplot(2, 3, i)
        plt.plot(avg['peak_tolerance'], avg['accuracy_mean'], marker='o')
        plt.title(f'Short: {row.short_window}ms, Long: {row.long_window}ms')
        plt.xlabel('Peak Tolerance')
        plt.ylabel('Accuracy')
        max_point = avg.loc[avg['accuracy_mean'].idxmax()]
        plt.scatter(max_point['peak_tolerance'], max_point['accuracy_mean'], color='red', s=100, 
                    label=f'Max: {max_point["accuracy_mean"]:.3f}')
        plt.legend()
plt.suptitle('Accuracy vs Peak Tolerance - MATLAB Compatible', fontsize=16)
plt.tight_layout()
plt.savefig('parameter_analysis_results/9_accuracy_vs_peak_tolerance.png', dpi=300)
plt.close()

# Graph 10: Accuracy vs Count Threshold
print("Creating Accuracy vs Count Threshold...")
plt.figure(figsize=(16, 12))
for i, row in enumerate(top_pairs.itertuples(), 1):
    filtered = grouped_data[(grouped_data['short_window'] == row.short_window) & (grouped_data['long_window'] == row.long_window)]
    if not filtered.empty:
        avg = filtered.groupby('count_threshold')['accuracy_mean'].mean().reset_index().sort_values('count_threshold')
        plt.subplot(2, 3, i)
        plt.plot(avg['count_threshold'], avg['accuracy_mean'], marker='o')
        plt.title(f'Short: {row.short_window}ms, Long: {row.long_window}ms')
        plt.xlabel('Count Threshold')
        plt.ylabel('Accuracy')
        max_point = avg.loc[avg['accuracy_mean'].idxmax()]
        plt.scatter(max_point['count_threshold'], max_point['accuracy_mean'], color='red', s=100, 
                    label=f'Max: {max_point["accuracy_mean"]:.3f}')
        plt.legend()
plt.suptitle('Accuracy vs Count Threshold - MATLAB Compatible', fontsize=16)
plt.tight_layout()
plt.savefig('parameter_analysis_results/10_accuracy_vs_count_threshold.png', dpi=300)
plt.close()

# Graph 11: Peak Speed Detection Visualization
print("Creating Peak Speed Detection Visualization with MATLAB-compatible throughput...")
def get_csv_files(subfolder, folder='extracted_data'):
    folder_path = os.path.join(folder, subfolder)
    return [f for f in os.listdir(folder_path) if f.endswith('.csv')] if os.path.exists(folder_path) else []

valid_test_case = False
max_attempts = 5
attempt = 0
test_cases_tried = []

top_test_cases = scenario_data.sort_values('accuracy', ascending=False).head(max_attempts) if not scenario_data.empty else pd.DataFrame()
if top_test_cases.empty:
    print("No test cases with '100mbps' and network_limit=50. Trying any '100mbps'.")
    scenario_data = success_data[success_data['file_name_clean'].str.startswith('100mbps')]
    top_test_cases = scenario_data.sort_values('accuracy', ascending=False).head(max_attempts)

for _, test_case in top_test_cases.iterrows():
    attempt += 1
    test_case_id = test_case['test_case']
    test_cases_tried.append(test_case_id)
    
    short_window = test_case['short_window']
    long_window = test_case['long_window']
    peak_tolerance = test_case['peak_tolerance']
    count_threshold = test_case['count_threshold']
    file_name_clean = test_case['file_name'].replace('.csv', '')
    
    print(f"\n🔧 Attempt {attempt}/{max_attempts} - Test case: {test_case_id}")
    print(f"File name: {file_name_clean}\nSubfolder: {test_case['subfolder']}")
    print(f"Parameters: SW={short_window}ms, LW={long_window}ms, PT={peak_tolerance}, CT={count_threshold}")
    
    raw_data_file = os.path.join('extracted_data', test_case['subfolder'], test_case['file_name'])
    print(f"Loading: {raw_data_file}")
    
    try:
        raw_data = pd.read_csv(raw_data_file)
        print(f"Raw data shape: {raw_data.shape}")
    except Exception as e:
        print(f"Error loading file: {str(e)}")
        continue
    
    if 'tx_bytes' not in raw_data.columns and 'rx_bytes' not in raw_data.columns:
        print(f"Error: No tx_bytes or rx_bytes in {raw_data_file}")
        continue
    
    print("Raw data sample (first 5 rows):")
    columns = ['relative_time'] + (['tx_bytes'] if 'tx_bytes' in raw_data.columns else []) + (['rx_bytes'] if 'rx_bytes' in raw_data.columns else [])
    print(raw_data[[c for c in columns if c in raw_data.columns]].head())
    
    for col in ['tx_bytes', 'rx_bytes']:
        if col in raw_data.columns:
            stats = raw_data[col]
            print(f"{col} stats: min={stats.min()}, max={stats.max()}, mean={stats.mean():.2f}, std={stats.std():.2f}")
            if stats.isna().any():
                print(f"Warning: {col} contains NaNs. Filling with 0.")
                raw_data[col] = stats.fillna(0)
    
    time_col = 'relative_time' if 'relative_time' in raw_data.columns else \
               next((col for col in raw_data.columns if 'time' in col.lower()), None)
    if time_col is None:
        print("Warning: No time column. Creating index_time.")
        raw_data['index_time'] = np.linspace(0, len(raw_data) * 0.02, len(raw_data))
        time_col = 'index_time'
    
    time_values = raw_data[time_col].values
    if np.all(np.diff(time_values) <= 0):
        print(f"Error: Time not increasing in {raw_data_file}")
        continue
    
    print(f"Time range: {time_values.min():.2f} to {time_values.max():.2f} s")
    
    # Use MATLAB-compatible throughput calculation
    throughput, time = calculate_throughput(raw_data, time_col=time_col)
    print(f"Throughput samples: {len(throughput)}")
    
    if len(throughput) < 100 or np.all(throughput == 0) or np.all(np.isnan(throughput)):
        print(f"Error: Insufficient or zero/NaN throughput in {raw_data_file}")
        continue
    
    valid_test_case = True
    break

if not valid_test_case:
    print(f"No valid test cases after trying: {test_cases_tried}")
    subfolder = top_test_cases.iloc[0]['subfolder'] if not top_test_cases.empty else 'test_100'
    csv_files = get_csv_files(subfolder)
    downstream_files = [f for f in csv_files if 'downstream' in f and '100mbps' in f]
    for f in downstream_files[:max_attempts]:
        attempt += 1
        print(f"\n🔧 Fallback attempt {attempt}/{max_attempts} - File: {f}")
        raw_data_file = os.path.join('extracted_data', subfolder, f)
        try:
            raw_data = pd.read_csv(raw_data_file)
            if 'tx_bytes' not in raw_data.columns and 'rx_bytes' not in raw_data.columns:
                print(f"Error: No tx_bytes or rx_bytes in {f}")
                continue
            time_col = 'relative_time' if 'relative_time' in raw_data.columns else 'index_time'
            if time_col == 'index_time':
                raw_data['index_time'] = np.linspace(0, len(raw_data) * 0.02, len(raw_data))
            for col in ['tx_bytes', 'rx_bytes']:
                if col in raw_data.columns and raw_data[col].isna().any():
                    raw_data[col] = raw_data[col].fillna(0)
            throughput, time = calculate_throughput(raw_data, time_col=time_col)
            if len(throughput) < 100 or np.all(throughput == 0) or np.all(np.isnan(throughput)):
                print(f"Error: Insufficient or zero/NaN throughput in {f}")
                continue
            valid_test_case = True
            file_name_clean = f.replace('.csv', '')
            short_window, long_window, peak_tolerance, count_threshold = 200, 2000, 0.1, 5
            test_case = {'accuracy': 0.0, 'network_limit': 50}
            break
        except Exception as e:
            print(f"Error loading {f}: {str(e)}")
            continue

if not valid_test_case:
    print("Error: No valid downstream data found.")
    exit()

network_capacity = round(test_case['network_limit'] * 1.15 if pd.notna(test_case['network_limit']) else 57.5)
print(f"Network capacity: {network_capacity} Mbps")

try:
    clipping_binary, long_maxima, filtered_throughput, peak_ratio, short_counts, short_peaks = peak_speed_detect(
        throughput, time, long_window_ms=long_window, short_window_ms=short_window,
        peak_tolerance=peak_tolerance, count_threshold=count_threshold)
    print(f"✅ Clipping regions detected: {np.sum(clipping_binary)} samples")
except Exception as e:
    print(f"Error in peak_speed_detect: {str(e)}")
    exit()

plt.figure(figsize=(14, 8))
plt.plot(time, throughput, label='Downstream Throughput (MATLAB method)', color='red', alpha=0.8)
plt.plot([min(time), max(time)], [network_capacity, network_capacity], 
         label=f'Network Capacity ({network_capacity} Mbps)', color='orange', linestyle='--')

mean_dt = np.mean(np.diff(time))
long_window_samples = max(1, int(long_window / 1000 / mean_dt))
short_window_samples = max(1, int(short_window / 1000 / mean_dt))

first_clipping_bin = True
for long_start in range(0, len(throughput), long_window_samples):
    long_end = min(long_start + long_window_samples, len(throughput))
    long_max = long_maxima[long_start]
    if long_max > 0:
        plt.fill_between(time[long_start:long_end], 0, long_max, color='red', alpha=0.2, 
                         step='post', label='Long Window' if long_start == 0 else "")
        for short_start in range(long_start, long_end, short_window_samples):
            short_end = min(short_start + short_window_samples, long_end)
            if clipping_binary[short_start] == 1:
                count = short_counts[short_start]
                mid_time = (time[short_start] + time[min(short_end-1, len(time)-1)]) / 2
                plt.text(mid_time, long_max * 1.05, f'N={int(count)}', ha='center', fontsize=10, 
                         bbox=dict(facecolor='white', alpha=0.8))
                plt.fill_between(time[short_start:short_end], 0, long_max, color='green', alpha=0.3, 
                                 step='post', label='Clipping Short Bin' if first_clipping_bin else "")
                first_clipping_bin = False

plt.xlabel('Time (s)')
plt.ylabel('Throughput (Mbps)')
plt.title(f'Downstream Peak Detection with MATLAB-Compatible Throughput: {file_name_clean}\n'
          f'SW={short_window}ms, LW={long_window}ms, PT={peak_tolerance}, CT={count_threshold}, Acc={test_case["accuracy"]:.3f}')
plt.legend(loc='upper right')
plt.grid(True)
plt.tight_layout()
plt.savefig('parameter_analysis_results/11_peak_graph.png', dpi=300)
plt.close()

print("\n🔧 Peak Speed Detection Statistics (MATLAB-Compatible):")
print(f"Total unique combinations: {len(grouped_data)}")
print(f"Max occurrence count: {grouped_data['occurrence_count'].max()}")
print(f"Avg occurrence count: {grouped_data['occurrence_count'].mean():.1f}")

top_10 = grouped_data.sort_values('accuracy_mean', ascending=False).head(10)
print("\nTop 10 Parameter Combinations:")
print(top_10[['short_window', 'long_window', 'peak_tolerance', 'count_threshold', 'accuracy_mean', 'occurrence_count']])

most_frequent = grouped_data.sort_values('occurrence_count', ascending=False).head(10)
print("\nMost Frequent Combinations:")
print(most_frequent[['short_window', 'long_window', 'peak_tolerance', 'count_threshold', 'occurrence_count', 'accuracy_mean']])

consistent = grouped_data[(grouped_data['occurrence_count'] >= 3) & (grouped_data['accuracy_mean'].notna())].sort_values(
    ['accuracy_mean', 'accuracy_std'], ascending=[False, True]).head(20)
print("\nConsistent High-Performing Combinations:")
print(consistent[['short_window', 'long_window', 'peak_tolerance', 'count_threshold', 'accuracy_mean', 'accuracy_std', 'accuracy_range', 'occurrence_count']])

print("\nBest Combinations Across Test Cases:")
print(param_counts[['short_window', 'long_window', 'peak_tolerance', 'count_threshold', 'count', 'max_accuracy']].head(10))

print("\nBest Window Pairs Across Test Cases:")
print(window_pairs[['short_window', 'long_window', 'count', 'max_accuracy']].head(10))

top_10.to_csv('parameter_analysis_results/top_10_by_accuracy.csv', index=False)
most_frequent.to_csv('parameter_analysis_results/most_frequent_combinations.csv', index=False)
consistent.to_csv('parameter_analysis_results/consistent_combinations.csv', index=False)
param_counts.to_csv('parameter_analysis_results/best_combinations_by_test.csv', index=False)
window_pairs.to_csv('parameter_analysis_results/best_window_pairs.csv', index=False)

print("\nParameter Distribution:")
print("Short Window:", success_data['short_window'].value_counts().sort_index())
print("Long Window:", success_data['long_window'].value_counts().sort_index())
print("Peak Tolerance:", success_data['peak_tolerance'].value_counts().sort_index())
print("Count Threshold:", success_data['count_threshold'].value_counts().sort_index())

print("\nFailure Analysis:")
failures = data[data['status'] != 'success']
print(f"Total failed tests: {len(failures)}")
if len(failures) > 0:
    print("Failure reasons:", failures['status'].value_counts())
    print("Common count_thresholds in failures:", failures['count_threshold'].value_counts().head())

print("\n✅ Analysis complete with MATLAB-compatible throughput calculation!")
print("🔧 Results saved to 'parameter_analysis_results' - now consistent with feature extraction")
print("📊 Visualization saved as '11_peak_graph.png'")