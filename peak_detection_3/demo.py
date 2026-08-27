import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

# Set style for better-looking plots
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'

def load_and_process_data(csv_file_path):
    """Load CSV data and extract relevant columns"""
    df = pd.read_csv(csv_file_path)
    
    # Focus on tx_packets column as requested
    df = df[['time', 'relative_time', 'tx_packets']].dropna()
    
    print(f"Loaded {len(df)} data points")
    print(f"Time span: {df['time'].max() - df['time'].min():.1f} seconds")
    print(f"TX packets range: {df['tx_packets'].min()} - {df['tx_packets'].max()}")
    
    return df

def calculate_long_term_maxima(df, window_size_seconds=60):
    """
    Step 1: Calculate maximum throughput in configurable long time windows
    """
    results = []
    time_start = df['time'].min()
    time_end = df['time'].max()
    time_span = time_end - time_start
    
    num_windows = int(time_span // window_size_seconds)
    
    for i in range(num_windows):
        window_start = time_start + i * window_size_seconds
        window_end = window_start + window_size_seconds
        
        # Filter data for this window
        window_data = df[(df['time'] >= window_start) & (df['time'] < window_end)]
        
        if len(window_data) > 0:
            results.append({
                'window_index': int(i),  # Ensure integer type
                'window_start': window_start,
                'window_end': window_end,
                'window_center': window_start + window_size_seconds / 2,
                'max_tx_packets': window_data['tx_packets'].max(),
                'avg_tx_packets': window_data['tx_packets'].mean(),
                'sample_count': len(window_data),
                'window_size': window_size_seconds
            })
    
    return pd.DataFrame(results)

def analyze_short_bins(df, long_windows, bin_size_seconds=2, tolerance_percent=15):
    """
    Step 2: Subdivide long windows into shorter bins and count samples 
    within tolerance of the long-term maximum
    """
    results = []
    
    for _, window in long_windows.iterrows():
        # Get data for this long window
        window_data = df[(df['time'] >= window['window_start']) & 
                        (df['time'] < window['window_end'])]
        
        if len(window_data) == 0:
            continue
            
        # Calculate tolerance threshold
        tolerance_threshold = window['max_tx_packets'] * (1 - tolerance_percent / 100)
        
        # Create short bins within this window
        num_bins = int(window['window_size'] // bin_size_seconds)
        
        for bin_idx in range(num_bins):
            bin_start = window['window_start'] + bin_idx * bin_size_seconds
            bin_end = bin_start + bin_size_seconds
            
            # Get data for this bin
            bin_data = window_data[(window_data['time'] >= bin_start) & 
                                  (window_data['time'] < bin_end)]
            
            if len(bin_data) > 0:
                # Count samples near maximum
                samples_near_max = len(bin_data[bin_data['tx_packets'] >= tolerance_threshold])
                
                results.append({
                    'window_index': int(window['window_index']),  # Ensure integer type
                    'bin_index': bin_idx,
                    'bin_start': bin_start,
                    'bin_end': bin_end,
                    'bin_center': bin_start + bin_size_seconds / 2,
                    'window_max_tx_packets': window['max_tx_packets'],
                    'tolerance_threshold': tolerance_threshold,
                    'samples_near_max': samples_near_max,
                    'total_samples': len(bin_data),
                    'percentage_near_max': (samples_near_max / len(bin_data)) * 100,
                    'avg_tx_packets': bin_data['tx_packets'].mean(),
                    'max_tx_packets': bin_data['tx_packets'].max(),
                    'bin_size': bin_size_seconds
                })
    
    return pd.DataFrame(results)

def classify_bins(bin_results, clipping_threshold=3):
    """
    Step 3: Binary classification - classify bins with sample counts 
    exceeding threshold as exhibiting "clipping" behavior
    """
    bin_results = bin_results.copy()
    bin_results['is_clipping'] = bin_results['samples_near_max'] >= clipping_threshold
    bin_results['clipping_score'] = bin_results['samples_near_max']
    
    return bin_results

def create_figure_1_long_term_maximum(df, long_windows):
    """Figure 1: Long-term maximum calculation with window divisions - Focus on 100-200s"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    
    # Focus on seconds 100-200
    focus_start = 100
    focus_end = 200
    focused_df = df[(df['relative_time'] >= focus_start) & (df['relative_time'] <= focus_end)]
    
    if len(focused_df) == 0:
        print("Warning: No data in the 100-200 second range")
        focused_df = df
        focus_start = 0
        focus_end = df['relative_time'].max()
    
    # Plot raw TX packets data
    ax.plot(focused_df['relative_time'], focused_df['tx_packets'], 'b-', linewidth=1, alpha=0.8, label='TX Packets')
    
    # Add window divisions and maximum lines
    colors = ['lightcoral', 'lightblue', 'lightgreen', 'lightyellow', 'lightpink']
    max_legend_added = False
    
    for i, (_, window) in enumerate(long_windows.iterrows()):
        window_start_rel = window['window_start'] - df['time'].min()
        window_end_rel = window['window_end'] - df['time'].min()
        
        # Skip windows that don't overlap with focus range
        if window_end_rel < focus_start or window_start_rel > focus_end:
            continue
        
        # Calculate overlap with focus range
        overlap_start = max(window_start_rel, focus_start)
        overlap_end = min(window_end_rel, focus_end)
        
        if overlap_start < overlap_end:
            # Add colored background for each window
            color = colors[i % len(colors)]
            ax.axvspan(overlap_start, overlap_end, alpha=0.3, color=color,
                      label='Long Windows' if i == 0 else "")
            
            # Add maximum line for this window
            ax.axhline(y=window['max_tx_packets'], xmin=(overlap_start-focus_start)/(focus_end-focus_start), 
                      xmax=(overlap_end-focus_start)/(focus_end-focus_start), 
                      color='red', linewidth=3, alpha=0.8,
                      label='Long Window Maximum' if not max_legend_added else "")
            max_legend_added = True
            
            # Add window label
            ax.text(overlap_start + (overlap_end - overlap_start)/2, 
                   ax.get_ylim()[1] * 0.95, f'W-{int(window["window_index"])+1}', 
                   ha='center', va='top', fontweight='bold', fontsize=10)
            
            # Add vertical lines for window boundaries in focus range
            if overlap_start > focus_start:
                ax.axvline(x=overlap_start, color='black', linestyle='--', alpha=0.6)
            if overlap_end < focus_end:
                ax.axvline(x=overlap_end, color='black', linestyle='--', alpha=0.6)
    
    # Set focus range
    ax.set_xlim(focus_start, focus_end)
    
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('TX Packets', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')  # Changed from 'lower left' to 'upper right'
    
    return fig

def create_figure_2_short_term_bins(df, long_windows, bin_results, tolerance_percent):
    """Figure 2: Short-term bin analysis with tolerance regions - Focus on 100-200s"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    
    # Focus on seconds 100-200
    focus_start = 100
    focus_end = 200
    focused_df = df[(df['relative_time'] >= focus_start) & (df['relative_time'] <= focus_end)]
    
    if len(focused_df) == 0:
        print("Warning: No data in the 100-200 second range")
        focused_df = df
        focus_start = 0
        focus_end = df['relative_time'].max()
    
    # Plot raw TX packets data
    ax.plot(focused_df['relative_time'], focused_df['tx_packets'], 'b-', linewidth=1, alpha=0.6, label='TX Packets')
    
    # Colors for different windows
    window_colors = ['lightcoral', 'lightblue', 'lightgreen', 'lightyellow', 'lightpink']
    
    # Process all windows first to ensure we add legend items
    windows_in_range = []
    for window_idx in bin_results['window_index'].unique():
        window_idx = int(window_idx)
        window_info = long_windows.iloc[window_idx]
        window_start_rel = window_info['window_start'] - df['time'].min()
        window_end_rel = window_info['window_end'] - df['time'].min()
        
        # Check if window overlaps with focus range
        if not (window_end_rel < focus_start or window_start_rel > focus_end):
            windows_in_range.append(window_idx)
    
    for i, window_idx in enumerate(windows_in_range):
        window_bins = bin_results[bin_results['window_index'] == window_idx]
        window_info = long_windows.iloc[window_idx]
        
        window_start_rel = window_info['window_start'] - df['time'].min()
        window_end_rel = window_info['window_end'] - df['time'].min()
        
        # Calculate overlap with focus range
        overlap_start = max(window_start_rel, focus_start)
        overlap_end = min(window_end_rel, focus_end)
        
        # Window background
        color = window_colors[window_idx % len(window_colors)]
        ax.axvspan(overlap_start, overlap_end, alpha=0.2, color=color,
                  label='Long Windows' if i == 0 else "")
        
        # Add tolerance zone as colored rectangle
        tolerance_threshold = window_info['max_tx_packets'] * (1 - tolerance_percent / 100)
        window_max = window_info['max_tx_packets']
        
        if window_max > tolerance_threshold:
            # Create tolerance zone as filled rectangle (from threshold to maximum)
            rect = plt.Rectangle((overlap_start, tolerance_threshold), 
                               overlap_end - overlap_start, 
                               window_max - tolerance_threshold,
                               facecolor='darkorange', alpha=0.6, 
                               label=f'{tolerance_percent}% Tolerance Zone' if i == 0 else "")
            ax.add_patch(rect)
        
        # Add maximum line for reference
        ax.axhline(y=window_max, xmin=(overlap_start-focus_start)/(focus_end-focus_start), 
                  xmax=(overlap_end-focus_start)/(focus_end-focus_start), 
                  color='red', linewidth=3, alpha=0.8,
                  label='Long Window Maximum' if i == 0 else "")
        
        # Filter bins for this window and focus range
        focused_bins = window_bins[
            (window_bins['bin_center'] - df['time'].min() >= focus_start) &
            (window_bins['bin_center'] - df['time'].min() <= focus_end)
        ]
        
        # Add bin divisions and short bin labeling
        first_bin = True
        for _, bin_info in focused_bins.iterrows():
            bin_start_rel = bin_info['bin_start'] - df['time'].min()
            bin_end_rel = bin_info['bin_end'] - df['time'].min()
            
            # Vertical lines for bin boundaries (with label for legend)
            ax.axvline(x=bin_start_rel, color='gray', linestyle=':', alpha=0.5,
                      label='Short Bin Boundaries' if i == 0 and first_bin else "")
            first_bin = False
            
            # Highlight bins with high percentage near max
            if bin_info['percentage_near_max'] > 50:
                ax.axvspan(bin_start_rel, bin_end_rel, alpha=0.4, color='yellow')
                
            # Add bin sample count annotation
            if bin_info['samples_near_max'] > 0:
                ax.text(bin_start_rel + (bin_end_rel - bin_start_rel)/2, 
                       tolerance_threshold + (window_info['max_tx_packets'] - tolerance_threshold)/2,
                       f"N={bin_info['samples_near_max']}", 
                       ha='center', va='center', fontsize=8, 
                       bbox=dict(boxstyle="round,pad=0.1", facecolor='white', alpha=0.8))
        
        # Window label
        ax.text(overlap_start + (overlap_end - overlap_start)/2, 
               ax.get_ylim()[1] * 0.95, f'W-{window_idx+1}', 
               ha='center', va='top', fontweight='bold', fontsize=10)
    
    # Set focus range
    ax.set_xlim(focus_start, focus_end)
    
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('TX Packets', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower left')  # Changed from 'upper right' to 'lower left'
    
    return fig

def create_figure_3_binary_classification(df, long_windows, classified_bins, clipping_threshold, tolerance_percent):
    """Figure 3: Binary classification - similar to Figure 2 but highlighting clipping bins"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    
    # Focus on seconds 100-200
    focus_start = 100
    focus_end = 200
    focused_df = df[(df['relative_time'] >= focus_start) & (df['relative_time'] <= focus_end)]
    
    if len(focused_df) == 0:
        print("Warning: No data in the 100-200 second range")
        focused_df = df
        focus_start = 0
        focus_end = df['relative_time'].max()
    
    # Plot raw TX packets data for focused range
    ax.plot(focused_df['relative_time'], focused_df['tx_packets'], 'b-', linewidth=1, alpha=0.6, label='TX Packets')
    
    # Colors for different windows
    window_colors = ['lightcoral', 'lightblue', 'lightgreen', 'lightyellow', 'lightpink']
    tolerance_legend_added = False
    max_legend_added = False
    clipping_legend_added = False
    short_bin_legend_added = False
    
    # Process windows similar to Figure 2
    for window_idx in classified_bins['window_index'].unique():
        window_idx = int(window_idx)  # Ensure integer type
        window_bins = classified_bins[classified_bins['window_index'] == window_idx]
        window_info = long_windows.iloc[window_idx]
        
        window_start_rel = window_info['window_start'] - df['time'].min()
        window_end_rel = window_info['window_end'] - df['time'].min()
        
        # Skip windows that don't overlap with focus range
        if window_end_rel < focus_start or window_start_rel > focus_end:
            continue
        
        # Calculate overlap with focus range
        overlap_start = max(window_start_rel, focus_start)
        overlap_end = min(window_end_rel, focus_end)
        
        if overlap_start < overlap_end:
            # Window background (same as Figure 2)
            color = window_colors[window_idx % len(window_colors)]
            ax.axvspan(overlap_start, overlap_end, alpha=0.2, color=color,
                      label='Long Windows' if window_idx == min(classified_bins['window_index'].unique()) else "")
            
            # Add tolerance zone as colored rectangle (same as Figure 2)
            tolerance_threshold = window_info['max_tx_packets'] * (1 - tolerance_percent / 100)
            window_max = window_info['max_tx_packets']
            
            if window_max > tolerance_threshold:
                # Create tolerance zone as filled rectangle
                rect = plt.Rectangle((overlap_start, tolerance_threshold), 
                                   overlap_end - overlap_start, 
                                   window_max - tolerance_threshold,
                                   facecolor='darkorange', alpha=0.6, 
                                   label=f'{tolerance_percent}% Tolerance Zone' if not tolerance_legend_added else "")
                ax.add_patch(rect)
                tolerance_legend_added = True
            
            # Add maximum line for reference (same as Figure 2)
            ax.axhline(y=window_max, xmin=(overlap_start-focus_start)/(focus_end-focus_start), 
                      xmax=(overlap_end-focus_start)/(focus_end-focus_start), 
                      color='red', linewidth=3, alpha=0.8,
                      label='Long Window Maximum' if not max_legend_added else "")
            max_legend_added = True
            
            # Filter bins for this window and focus range
            focused_bins = window_bins[
                (window_bins['bin_center'] - df['time'].min() >= focus_start) &
                (window_bins['bin_center'] - df['time'].min() <= focus_end)
            ]
            
            # Add bin divisions (same as Figure 2)
            for bin_idx, (_, bin_info) in enumerate(focused_bins.iterrows()):
                bin_start_rel = bin_info['bin_start'] - df['time'].min()
                bin_end_rel = bin_info['bin_end'] - df['time'].min()
                
                # Vertical lines for bin boundaries (with label for legend) - only add label once
                ax.axvline(x=bin_start_rel, color='gray', linestyle=':', alpha=0.5,
                          label='Short Bin Boundaries' if not short_bin_legend_added else "")
                short_bin_legend_added = True
                
                # NEW: Highlight clipping bins (N >= threshold) with red overlay
                if bin_info['is_clipping']:
                    # Red overlay for clipping bins (the key difference from Figure 2)
                    rect = plt.Rectangle((bin_start_rel, 0), 
                                       bin_end_rel - bin_start_rel, 
                                       window_max,
                                       facecolor='red', alpha=0.5, 
                                       label='Clipping Bins (N≥3)' if not clipping_legend_added else "")
                    ax.add_patch(rect)
                    clipping_legend_added = True
                
                # Add bin sample count annotation (same as Figure 2)
                if bin_info['samples_near_max'] > 0:
                    # Use different colors for clipping vs normal bins
                    text_color = 'white' if bin_info['is_clipping'] else 'black'
                    bg_color = 'darkred' if bin_info['is_clipping'] else 'white'
                    
                    ax.text(bin_start_rel + (bin_end_rel - bin_start_rel)/2, 
                           tolerance_threshold + (window_max - tolerance_threshold)/2,
                           f"N={int(bin_info['samples_near_max'])}", 
                           ha='center', va='center', fontsize=8, color=text_color,
                           bbox=dict(boxstyle="round,pad=0.1", facecolor=bg_color, alpha=0.9))
            
            # Window label
            ax.text(overlap_start + (overlap_end - overlap_start)/2, 
                   ax.get_ylim()[1] * 0.95, f'W-{window_idx+1}', 
                   ha='center', va='top', fontweight='bold', fontsize=10)
    
    # Set focus range
    ax.set_xlim(focus_start, focus_end)
    
    # Let matplotlib auto-scale y-axis (same as Figures 1 and 2)
    
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('TX Packets', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower left')  # Changed from 'upper right' to avoid overlap
    
    return fig

def print_analysis_summary(df, long_windows, bin_results, classified_bins):
    """Print summary statistics of the analysis"""
    clipping_bins = classified_bins[classified_bins['is_clipping']]
    
    print("\n" + "="*60)
    print("NETWORK PEAK DETECTION ANALYSIS SUMMARY")
    print("="*60)
    print(f"Total data points: {len(df):,}")
    print(f"Time span: {df['time'].max() - df['time'].min():.1f} seconds")
    print(f"Long windows analyzed: {len(long_windows)}")
    print(f"Short bins analyzed: {len(bin_results)}")
    print(f"Bins showing clipping behavior: {len(clipping_bins)} ({len(clipping_bins)/len(bin_results)*100:.1f}%)")
    print(f"Overall max TX packets: {df['tx_packets'].max():,}")
    print(f"Overall average TX packets: {df['tx_packets'].mean():.1f}")
    
    if len(clipping_bins) > 0:
        print(f"\nClipping behavior detected in windows:")
        clipping_windows = clipping_bins['window_index'].unique()
        for window_idx in clipping_windows:
            window_idx = int(window_idx)  # Ensure integer type
            window_clipping = clipping_bins[clipping_bins['window_index'] == window_idx]
            total_samples = window_clipping['samples_near_max'].sum()
            print(f"  - Window {window_idx + 1}: {total_samples} samples near maximum")

def main():
    """Main analysis function"""
    
    # CONFIGURATION PARAMETERS - Adjust these as needed
    CSV_FILE_PATH = "/Users/admin/Desktop/Projects/nbn-testbed/extracted_data/test_50/50mbps_overload_s2-mgmt_s2_downstream.csv"
    LONG_WINDOW_SIZE = 60      # seconds
    SHORT_BIN_SIZE = 2         # seconds  
    TOLERANCE_PERCENT = 15     # percentage
    CLIPPING_THRESHOLD = 3     # number of samples
    
    print("Starting Network Peak Detection Analysis...")
    print(f"Parameters: Window={LONG_WINDOW_SIZE}s, Bin={SHORT_BIN_SIZE}s, Tolerance={TOLERANCE_PERCENT}%, Threshold={CLIPPING_THRESHOLD}")
    print(f"Focus: Analyzing time range 100-200 seconds for detailed visualization")
    
    # Step 0: Load data
    df = load_and_process_data(CSV_FILE_PATH)
    
    # Step 1: Calculate long-term maxima
    print("\nStep 1: Calculating long-term maxima...")
    long_windows = calculate_long_term_maxima(df, LONG_WINDOW_SIZE)
    print(f"Generated {len(long_windows)} long windows")
    
    # Step 2: Analyze short bins
    print("\nStep 2: Analyzing short bins...")
    bin_results = analyze_short_bins(df, long_windows, SHORT_BIN_SIZE, TOLERANCE_PERCENT)
    print(f"Generated {len(bin_results)} short bins")
    
    # Step 3: Binary classification
    print("\nStep 3: Performing binary classification...")
    classified_bins = classify_bins(bin_results, CLIPPING_THRESHOLD)
    clipping_count = len(classified_bins[classified_bins['is_clipping']])
    print(f"Found {clipping_count} bins with clipping behavior")
    
    # Generate the three figures separately
    print("\nGenerating the three analysis figures...")
    
    # Figure 1: Long-term maximum calculation
    fig1 = create_figure_1_long_term_maximum(df, long_windows)
    
    # Figure 2: Short-term bin analysis
    fig2 = create_figure_2_short_term_bins(df, long_windows, bin_results, TOLERANCE_PERCENT)
    
    # Figure 3: Binary classification
    fig3 = create_figure_3_binary_classification(df, long_windows, classified_bins, CLIPPING_THRESHOLD, TOLERANCE_PERCENT)
    
    # Print summary
    print_analysis_summary(df, long_windows, bin_results, classified_bins)
    
    # Save the figures as PNG files
    print("\nSaving figures as PNG files...")
    fig1.savefig('figure1_long_term_maximum.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print("✓ Saved: figure1_long_term_maximum.png")
    
    fig2.savefig('figure2_short_term_bins.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print("✓ Saved: figure2_short_term_bins.png")
    
    fig3.savefig('figure3_binary_classification.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print("✓ Saved: figure3_binary_classification.png")
    
    # Show all plots
    plt.show()
    
    print("\nAnalysis complete! Three figures generated and saved as PNG files.")

if __name__ == "__main__":
    main()