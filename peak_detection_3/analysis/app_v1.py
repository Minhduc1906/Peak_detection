import streamlit as st
st.set_page_config(page_title="Network Throughput Analysis", layout="wide")

import pandas as pd
import numpy as np
import os
import time
import re
import plotly.graph_objects as go
import plotly.io as pio
import torch
from sklearn.metrics import confusion_matrix, classification_report
import io
import base64

# Import from accuracy_comparison.py for LSTM model and evaluation
from visualization.accuracy_comparison import (
    LSTMModel,
    preprocess_data,
    load_model,
    evaluate_model,
    extract_network_limit,
    extract_scenario_name
)

# Import peak detection function
from processes.peak_speed_detect import peak_speed_detect

# Define the main folders
EXTRACTED_DATA_FOLDER = 'extracted_data'  # Raw network data
FEATURE_DATA_FOLDER = 'feature_data'      # Generated features
RESULTS_FOLDER = 'results'                # LSTM models and results
CONGESTION_THRESHOLD = 100  # Queue threshold


# ===========================================
# Helper Functions
# ===========================================
@st.cache_data(ttl=300)  # Cache data for 5 minutes
def get_available_subfolders():
    """Get all available subfolders in the EXTRACTED_DATA_FOLDER"""
    subfolders = []
    try:
        for root, dirs, _ in os.walk(EXTRACTED_DATA_FOLDER):
            rel_path = os.path.relpath(root, EXTRACTED_DATA_FOLDER)
            if rel_path != '.':
                subfolders.append(rel_path)
    except Exception as e:
        st.error(f"Error scanning directories: {str(e)}")
    return sorted(subfolders)

@st.cache_data(ttl=300)  # Cache data for 5 minutes
def get_csv_files(subfolder, folder=EXTRACTED_DATA_FOLDER):
    """Get all CSV files in the specified subfolder"""
    folder_path = os.path.join(folder, subfolder)
    if not os.path.exists(folder_path):
        return []
    return [f for f in os.listdir(folder_path) if f.endswith('.csv')]

@st.cache_data(ttl=300)  # Cache data for 5 minutes
def load_processed_data(subfolder, filename, use_features=False):
    """Load processed CSV data from specified subfolder"""
    if use_features:
        # Try to load from feature folder first
        file_path = os.path.join(FEATURE_DATA_FOLDER, subfolder, filename)
        if not os.path.exists(file_path):
            # If feature file doesn't exist, return None
            return None
    else:
        # Load from extracted data folder
        file_path = os.path.join(EXTRACTED_DATA_FOLDER, subfolder, filename)
    
    if not os.path.exists(file_path):
        st.warning(f"File not found: {file_path}")
        return None
    
    df = pd.read_csv(file_path)
    
    # Add extra features needed for the LSTM model
    if 'network_limit' not in df.columns:
        df['network_limit'] = extract_network_limit(subfolder)
    
    # Add direction column if it doesn't exist
    if 'direction' not in df.columns:
        if 'downstream' in filename:
            df['direction'] = 'downstream'
        elif 'upstream' in filename:
            df['direction'] = 'upstream'
    
    # Add file_name if it doesn't exist
    if 'file_name' not in df.columns:
        base_name = filename.replace('_downstream.csv', '').replace('_upstream.csv', '')
        base_name = base_name.replace('_features.csv', '')
        df['file_name'] = base_name
    
    return df

def create_time_series_csv(time_array, throughput_filtered, queue_exists, traditional_predictions, 
                          lstm_predictions=None, queue_size=None):
    """Create a DataFrame with all time series data for CSV export"""
    
    # Create the base DataFrame
    export_df = pd.DataFrame({
        'time_seconds': time_array,
        'throughput_filtered_mbps': throughput_filtered,
        'queue_exists_ground_truth': queue_exists.astype(int),
        'traditional_detection': traditional_predictions.astype(int)
    })
    
    # Add LSTM predictions if available
    if lstm_predictions is not None:
        export_df['lstm_prediction_probability'] = lstm_predictions
        export_df['lstm_detection_binary'] = (lstm_predictions >= 0.5).astype(int)
    
    # Add queue size if available
    if queue_size is not None:
        export_df['queue_size_packets'] = queue_size
    
    return export_df

# ===========================================
# PDF Export Functions
# ===========================================

def create_pdf_download_link(fig, filename, title="Download as PDF"):
    """Create a download link for a Plotly figure as PDF"""
    try:
        # Convert figure to PDF bytes
        pdf_bytes = pio.to_image(fig, format="pdf", width=1200, height=800, scale=2)
        
        # Create download link
        b64 = base64.b64encode(pdf_bytes).decode()
        href = f'<a href="data:application/pdf;base64,{b64}" download="{filename}" style="text-decoration: none; background-color: #4CAF50; color: white; padding: 8px 12px; border-radius: 4px; font-size: 14px; display: inline-block; margin: 5px;">📄 {title}</a>'
        return href
    except Exception as e:
        return f"<span style='color: red;'>PDF export failed: {str(e)}</span>"

def save_all_figures_as_pdf(figures_dict, experiment_name, subfolder):
    """Save all figures as individual PDF files and create download links"""
    download_links = []
    
    for fig_name, fig in figures_dict.items():
        if fig is not None:
            safe_filename = f"{experiment_name}_{subfolder}_{fig_name}.pdf"
            safe_filename = re.sub(r'[^\w\-_\.]', '_', safe_filename)
            
            link = create_pdf_download_link(fig, safe_filename, f"Download {fig_name}")
            download_links.append(link)
    
    return download_links

# ===========================================
# Throughput Calculation Functions
# ===========================================

def calculate_bit_rate(df, byte_col, time_col='relative_time'):
    """
    Calculate throughput - byte values are already rates per interval
    Formula: 8 * ds(2:end, 5) / ds_interval / 1e6
    """
    if byte_col not in df.columns:
        return np.zeros(len(df))
    
    # Skip first row and calculate mean interval
    time_values = df[time_col].values
    if len(time_values) < 3:
        return np.zeros(len(df))
    
    # Calculate relative time from second row
    relative_time = time_values[1:] - time_values[1]  # Start from second row
    if len(relative_time) < 2:
        return np.zeros(len(df))
    
    # Calculate mean interval from time differences
    interval = np.mean(np.diff(relative_time))
    if interval <= 0:
        interval = 0.02  # Default fallback
    
    # Byte values from row 2 onwards (skip first data row)
    byte_values = df[byte_col].values[1:]  # Skip first row
    
    # Calculate Mbps: convert bytes to bits, normalize by interval, convert to Mbps
    bit_rate = 8 * byte_values / interval / 1e6
    
    # Pad with zero at the beginning to match original dataframe length
    bit_rate_full = np.concatenate([np.zeros(1), bit_rate])
    
    # Handle negative values (counter resets) by setting to 0
    bit_rate_full = np.maximum(bit_rate_full, 0)
    
    return bit_rate_full

def calculate_packet_rate(df, packet_col, time_col='relative_time'):
    """
    Calculate packet rate - packet values are already rates per interval
    Formula: ds(2:end, 4) / ds_interval
    """
    if packet_col not in df.columns:
        return np.zeros(len(df))
    
    # Skip first row and calculate mean interval
    time_values = df[time_col].values
    if len(time_values) < 3:
        return np.zeros(len(df))
    
    # Calculate mean interval
    relative_time = time_values[1:] - time_values[1]
    if len(relative_time) < 2:
        return np.zeros(len(df))
    
    interval = np.mean(np.diff(relative_time))
    if interval <= 0:
        interval = 0.02
    
    # Packet values from row 2 onwards (skip first row)
    packet_values = df[packet_col].values[1:]  # Skip first row
    packet_rate = packet_values / interval
    
    # Pad with zero at the beginning
    packet_rate_full = np.concatenate([np.zeros(1), packet_rate])
    
    # Handle negative values by setting to 0
    packet_rate_full = np.maximum(packet_rate_full, 0)
    
    return packet_rate_full

# Function to safely run peak detection
def safe_peak_detection(time_array, throughput, long_window_ms=1000, short_window_ms=200, 
                       peak_tolerance=0.1, count_threshold=5, use_median_filter=True):
    """A safe wrapper around peak detection to handle various edge cases"""
    try:
        if len(time_array) <= 1 or len(throughput) <= 1:
            return None, None, None, None, None
            
        # Check for all zero/constant data
        if np.all(throughput == throughput[0]):
            return np.zeros_like(throughput), np.zeros_like(throughput), np.zeros_like(throughput), np.zeros_like(throughput), throughput
        
        # Handle potential issues from diff-based calculation
        # Remove any remaining inf/nan values that might have slipped through
        throughput_clean = np.nan_to_num(throughput, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Use the corrected peak_speed_detect function with proper parameters including filter option
        clipping_binary, long_maxima, filtered_throughput, peak_ratio, short_counts, short_peaks = peak_speed_detect(
            throughput_clean, 
            time_array,
            long_window_ms=long_window_ms,
            short_window_ms=short_window_ms,
            peak_tolerance=peak_tolerance,
            count_threshold=count_threshold,
            use_median_filter=use_median_filter
        )
        
        # If peak detection returns None values, create zeros
        if clipping_binary is None or long_maxima is None or filtered_throughput is None or peak_ratio is None or short_counts is None:
            return np.zeros_like(throughput), np.zeros_like(throughput), np.zeros_like(throughput), np.zeros_like(throughput), throughput
            
        # Convert binary output to float for visualization (0.0 or 1.0)
        clipping_score = clipping_binary.astype(float)
        
        # Return in format expected by visualization: (score, long_peaks, short_peaks, ratio, filtered)
        return clipping_score, long_maxima, short_counts, peak_ratio, filtered_throughput
        
    except Exception as e:
        print(f"Peak detection failed: {str(e)}")
        return None, None, None, None, None

# Initialize LSTM models
@st.cache_resource
def load_lstm_models():
    """Load LSTM models using accuracy_comparison approach"""
    models_dir = os.path.join(RESULTS_FOLDER, 'models')
    
    if not os.path.exists(models_dir):
        st.warning(f"Models directory not found: {models_dir}")
        return None
    
    models = {}
    
    # Load finetune model
    finetune_model_path = os.path.join(models_dir, 'finetune_final_model.pt')
    if not os.path.exists(finetune_model_path):
        # Try looking for the last finetune step model
        finetune_files = [f for f in os.listdir(models_dir) if f.startswith('finetune_step') and f.endswith('.pt')]
        if finetune_files:
            # Get the highest step number
            step_nums = [int(re.search(r'step(\d+)', f).group(1)) for f in finetune_files]
            highest_step = max(step_nums)
            finetune_model_path = os.path.join(models_dir, [f for f in finetune_files if f'step{highest_step}' in f][0])
        else:
            print("Finetune model not found")
    
    # Load combined model
    combined_model_path = os.path.join(models_dir, 'combined_model_downstream_only.pt')
    
    # Try to load models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        if os.path.exists(finetune_model_path):
            finetune_state_dict, finetune_time_steps = load_model(finetune_model_path)
            models['finetune'] = {
                'state_dict': finetune_state_dict,
                'time_steps': finetune_time_steps,
                'path': finetune_model_path
            }
    except Exception as e:
        print(f"Failed to load finetune model: {str(e)}")
    
    try:
        if os.path.exists(combined_model_path):
            combined_state_dict, combined_time_steps = load_model(combined_model_path)
            models['combined'] = {
                'state_dict': combined_state_dict,
                'time_steps': combined_time_steps,
                'path': combined_model_path
            }
    except Exception as e:
        print(f"Failed to load combined model: {str(e)}")
    
    return models if models else None

def predict_with_lstm(df, model_data, model_name):
    """Make predictions using the selected LSTM model with the correct feature set"""
    if not model_data or model_name not in model_data:
        return None
        
    # Get model information
    state_dict = model_data[model_name]['state_dict']
    time_steps = model_data[model_name]['time_steps']
    
    # Create a copy of the dataframe to avoid modification warnings
    df_copy = df.copy()
    
    # Ensure required features exist
    required_features = ["clipping_score",
        "long_term_peak",
        "short_term_counts",
        "short_term_peaks",
        "filtered_throughput",
        "peak_ratio"]
    for feature in required_features:
        if feature not in df_copy.columns:
            print(f"Missing required feature: {feature}")
            return None
    
    # Ensure direction is set correctly
    if 'direction' not in df_copy.columns:
        df_copy['direction'] = 'downstream'  # Assume downstream for prediction
    
    # Create a new dataframe with ONLY the required features
    features_df = df_copy[required_features].copy()
    
    # Check for NaN or infinite values and replace them
    features_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    for col in features_df.columns:
        if features_df[col].isna().any():
            features_df[col].fillna(features_df[col].mean(), inplace=True)
    
    # Add necessary additional columns
    if 'queue_exists' in df_copy.columns:
        features_df['queue_exists'] = df_copy['queue_exists']
    else:
        features_df['queue_exists'] = 0  # Default value if missing
        
    if 'network_limit' in df_copy.columns:
        features_df['network_limit'] = df_copy['network_limit']
    else:
        features_df['network_limit'] = -1  # Use default value
    
    # Preprocess the data with only the required features
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_seq, y_seq, feature_cols, _ = preprocess_data(
        features_df, 
        include_network_limit=True, 
        time_steps=time_steps
    )
    
    # Create and load the model
    model = LSTMModel(input_size=X_seq.shape[2])
    model.load_state_dict(state_dict)
    model.to(device)
    
    # Make predictions
    model.eval()
    X_tensor = torch.FloatTensor(X_seq).to(device)
    
    with torch.no_grad():
        outputs = model(X_tensor)
        predicted = outputs.cpu().numpy().flatten()
    
    # Account for the sequence effect (first time_steps entries have no prediction)
    full_predictions = np.concatenate([
        np.zeros(time_steps),  # No predictions for first time_steps elements
        predicted
    ])
    
    # Handle any length mismatch
    if len(full_predictions) < len(df):
        # Pad with zeros if necessary
        full_predictions = np.concatenate([
            full_predictions,
            np.zeros(len(df) - len(full_predictions))
        ])
    elif len(full_predictions) > len(df):
        # Truncate if necessary
        full_predictions = full_predictions[:len(df)]
    
    return full_predictions

# ===========================================
# Visualization Functions
# ===========================================
def plot_confusion_matrix(cm, title="Confusion Matrix"):
    """Create a plotly heatmap for confusion matrix visualization."""
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['No Peak', 'Peak'],
        y=['No Peak', 'Peak'],
        text=cm,
        texttemplate="%{text}",
        textfont={"size": 16},
        colorscale='Blues'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Predicted",
        yaxis_title="Actual",
        height=400,
        xaxis=dict(
            title=dict(font=dict(size=32)),
            tickfont=dict(size=28)
        ),
        yaxis=dict(
            title=dict(font=dict(size=32)),
            tickfont=dict(size=28)
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=24)
        )
    )
    
    return fig

def create_throughput_figures(df, is_downstream=True, show_peak_detection=False, 
                            long_window_ms=1000, short_window_ms=200, peak_tolerance=0.1, 
                            count_threshold=5, use_lstm=False, lstm_predictions=None):
    """Create throughput visualization figures with optional LSTM predictions"""
    direction = "Downstream" if is_downstream else "Upstream"
    
    # Skip if insufficient data
    if len(df) < 2:
        return []
        
    t = df['relative_time'].values
    
    # Calculate rates
    tx_packet_rate = calculate_packet_rate(df, 'tx_packets', 'relative_time')
    rx_packet_rate = calculate_packet_rate(df, 'rx_packets', 'relative_time')
    tx_bit_rate = calculate_bit_rate(df, 'tx_bytes', 'relative_time')
    rx_bit_rate = calculate_bit_rate(df, 'rx_bytes', 'relative_time')
    
    # Get network capacity from dataframe if available, otherwise use default (100)
    network_capacity = 100
    if 'network_limit' in df.columns:
        network_limit = df['network_limit'].iloc[0]
        if network_limit > 0:
            network_capacity = network_limit
    network_capacity = round(network_capacity * 1.15)
    
    figures = []
    
    # ------- Mbits/sec Figure -------
    if 'tx_bytes' in df.columns and 'rx_bytes' in df.columns:
        fig = go.Figure()
        
        # Add traces
        fig.add_trace(go.Scatter(x=t, y=tx_bit_rate, name='Tx', line=dict(color='blue'), showlegend=is_downstream))
        fig.add_trace(go.Scatter(x=t, y=rx_bit_rate, name='Rx', line=dict(color='red'), showlegend=is_downstream))
        
        # Add network capacity line
        fig.add_trace(go.Scatter(x=[min(t), max(t)], y=[network_capacity, network_capacity],
                                 name=f'Network Capacity ({network_capacity} Mbps)', 
                                 line=dict(color='orange', dash='dash'),
                                 showlegend=is_downstream))
        
        # Add peak detection if enabled
        if show_peak_detection:
            # Run peak detection with current parameters
            peak_score, long_maxima, short_counts, peak_ratio, filtered_tp = safe_peak_detection(
                t, tx_bit_rate, long_window_ms, short_window_ms, peak_tolerance, count_threshold
            )
            
            if peak_score is not None:
                max_val = np.max(tx_bit_rate) if len(tx_bit_rate) > 0 else 1
                fig.add_trace(go.Scatter(
                    x=t, 
                    y=peak_score * max_val, 
                    name='Peak Detection',
                    line=dict(color='orange'), 
                    showlegend=is_downstream
                ))
        
        # Add LSTM predictions if enabled
        if use_lstm and lstm_predictions is not None:
            max_val = np.max(tx_bit_rate) if len(tx_bit_rate) > 0 else 1
            fig.add_trace(go.Scatter(
                x=t, 
                y=lstm_predictions * max_val, 
                name='LSTM Prediction',
                line=dict(color='green'), 
                showlegend=is_downstream
            ))
        
        # Layout with legend on top
        fig.update_layout(
            title=f'{direction} Throughput (Mb/s)',
            xaxis_title='Time (seconds)',
            yaxis_title=f'{direction} throughput (Mb/s)',
            template='plotly_white',
            height=500,
            xaxis=dict(
                showgrid=True, 
                title=dict(font=dict(size=16)),
                tickfont=dict(size=14)
            ),
            yaxis=dict(
                showgrid=True,
                title=dict(font=dict(size=16)),
                tickfont=dict(size=14)
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            )
        )
        figures.append(fig)
    
    # ------- Buffer Occupancy Figure -------
    if 'queue_exists' in df.columns and 'queue_size' in df.columns:
        fig = go.Figure()

        # Queue exists visualization
        queue_exists = df['queue_exists'].values
        max_queue_size = df['queue_size'].max() if df['queue_size'].max() > 0 else 10
        
        # Create a continuous area for queue_exists
        queue_indicator = queue_exists * max_queue_size * 1.1
        fig.add_trace(go.Scatter(
            x=t, 
            y=queue_indicator,
            name='Queue Exists (Ground Truth)',
            fill='tozeroy',
            mode='none',
            fillcolor="rgba(255,200,200,0.3)",
            showlegend=is_downstream
        ))

        # Add actual queue size line
        fig.add_trace(go.Scatter(
            x=t, 
            y=df['queue_size'], 
            name='Queue Size',
            line=dict(color='blue'),
            showlegend=is_downstream
        ))

        # Add queue threshold line
        fig.add_trace(go.Scatter(
            x=[min(t), max(t)], 
            y=[CONGESTION_THRESHOLD, CONGESTION_THRESHOLD],  # Threshold used to determine queue_exists
            name='Queue Threshold',
            line=dict(color='red', dash='dash'),
            showlegend=is_downstream
        ))
        
        # Add peak detection if enabled
        if show_peak_detection:
            # Run peak detection with current parameters
            peak_score, long_maxima, short_counts, peak_ratio, filtered_tp = safe_peak_detection(
                t, tx_bit_rate, long_window_ms, short_window_ms, peak_tolerance, count_threshold
            )
            
            if peak_score is not None:
                peak_indicator = peak_score * max_queue_size * 1.1
                fig.add_trace(go.Scatter(
                    x=t,
                    y=peak_indicator,
                    name='Peak Detection',
                    fill='tozeroy',
                    mode='none',
                    fillcolor="rgba(255,165,0,0.3)",
                    showlegend=is_downstream
                ))
        
        # Add LSTM predictions as additional area if enabled
        if use_lstm and lstm_predictions is not None:
            lstm_indicator = lstm_predictions * max_queue_size * 1.1
            fig.add_trace(go.Scatter(
                x=t,
                y=lstm_indicator,
                name='LSTM Prediction',
                fill='tozeroy',
                mode='none',
                fillcolor="rgba(100,255,100,0.3)",
                showlegend=is_downstream
            ))
        
        # Layout with legend on top
        fig.update_layout(
            title=f'{direction} Buffer Occupancy',
            xaxis_title='Time (seconds)',
            yaxis_title=f'{direction} tx queue buffer occupancy (packets)',
            template='plotly_white',
            height=500,
            xaxis=dict(
                showgrid=True,
                title=dict(font=dict(size=16)),
                tickfont=dict(size=14)
            ),
            yaxis=dict(
                showgrid=True,
                title=dict(font=dict(size=16)),
                tickfont=dict(size=14)
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            )
        )
        figures.append(fig)
    
    # ------- Feature Visualization -------
    if all(feature in df.columns for feature in ["clipping_score",
        "long_term_peak",
        "short_term_counts",
        "short_term_peaks",
        "filtered_throughput",
        "peak_ratio"]):
        fig = go.Figure()
        
        # Add ratio trace
        fig.add_trace(go.Scatter(
            x=t, 
            y=df["peak_ratio"].values, 
            name='Peak Ratio',
            line=dict(color='orange'),
            showlegend=is_downstream
        ))
        
        # Add filtered throughput (normalized)
        max_filtered = max(1, np.max(df["filtered_throughput"].values))
        norm_filtered = df["filtered_throughput"].values / max_filtered
        
        fig.add_trace(go.Scatter(
            x=t, 
            y=norm_filtered, 
            name='Filtered Throughput (norm)',
            line=dict(color='blue', dash='dot'),
            showlegend=is_downstream
        ))
        
        # Add peak score
        fig.add_trace(go.Scatter(
            x=t, 
            y=df["clipping_score"].values, 
            name='Peak Score',
            line=dict(color='red'),
            showlegend=is_downstream
        ))
        
        # Layout with legend on top
        fig.update_layout(
            title=f'{direction} Feature Visualization',
            xaxis_title='Time (seconds)',
            yaxis_title='Feature Values',
            template='plotly_white',
            height=500,
            xaxis=dict(
                showgrid=True,
                title=dict(font=dict(size=16)),
                tickfont=dict(size=14)
            ),
            yaxis=dict(
                showgrid=True,
                title=dict(font=dict(size=16)),
                tickfont=dict(size=14)
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            )
        )
        figures.append(fig)
    
    return figures

def calculate_lstm_metrics(ground_truth, predictions):
    """Calculate metrics comparing predictions to ground truth queue existence"""
    # Handle different prediction formats
    if hasattr(predictions, 'dtype') and predictions.dtype == bool:
        # Boolean predictions from old algorithm
        binary_predictions = predictions.astype(int)
    elif np.all((predictions == 0) | (predictions == 1)):
        # Already binary (0/1) from new algorithm
        binary_predictions = predictions.astype(int)
    else:
        # Continuous predictions (convert with threshold)
        binary_predictions = (predictions >= 0.5).astype(int)
    
    # Ensure ground truth is also binary
    ground_truth = ground_truth.astype(int)
    
    # Calculate confusion matrix
    try:
        cm = confusion_matrix(ground_truth, binary_predictions)
        report = classification_report(ground_truth, binary_predictions, output_dict=True)
        return cm, report
    except Exception as e:
        print(f"Error calculating metrics: {str(e)}")
        # Return dummy metrics
        num_negatives = len(ground_truth) - sum(ground_truth)
        dummy_cm = np.array([[num_negatives, 0], [sum(ground_truth), 0]])
        dummy_report = {
            'accuracy': num_negatives / len(ground_truth) if len(ground_truth) > 0 else 0,
            '0': {'precision': 1.0, 'recall': 1.0 if num_negatives > 0 else 0.0, 'f1-score': 1.0 if num_negatives > 0 else 0.0}
        }
        # Add class 1 metrics if there are positive samples
        if sum(ground_truth) > 0:
            dummy_report['1'] = {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0}
        return dummy_cm, dummy_report

def create_comparison_dataframe(traditional_report, lstm_report):
    """Create a comparison dataframe of traditional vs LSTM performance"""
    if '1' not in traditional_report or '1' not in lstm_report:
        return None
        
    comparison_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        'Traditional': [
            str(round(traditional_report['accuracy'], 3)),
            str(round(traditional_report['1']['precision'], 3)),
            str(round(traditional_report['1']['recall'], 3)),
            str(round(traditional_report['1']['f1-score'], 3))
        ],
        'LSTM': [
            str(round(lstm_report['accuracy'], 3)),
            str(round(lstm_report['1']['precision'], 3)),
            str(round(lstm_report['1']['recall'], 3)),
            str(round(lstm_report['1']['f1-score'], 3))
        ]
    })
    return comparison_df.set_index('Metric')

# ===========================================
# Visualization Page 1: Data Visualization
# ===========================================
def data_visualization_page(downstream_df, upstream_df):
    st.header("Network Data Visualization")
    
    # Create visualizations for downstream and upstream data
    downstream_figs = create_throughput_figures(downstream_df, is_downstream=True)
    upstream_figs = create_throughput_figures(upstream_df, is_downstream=False)

    # Display figures side by side
    if downstream_figs and upstream_figs:
        for i in range(min(len(downstream_figs), len(upstream_figs))):
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(downstream_figs[i], use_container_width=True)
            with col2:
                st.plotly_chart(upstream_figs[i], use_container_width=True)
    else:
        st.warning("Insufficient data for visualization.")
    
    # Display network statistics
    st.subheader("Network Statistics")
    
    # Calculate rate statistics
    if 'tx_bytes' in downstream_df.columns and 'rx_bytes' in downstream_df.columns:
        ds_tx_bit_rate = calculate_bit_rate(downstream_df, 'tx_bytes', 'relative_time')
        ds_rx_bit_rate = calculate_bit_rate(downstream_df, 'rx_bytes', 'relative_time')
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Downstream Statistics**")
            
            # Throughput metrics
            st.write(f"Average TX rate: {np.mean(ds_tx_bit_rate):.2f} Mb/s")
            st.write(f"Maximum TX rate: {np.max(ds_tx_bit_rate):.2f} Mb/s")
            st.write(f"Average RX rate: {np.mean(ds_rx_bit_rate):.2f} Mb/s")
            st.write(f"Maximum RX rate: {np.max(ds_rx_bit_rate):.2f} Mb/s")
            
            # Network utilization
            if 'network_limit' in downstream_df.columns:
                ds_network_capacity = downstream_df['network_limit'].iloc[0]
                if ds_network_capacity > 0:
                    above_90 = ds_tx_bit_rate > (ds_network_capacity * 0.9)
                    above_95 = ds_tx_bit_rate > (ds_network_capacity * 0.95)
                    above_99 = ds_tx_bit_rate > (ds_network_capacity * 0.99)
                    st.write(f"Time above 90% capacity: {(np.mean(above_90) * 100):.2f}%")
                    st.write(f"Time above 95% capacity: {(np.mean(above_95) * 100):.2f}%")
                    st.write(f"Time above 99% capacity: {(np.mean(above_99) * 100):.2f}%")
            
            # Queue statistics
            if 'queue_size' in downstream_df.columns:
                st.write(f"Average queue size: {downstream_df['queue_size'].mean():.2f} packets")
                st.write(f"Maximum queue size: {downstream_df['queue_size'].max():.2f} packets")
                
            if 'queue_exists' in downstream_df.columns:
                st.write(f"Time with queue: {(downstream_df['queue_exists'].mean() * 100):.2f}%")
        
        if 'tx_bytes' in upstream_df.columns and 'rx_bytes' in upstream_df.columns:
            us_tx_bit_rate = calculate_bit_rate(upstream_df, 'tx_bytes', 'relative_time')
            us_rx_bit_rate = calculate_bit_rate(upstream_df, 'rx_bytes', 'relative_time')
            
            with col2:
                st.write("**Upstream Statistics**")
                
                # Throughput metrics
                st.write(f"Average TX rate: {np.mean(us_tx_bit_rate):.2f} Mb/s")
                st.write(f"Maximum TX rate: {np.max(us_tx_bit_rate):.2f} Mb/s")
                st.write(f"Average RX rate: {np.mean(us_rx_bit_rate):.2f} Mb/s")
                st.write(f"Maximum RX rate: {np.max(us_rx_bit_rate):.2f} Mb/s")
                
                # Network utilization
                if 'network_limit' in upstream_df.columns:
                    us_network_capacity = upstream_df['network_limit'].iloc[0]
                    if us_network_capacity > 0:
                        above_90 = us_tx_bit_rate > (us_network_capacity * 0.9)
                        above_95 = us_tx_bit_rate > (us_network_capacity * 0.95)
                        above_99 = us_tx_bit_rate > (us_network_capacity * 0.99)
                        st.write(f"Time above 90% capacity: {(np.mean(above_90) * 100):.2f}%")
                        st.write(f"Time above 95% capacity: {(np.mean(above_95) * 100):.2f}%")
                        st.write(f"Time above 99% capacity: {(np.mean(above_99) * 100):.2f}%")
                
                # Queue statistics
                if 'queue_size' in upstream_df.columns:
                    st.write(f"Average queue size: {upstream_df['queue_size'].mean():.2f} packets")
                    st.write(f"Maximum queue size: {upstream_df['queue_size'].max():.2f} packets")
                    
                if 'queue_exists' in upstream_df.columns:
                    st.write(f"Time with queue: {(upstream_df['queue_exists'].mean() * 100):.2f}%")
    else:
        st.warning("Missing tx_bytes or rx_bytes data columns required for statistics.")

# ===========================================
# Peak Detection Visualization Function
# ===========================================

def create_peak_detection_visualization(time_array, throughput, network_capacity, 
                                      long_window_ms, short_window_ms, peak_tolerance, 
                                      count_threshold, use_median_filter, file_name=""):
    """Create a detailed peak detection visualization using Plotly"""
    
    # Run peak detection with current parameters including filter option
    try:
        clipping_binary, long_maxima, filtered_throughput, peak_ratio, short_counts, short_peaks = peak_speed_detect(
            throughput, time_array,
            long_window_ms=long_window_ms,
            short_window_ms=short_window_ms,
            peak_tolerance=peak_tolerance,
            count_threshold=count_threshold,
            use_median_filter=use_median_filter
        )
        
        if clipping_binary is None:
            return None
            
    except Exception as e:
        st.error(f"Peak detection failed: {str(e)}")
        return None
    
    # Create the main figure
    fig = go.Figure()
    
    # Add throughput line (show what the algorithm actually sees)
    if use_median_filter:
        # Show filtered data when filter is ON (what algorithm actually processes)
        display_throughput = filtered_throughput  # This comes from peak detection results
        throughput_label = 'Downstream Throughput (Filtered)'
        throughput_help = 'Median-filtered throughput data used by algorithm'
    else:
        # Show raw data when filter is OFF (filtered_throughput equals original when no filtering)
        display_throughput = filtered_throughput  # This equals original throughput when filtering is off
        throughput_label = 'Downstream Throughput (Raw)'
        throughput_help = 'Raw throughput data used by algorithm'
    
    fig.add_trace(go.Scatter(
        x=time_array,
        y=display_throughput,
        name=throughput_label,
        line=dict(color='red', width=2),
        mode='lines'
    ))
    
    # Add network capacity line
    fig.add_trace(go.Scatter(
        x=[time_array.min(), time_array.max()],
        y=[network_capacity, network_capacity],
        name=f'Network Capacity ({network_capacity} Mbps)',
        line=dict(color='black', dash='dash', width=2),
        mode='lines'
    ))
    
    # Calculate window parameters
    mean_dt = np.mean(np.diff(time_array)) if len(time_array) > 1 else 0.02
    long_window_samples = max(1, int(long_window_ms / 1000 / mean_dt))
    short_window_samples = max(1, int(short_window_ms / 1000 / mean_dt))
    
    # Track legend entries to avoid duplicates
    long_window_legend_added = False
    clipping_bin_legend_added = False
    tolerance_legend_added = False
    
    # Store all shapes and annotations to add them efficiently
    shapes = []
    annotations = []
    
    # Process each long window using the exact same data the algorithm used
    for long_start in range(0, len(display_throughput), long_window_samples):
        long_end = min(long_start + long_window_samples, len(display_throughput))
        
        # Always use the algorithm's detected maximum (from long_maxima)
        algo_long_max = long_maxima[long_start] if long_start < len(long_maxima) else 0
        
        # Use algorithm's maximum for all visualizations (perfect consistency)
        display_max = algo_long_max
        
        if display_max > 0:
            # Add long window as a light blue shaded area using ACTUAL maximum height
            shapes.append(dict(
                type="rect",
                x0=time_array[long_start],
                x1=time_array[long_end-1],
                y0=0,
                y1=algo_long_max,
                fillcolor="rgba(135,206,250,0.3)",  # Light sky blue
                line=dict(width=0),
                layer="below"
            ))
            
            # Add peak tolerance visualization (darker region at the TOP)
            if algo_long_max > 0:
                # Use algo_long_max for tolerance calculation so it's at the TOP of light blue area
                tolerance_threshold = (1 - peak_tolerance) * algo_long_max
                tolerance_height = algo_long_max  # Goes to the very top
                
                # Add darker blue region showing the peak tolerance zone (always at the top)
                shapes.append(dict(
                    type="rect",
                    x0=time_array[long_start],
                    x1=time_array[long_end-1],
                    y0=tolerance_threshold,
                    y1=tolerance_height,
                    fillcolor="rgba(70,130,180,0.6)",  # Steel blue, more opaque
                    line=dict(width=0),
                    layer="below"
                ))
            
            # Add invisible traces for legend (only once each)
            if not long_window_legend_added:
                fig.add_trace(go.Scatter(
                    x=[time_array[long_start], time_array[long_start]],
                    y=[0, algo_long_max],
                    fill='tozeroy',
                    fillcolor='rgba(135,206,250,0.3)',
                    line=dict(color='rgba(135,206,250,0.3)'),
                    name='Long Window (Max Detection)',
                    showlegend=True,
                    mode='none'
                ))
                long_window_legend_added = True
                
            if not tolerance_legend_added and algo_long_max > 0:
                tolerance_threshold = (1 - peak_tolerance) * algo_long_max
                fig.add_trace(go.Scatter(
                    x=[time_array[long_start], time_array[long_start]],
                    y=[tolerance_threshold, algo_long_max],
                    fill='tozeroy',
                    fillcolor='rgba(70,130,180,0.6)',
                    line=dict(color='rgba(70,130,180,0.6)'),
                    name=f'Peak Tolerance Zone ({int(peak_tolerance*100)}%)',
                    showlegend=True,
                    mode='none'
                ))
                tolerance_legend_added = True
            
            # Process short bins within this long window
            for short_start in range(long_start, long_end, short_window_samples):
                short_end = min(short_start + short_window_samples, long_end)
                
                if short_start < len(clipping_binary) and clipping_binary[short_start] == 1:
                    count = short_counts[short_start] if short_start < len(short_counts) else 0
                    
                    # Add clipping short bin as orange shaded area (use consistent display height)
                    shapes.append(dict(
                        type="rect",
                        x0=time_array[short_start],
                        x1=time_array[short_end-1],
                        y0=0,
                        y1=display_max,
                        fillcolor="rgba(255,140,0,0.6)",  # Dark orange
                        line=dict(width=0),
                        layer="below"
                    ))
                    
                    # Add invisible trace for legend (only once)
                    if not clipping_bin_legend_added:
                        fig.add_trace(go.Scatter(
                            x=[time_array[short_start], time_array[short_start]],
                            y=[0, display_max],
                            fill='tozeroy',
                            fillcolor='rgba(255,140,0,0.6)',
                            line=dict(color='rgba(255,140,0,0.6)'),
                            name='Clipping Short Bin (N≥threshold)',
                            showlegend=True,
                            mode='none'
                        ))
                        clipping_bin_legend_added = True
                    
                    # Add N=count annotation with better styling (position above display max)
                    mid_time = (time_array[short_start] + time_array[short_end-1]) / 2
                    annotations.append(dict(
                        x=mid_time,
                        y=display_max * 1.08,
                        text=f'N={int(count)}',
                        showarrow=False,
                        font=dict(size=12, color='black', family='Arial Black'),
                        bgcolor='rgba(255,255,255,0.9)',
                        bordercolor='rgba(255,140,0,0.8)',
                        borderwidth=2,
                        borderpad=3
                    ))
    
    # Add all shapes and annotations to the figure
    fig.update_layout(shapes=shapes)
    fig.update_layout(annotations=annotations)
    
    # Update layout with legend on top
    fig.update_layout(
        xaxis_title='Time (seconds)',
        yaxis_title='Throughput (Mbps)',
        template='plotly_white',
        height=600,
        showlegend=True,
        xaxis=dict(
            title=dict(font=dict(size=32)),
            tickfont=dict(size=28)
        ),
        yaxis=dict(
            title=dict(font=dict(size=32)),
            tickfont=dict(size=28)
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=24)
        )
    )
    
    return fig

# ===========================================
# Updated Detection Comparison Page with Ground Truth Throughput and PDF Export
# ===========================================
def detection_comparison_page(downstream_df, downstream_features_df, lstm_models):
    st.header("Downstream Detection Comparison")
    
    # Check if features dataframe is available
    has_lstm_features = False
    if downstream_features_df is not None:
        required_features = ["clipping_score",
        "long_term_peak",
        "short_term_counts",
        "short_term_peaks",
        "filtered_throughput",
        "peak_ratio"]
        missing_features = [f for f in required_features if f not in downstream_features_df.columns]
        
        if missing_features:
            st.warning(f"Missing required features: {', '.join(missing_features)}")
            st.info("Please run feature_extraction.py on your data first.")
        else:
            has_lstm_features = True
            # Merge features with original data if they're not already present
            for feature in required_features:
                if feature not in downstream_df.columns:
                    downstream_df[feature] = downstream_features_df[feature]
    else:
        st.warning("Feature data not found. LSTM requires features from feature_extraction.py.")
    
    # Get available models
    if lstm_models and has_lstm_features:
        available_models = list(lstm_models.keys())
        selected_model = st.selectbox(
            "Select LSTM Model",
            options=available_models,
            help="Choose which LSTM model to use for predictions"
        )
        
        # Display model info
        st.info(f"Using model from: {lstm_models[selected_model]['path']}")
    else:
        if not lstm_models:
            st.warning("LSTM models not loaded. Only traditional peak detection will be shown.")
        selected_model = None
    
    # Parameter sliders for peak detection
    st.subheader("Peak Detection Parameters")
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        long_window_ms = st.slider(
            "Long Window (ms)",
            min_value=5000,
            max_value=25500,
            value=5000,
            step=500,
            help="Long window duration for peak detection"
        )
    
    with col2:
        short_window_ms = st.slider(
            "Short Window (ms)",
            min_value=50,
            max_value=1000,
            value=200,
            step=50,
            help="Short window duration for peak detection"
        )
    
    with col3:
        peak_tolerance = st.slider(
            "Peak Tolerance",
            min_value=0.05,
            max_value=0.3,
            value=0.1,
            step=0.05,
            help="Tolerance for being close to peak"
        )
    
    with col4:
        count_threshold = st.slider(
            "Count Threshold",
            min_value=1,
            max_value=20,
            value=5,
            step=1,
            help="Minimum sample count threshold"
        )
    
    with col5:
        use_median_filter = st.checkbox(
            "Median Filter",
            value=True,
            help="Apply median filter to reduce noise (recommended for real data)"
        )
    
    with col6:
        limit_to_100s = st.checkbox(
            "Limit PDFs to 100s",
            value=True,
            help="Limit PDF export and CSV to first 100 seconds of data for focused analysis"
        )
    
    # Run LSTM prediction if available
    if lstm_models and selected_model and has_lstm_features:
        with st.spinner("Running LSTM predictions..."):
            downstream_predictions = predict_with_lstm(
                downstream_df, 
                lstm_models, 
                selected_model
            )
    else:
        downstream_predictions = None
    
    # Calculate traditional peak detection
    with st.spinner("Calculating traditional peak detection..."):
        # Prepare data for peak detection
        ds_time = downstream_df['relative_time'].values
        
        # Calculate throughput rates
        if 'tx_bytes' in downstream_df.columns:
            ds_tx_rate = calculate_bit_rate(downstream_df, 'tx_bytes', 'relative_time')
            
            # Run peak detection with updated parameters including filter option
            ds_peaking, ds_long_maxima, ds_short_counts, ds_ratio, ds_filtered = safe_peak_detection(
                ds_time, ds_tx_rate, long_window_ms, short_window_ms, peak_tolerance, count_threshold, use_median_filter
            )
            
            # If peak detection failed, create dummy values
            if ds_peaking is None:
                ds_peaking = np.zeros_like(ds_time)
                ds_peak_diff = np.zeros_like(ds_time)
                ds_filtered = ds_tx_rate  # Use raw data as fallback
                st.warning("Traditional peak detection failed. Showing zeros instead.")
            else:
                # Use clipping_score as the detection signal (already 0.0 or 1.0)
                ds_peak_diff = ds_peaking
        else:
            st.warning("Missing tx_bytes data required for peak detection")
            ds_tx_rate = np.zeros_like(ds_time)
            ds_peaking = np.zeros_like(ds_time)
            ds_peak_diff = np.zeros_like(ds_time)
            ds_filtered = ds_tx_rate  # Use raw data as fallback
    
    # Prepare data for visualization
    queue_exists = downstream_df['queue_exists'].values if 'queue_exists' in downstream_df.columns else np.zeros_like(ds_time)
    time_array = downstream_df['relative_time'].values
    
    # Get network capacity
    network_capacity = 100
    if 'network_limit' in downstream_df.columns:
        network_limit = downstream_df['network_limit'].iloc[0]
        if network_limit > 0:
            network_capacity = round(network_limit * 1.15)
    
    # Add detailed peak detection visualization
    st.subheader("Detailed Peak Detection Analysis")
    
    if 'tx_bytes' in downstream_df.columns:
        # Get file name for title
        file_name = ""
        if 'file_name' in downstream_df.columns:
            file_name = str(downstream_df['file_name'].iloc[0])
        
        # Create the detailed visualization using the corrected throughput
        with st.spinner("Creating detailed peak detection visualization..."):            
            peak_viz_fig = create_peak_detection_visualization(
                ds_time, ds_tx_rate, network_capacity,
                long_window_ms, short_window_ms, peak_tolerance, count_threshold, 
                use_median_filter, file_name
            )
        
        if peak_viz_fig:
            st.plotly_chart(peak_viz_fig, use_container_width=True)
            
            # Add explanation text that changes based on filter setting
            if use_median_filter:
                st.markdown("""
                    **Understanding the Peak Detection Visualization (WITH Median Filtering):**
                    - **Red Line**: **Median-filtered** downstream throughput - what the algorithm analyzes after filtering
                    - **Black Dashed Line**: Network capacity limit 
                    - **Light Blue Areas**: Long windows showing maximum throughput from filtered data
                    - **Dark Blue Areas**: Peak tolerance zones - raw samples in this range count as "close to peak"
                    - **Orange Areas**: Short bins where clipping is detected (sustained high throughput near capacity)
                    - **N= Labels**: Number of **raw** samples within each orange short bin that fall in the tolerance zone
                    """)
            else:
                st.markdown("""
                    **Understanding the Peak Detection Visualization (WITHOUT Median Filtering):**
                    - **Red Line**: **Raw** downstream throughput
                    - **Black Dashed Line**: Network capacity limit 
                    - **Light Blue Areas**: Long windows showing maximum throughput from raw data
                    - **Dark Blue Areas**: Peak tolerance zones - samples in this range count as "close to peak"
                    - **Orange Areas**: Short bins where clipping is detected (sustained high throughput near capacity)
                    - **N= Labels**: Number of samples within each orange short bin that fall in the tolerance zone
                    """)
        else:
            st.error("Failed to create peak detection visualization")
    else:
        st.error("Cannot create peak detection visualization without tx_bytes data")
    
    # Apply optional time filter for PDF exports and CSV
    if limit_to_100s:
        PDF_TIME_LIMIT = 100  # seconds
        time_mask = time_array <= PDF_TIME_LIMIT
        st.info(f"PDF exports and CSV download will be limited to first {PDF_TIME_LIMIT} seconds of data.")
    else:
        time_mask = np.ones(len(time_array), dtype=bool)  # Include all data
        st.info("PDF exports and CSV download will include all available data.")
    
    # Apply time filter first
    time_array_filtered = time_array[time_mask]
    queue_exists_filtered = queue_exists[time_mask]
    ds_peak_diff_filtered = ds_peak_diff[time_mask]
    ds_tx_rate_filtered = ds_tx_rate[time_mask]
    ds_filtered_filtered = ds_filtered[time_mask]
    
    if downstream_predictions is not None:
        downstream_predictions_filtered = downstream_predictions[time_mask]
    else:
        downstream_predictions_filtered = None
    
    # Then downsample the filtered data
    downsample_step = max(1, len(time_array_filtered) // 1000)  # Limit to ~1000 points for performance
    time_array_downsampled = time_array_filtered[::downsample_step]
    queue_exists_downsampled = queue_exists_filtered[::downsample_step]
    ds_peak_diff_downsampled = ds_peak_diff_filtered[::downsample_step]
    ds_tx_rate_downsampled = ds_tx_rate_filtered[::downsample_step]
    ds_filtered_downsampled = ds_filtered_filtered[::downsample_step]
    
    if downstream_predictions_filtered is not None:
        downstream_predictions_downsampled = downstream_predictions_filtered[::downsample_step]
    else:
        downstream_predictions_downsampled = None
    
    # Create dictionary to store all figures for PDF export
    figures_for_export = {}
    
    # Figure 1 - Downstream Throughput (Median Filtered from peak detection)
    st.subheader("Downstream Throughput (Median Filtered)")
    
    fig_throughput = go.Figure()
    
    fig_throughput.add_trace(
        go.Scatter(
            x=time_array_downsampled, 
            y=ds_filtered_downsampled,  # Use the filtered throughput from peak detection
            name='Downstream Throughput (Filtered)',
            line=dict(color='blue', width=2),
            mode='lines'
        )
    )
    
    # Add network capacity line
    fig_throughput.add_trace(
        go.Scatter(
            x=[time_array_downsampled.min(), time_array_downsampled.max()],
            y=[network_capacity, network_capacity],
            name=f'Network Capacity ({network_capacity} Mbps)',
            line=dict(color='red', dash='dash', width=2),
            mode='lines'
        )
    )
    
    fig_throughput.update_layout(
        title="",  # Remove duplicate title
        xaxis_title="Time (s)",
        yaxis_title="Throughput (Mbps)",
        height=500,
        template='plotly_white',
        xaxis=dict(
            title=dict(font=dict(size=32)),
            tickfont=dict(size=28)
        ),
        yaxis=dict(
            title=dict(font=dict(size=32)),
            tickfont=dict(size=28)
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=24)
        )
    )
    
    st.plotly_chart(fig_throughput, use_container_width=True)
    figures_for_export['Ground_Truth_Throughput_Filtered'] = fig_throughput
    
    # Figure 2 - Throughput During Predicted Congestion (Both Methods) with filtering options
    if downstream_predictions_downsampled is not None:
        st.subheader(f"Throughput During Predicted Congestion (Traditional: LW:{long_window_ms}ms, SW:{short_window_ms}ms, PT:{peak_tolerance}, CT:{count_threshold})")
        
        # Add filter options
        st.write("Filter visualization:")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            show_non_congested = st.checkbox("Show Non-Congested", value=True)
        with col2:
            show_traditional = st.checkbox("Show Traditional Method", value=True)
        with col3:
            show_lstm = st.checkbox("Show LSTM Method", value=True)
        with col4:
            show_disagreement = st.checkbox("Show Disagreement Only", value=False)
        
        # Create congestion masks
        congestion_threshold = 0.5
        lstm_congestion_mask = downstream_predictions_downsampled >= congestion_threshold
        traditional_congestion_mask = ds_peak_diff_downsampled >= 0.5
        
        # Create specialized masks for filtering
        background_only_mask = ~(lstm_congestion_mask | traditional_congestion_mask)
        traditional_only_mask = traditional_congestion_mask & ~lstm_congestion_mask
        lstm_only_mask = lstm_congestion_mask & ~traditional_congestion_mask
        both_agree_mask = lstm_congestion_mask & traditional_congestion_mask
        
        # Apply the disagreement filter if selected
        if show_disagreement:
            # Override other selections to show only where methods disagree
            show_non_congested = False
            traditional_mask_to_use = traditional_only_mask
            lstm_mask_to_use = lstm_only_mask
            show_both_agree = False
        else:
            traditional_mask_to_use = traditional_congestion_mask
            lstm_mask_to_use = lstm_congestion_mask
            show_both_agree = True
        
        fig_lstm_throughput = go.Figure()
        
        fig_lstm_throughput.add_trace(
            go.Scatter(
                x=time_array_downsampled, 
                y=ds_filtered_downsampled,  # Use the filtered throughput from peak detection
                name='Downstream Throughput (Filtered)',
                line=dict(color='blue', width=1.5),
                mode='lines'
            )
        )
        
        # Add non-congested throughput as a filled area
        if show_non_congested:
            background_throughput = np.zeros_like(ds_filtered_downsampled)
            background_throughput[background_only_mask] = ds_filtered_downsampled[background_only_mask]
            fig_lstm_throughput.add_trace(
                go.Scatter(
                    x=time_array_downsampled,
                    y=background_throughput,
                    name='Non-Congested Throughput',
                    fill='tozeroy',
                    mode='none',
                    fillcolor="rgba(135, 206, 250, 0.8)"  # Bright sky blue filled area
                )
            )
        
        # Add traditional method's congestion as filled area
        if show_traditional:
            traditional_masked_throughput = np.zeros_like(ds_filtered_downsampled)
            
            if show_disagreement:
                # Only show where traditional detects but LSTM doesn't
                traditional_masked_throughput[traditional_only_mask] = ds_filtered_downsampled[traditional_only_mask]
                traditional_label = "Traditional Only (LSTM Disagrees)"
            else:
                traditional_masked_throughput[traditional_mask_to_use] = ds_filtered_downsampled[traditional_mask_to_use]
                traditional_label = "Heuristic Predicted Congestion"
                
            fig_lstm_throughput.add_trace(
                go.Scatter(
                    x=time_array_downsampled,
                    y=traditional_masked_throughput,
                    name=traditional_label,
                    fill='tozeroy',
                    mode='none',
                    fillcolor="rgba(255, 69, 0, 0.8)"  # Bright orange-red filled area
                )
            )
        
        # Add LSTM congestion as filled area
        if show_lstm:
            lstm_masked_throughput = np.zeros_like(ds_filtered_downsampled)
            
            if show_disagreement:
                # Only show where LSTM detects but traditional doesn't
                lstm_masked_throughput[lstm_only_mask] = ds_filtered_downsampled[lstm_only_mask]
                lstm_label = "LSTM Only (Traditional Disagrees)"
            else:
                lstm_masked_throughput[lstm_mask_to_use] = ds_filtered_downsampled[lstm_mask_to_use]
                lstm_label = "LSTM Predicted Congestion"
                
            fig_lstm_throughput.add_trace(
                go.Scatter(
                    x=time_array_downsampled,
                    y=lstm_masked_throughput,
                    name=lstm_label,
                    fill='tozeroy',
                    mode='none',
                    fillcolor="rgba(50, 205, 50, 0.8)"  # Bright lime green filled area
                )
            )
        
        # Add "both agree" area if showing both methods and not in disagreement mode
        if show_traditional and show_lstm and show_both_agree and not show_disagreement:
            both_agree_throughput = np.zeros_like(ds_filtered_downsampled)
            both_agree_throughput[both_agree_mask] = ds_filtered_downsampled[both_agree_mask]
            fig_lstm_throughput.add_trace(
                go.Scatter(
                    x=time_array_downsampled,
                    y=both_agree_throughput,
                    name="Both Methods Agree",
                    fill='tozeroy',
                    mode='none',
                    fillcolor="rgba(148, 0, 211, 0.8)"  # Purple filled area for agreement
                )
            )
        
        # Add network capacity line for reference
        fig_lstm_throughput.add_trace(
            go.Scatter(
                x=[time_array_downsampled.min(), time_array_downsampled.max()],
                y=[network_capacity, network_capacity],
                name=f'Network Capacity ({network_capacity} Mbps)',
                line=dict(color='#FF0000', dash='dash', width=2),  # Bright red
                mode='lines'
            )
        )
        
        fig_lstm_throughput.update_layout(
            title="",  # Remove duplicate title
            xaxis_title="Time (s)",
            yaxis_title="Throughput (Mbps)",
            height=500,
            template='plotly_white',
            xaxis=dict(
                title=dict(font=dict(size=32)),
                tickfont=dict(size=28)
            ),
            yaxis=dict(
                title=dict(font=dict(size=32)),
                tickfont=dict(size=28)
            ),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                font=dict(size=24)
            )
        )
        st.plotly_chart(fig_lstm_throughput, use_container_width=True)
        figures_for_export['Predicted_Congestion_Throughput'] = fig_lstm_throughput
        
        # Add explanation with current parameter values and color references
        if show_disagreement:
            st.markdown(f"""
            **Understanding the Disagreement Visualization:**
            - **<span style='color:#FF4500'>Orange-Red Filled Area</span>**: Throughput where only the traditional method detects congestion (LSTM disagrees)
            - **<span style='color:#32CD32'>Lime Green Filled Area</span>**: Throughput where only the LSTM method detects congestion (traditional disagrees)
            - **<span style='color:#FF0000'>Red Dashed Line</span>**: Network capacity limit ({network_capacity} Mbps)
            
            This visualization highlights areas where the two detection methods disagree with each other.
            """, unsafe_allow_html=True)
        else:
            explanation = """
            **Understanding the Predicted Congestion Throughput Graph:**
            """
            if show_non_congested:
                explanation += """
            - **<span style='color:#87CEFA'>Sky Blue Filled Area</span>**: Throughput during periods where neither method detects congestion
                """
            if show_traditional:
                explanation += f"""
            - **<span style='color:#FF4500'>Orange-Red Filled Area</span>**: Throughput during traditional method's predicted congestion 
            (Parameters: LW:{long_window_ms}ms, SW:{short_window_ms}ms, PT:{peak_tolerance}, CT:{count_threshold})
                """
            if show_lstm:
                explanation += f"""
            - **<span style='color:#32CD32'>Lime Green Filled Area</span>**: Throughput during LSTM-predicted congestion periods (probability ≥ {congestion_threshold})
                """
            if show_traditional and show_lstm and show_both_agree:
                explanation += """
            - **<span style='color:#9400D3'>Purple Filled Area</span>**: Throughput where both methods agree there is congestion
                """
            explanation += """
            - **<span style='color:#FF0000'>Red Dashed Line</span>**: Network capacity limit
            
            Use the checkboxes above to filter which detection results you want to visualize.
            """
            st.markdown(explanation, unsafe_allow_html=True)
    else:
        st.warning("LSTM predictions not available for congestion-specific throughput visualization")
        
    # NEW: Ground Truth Congestion Detection Comparison with Filtering Options
    if 'queue_size' in downstream_df.columns:
        st.subheader("Ground Truth Congestion Detection Comparison")
        
        # Create ground truth congestion mask (queue_size > 100 packets)
        queue_size_filtered = downstream_df['queue_size'].values[time_mask]
        queue_size_downsampled = queue_size_filtered[::downsample_step]
        ground_truth_congestion_mask = queue_size_downsampled > 100
        
        # Add filter options (same as predicted congestion figure)
        st.write("Filter visualization:")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            show_non_congested_gt = st.checkbox("Show Non-Congested", value=True, key="gt_non_congested")
        with col2:
            show_traditional_gt = st.checkbox("Show Traditional Method", value=True, key="gt_traditional")
        with col3:
            show_lstm_gt = st.checkbox("Show LSTM Method", value=True, key="gt_lstm") if downstream_predictions_downsampled is not None else False
        with col4:
            show_disagreement_gt = st.checkbox("Show Disagreement Only", value=False, key="gt_disagreement")
        
        # Create detection masks
        congestion_threshold = 0.5
        if downstream_predictions_downsampled is not None:
            lstm_congestion_mask = downstream_predictions_downsampled >= congestion_threshold
        else:
            lstm_congestion_mask = np.zeros_like(ground_truth_congestion_mask, dtype=bool)
            show_lstm_gt = False
            
        traditional_congestion_mask = ds_peak_diff_downsampled >= 0.5
        
        # Create specialized masks for filtering
        background_only_mask = ~(lstm_congestion_mask | traditional_congestion_mask)
        traditional_only_mask = traditional_congestion_mask & ~lstm_congestion_mask
        lstm_only_mask = lstm_congestion_mask & ~traditional_congestion_mask
        both_agree_mask = lstm_congestion_mask & traditional_congestion_mask
        
        # Apply the disagreement filter if selected
        if show_disagreement_gt:
            # Override other selections to show only where methods disagree
            show_non_congested_gt = False
            traditional_mask_to_use = traditional_only_mask
            lstm_mask_to_use = lstm_only_mask
            show_both_agree = False
        else:
            traditional_mask_to_use = traditional_congestion_mask
            lstm_mask_to_use = lstm_congestion_mask
            show_both_agree = True
        
        fig_gt_comparison = go.Figure()
        
        fig_gt_comparison.add_trace(
            go.Scatter(
                x=time_array_downsampled, 
                y=ds_filtered_downsampled,  # Use the filtered throughput from peak detection
                name='Downstream Throughput (Filtered)',
                line=dict(color='blue', width=1.5),
                mode='lines'
            )
        )
        
        # Add grey background areas for ground truth congestion up to network capacity
        # Create rectangles for each congested period
        in_congestion = False
        congestion_start = None
        
        for i, is_congested in enumerate(ground_truth_congestion_mask):
            if is_congested and not in_congestion:
                # Start of congestion period
                congestion_start = time_array_downsampled[i]
                in_congestion = True
            elif not is_congested and in_congestion:
                # End of congestion period
                congestion_end = time_array_downsampled[i-1] if i > 0 else time_array_downsampled[i]
                fig_gt_comparison.add_shape(
                    type="rect",
                    x0=congestion_start,
                    x1=congestion_end,
                    y0=0,
                    y1=network_capacity,
                    fillcolor="rgba(128, 128, 128, 0.3)",  # Grey background
                    line=dict(width=0),
                    layer="below"
                )
                in_congestion = False
        
        # Handle case where congestion extends to the end
        if in_congestion and congestion_start is not None:
            fig_gt_comparison.add_shape(
                type="rect",
                x0=congestion_start,
                x1=time_array_downsampled[-1],
                y0=0,
                y1=network_capacity,
                fillcolor="rgba(128, 128, 128, 0.3)",  # Grey background
                line=dict(width=0),
                layer="below"
            )
        
        # Add non-congested throughput as a filled area
        if show_non_congested_gt:
            background_throughput = np.zeros_like(ds_filtered_downsampled)
            background_throughput[background_only_mask] = ds_filtered_downsampled[background_only_mask]
            fig_gt_comparison.add_trace(
                go.Scatter(
                    x=time_array_downsampled,
                    y=background_throughput,
                    name='Non-Congested Throughput',
                    fill='tozeroy',
                    mode='none',
                    fillcolor="rgba(135, 206, 250, 0.8)"  # Bright sky blue filled area
                )
            )
        
        # Add traditional method's congestion as filled area
        if show_traditional_gt:
            traditional_masked_throughput = np.zeros_like(ds_filtered_downsampled)
            
            if show_disagreement_gt:
                # Only show where traditional detects but LSTM doesn't
                traditional_masked_throughput[traditional_only_mask] = ds_filtered_downsampled[traditional_only_mask]
                traditional_label = "Traditional Only (LSTM Disagrees)"
            else:
                traditional_masked_throughput[traditional_mask_to_use] = ds_filtered_downsampled[traditional_mask_to_use]
                traditional_label = "Traditional Method Congestion"
                
            fig_gt_comparison.add_trace(
                go.Scatter(
                    x=time_array_downsampled,
                    y=traditional_masked_throughput,
                    name=traditional_label,
                    fill='tozeroy',
                    mode='none',
                    fillcolor="rgba(255, 69, 0, 0.8)"  # Bright orange-red filled area
                )
            )
        
        # Add LSTM congestion as filled area
        if show_lstm_gt and downstream_predictions_downsampled is not None:
            lstm_masked_throughput = np.zeros_like(ds_filtered_downsampled)
            
            if show_disagreement_gt:
                # Only show where LSTM detects but traditional doesn't
                lstm_masked_throughput[lstm_only_mask] = ds_filtered_downsampled[lstm_only_mask]
                lstm_label = "LSTM Only (Traditional Disagrees)"
            else:
                lstm_masked_throughput[lstm_mask_to_use] = ds_filtered_downsampled[lstm_mask_to_use]
                lstm_label = "LSTM Predicted Congestion"
                
            fig_gt_comparison.add_trace(
                go.Scatter(
                    x=time_array_downsampled,
                    y=lstm_masked_throughput,
                    name=lstm_label,
                    fill='tozeroy',
                    mode='none',
                    fillcolor="rgba(50, 205, 50, 0.8)"  # Bright lime green filled area
                )
            )
        
        # Add "both agree" area if showing both methods and not in disagreement mode
        if show_traditional_gt and show_lstm_gt and show_both_agree and not show_disagreement_gt:
            both_agree_throughput = np.zeros_like(ds_filtered_downsampled)
            both_agree_throughput[both_agree_mask] = ds_filtered_downsampled[both_agree_mask]
            fig_gt_comparison.add_trace(
                go.Scatter(
                    x=time_array_downsampled,
                    y=both_agree_throughput,
                    name="Both Methods Agree",
                    fill='tozeroy',
                    mode='none',
                    fillcolor="rgba(148, 0, 211, 0.8)"  # Purple filled area for agreement
                )
            )
        
        # Add network capacity line for reference
        fig_gt_comparison.add_trace(
            go.Scatter(
                x=[time_array_downsampled.min(), time_array_downsampled.max()],
                y=[network_capacity, network_capacity],
                name=f'Network Capacity ({network_capacity} Mbps)',
                line=dict(color='#FF0000', dash='dash', width=2),  # Bright red
                mode='lines'
            )
        )
        
        # Add invisible trace for grey background legend
        fig_gt_comparison.add_trace(
            go.Scatter(
                x=[time_array_downsampled[0], time_array_downsampled[0]],
                y=[0, network_capacity],
                fill='tozeroy',
                fillcolor='rgba(128, 128, 128, 0.3)',
                line=dict(color='rgba(128, 128, 128, 0.3)'),
                name='Ground Truth Congestion Zone',
                showlegend=True,
                mode='none'
            )
        )
        
        fig_gt_comparison.update_layout(
            title="",  # Remove duplicate title
            xaxis_title="Time (s)",
            yaxis_title="Throughput (Mbps)",
            height=500,
            template='plotly_white',
            xaxis=dict(
                title=dict(font=dict(size=32)),
                tickfont=dict(size=28)
            ),
            yaxis=dict(
                title=dict(font=dict(size=32)),
                tickfont=dict(size=28)
            ),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                font=dict(size=24)
            )
        )
        
        st.plotly_chart(fig_gt_comparison, use_container_width=True)
        figures_for_export['Ground_Truth_Congestion_Comparison'] = fig_gt_comparison
        
        # Add explanation with current parameter values and color references
        if show_disagreement_gt:
            st.markdown(f"""
            **Understanding the Ground Truth Disagreement Visualization:**
            - **<span style='color:#808080'>Grey Background</span>**: Actual congestion periods (queue > 100 packets)
            - **<span style='color:#FF4500'>Orange-Red Filled Area</span>**: Throughput where only the traditional method detects congestion (LSTM disagrees)
            - **<span style='color:#32CD32'>Lime Green Filled Area</span>**: Throughput where only the LSTM method detects congestion (traditional disagrees)
            - **<span style='color:#FF0000'>Red Dashed Line</span>**: Network capacity limit ({network_capacity} Mbps)
            
            This shows how the detection methods disagree against the ground truth congestion periods.
            """, unsafe_allow_html=True)
        else:
            explanation = """
            **Understanding the Ground Truth Congestion Detection Comparison:**
            - **<span style='color:#808080'>Grey Background</span>**: Actual congestion periods (queue > 100 packets)
            """
            if show_non_congested_gt:
                explanation += """
            - **<span style='color:#87CEFA'>Sky Blue Filled Area</span>**: Throughput during periods where neither method detects congestion
                """
            if show_traditional_gt:
                explanation += f"""
            - **<span style='color:#FF4500'>Orange-Red Filled Area</span>**: Throughput during traditional method's predicted congestion 
            (Parameters: LW:{long_window_ms}ms, SW:{short_window_ms}ms, PT:{peak_tolerance}, CT:{count_threshold})
                """
            if show_lstm_gt:
                explanation += f"""
            - **<span style='color:#32CD32'>Lime Green Filled Area</span>**: Throughput during LSTM-predicted congestion periods (probability ≥ {congestion_threshold})
                """
            if show_traditional_gt and show_lstm_gt and show_both_agree:
                explanation += """
            - **<span style='color:#9400D3'>Purple Filled Area</span>**: Throughput where both methods agree there is congestion
                """
            explanation += """
            - **<span style='color:#FF0000'>Red Dashed Line</span>**: Network capacity limit
            
            Use the checkboxes above to filter which detection results you want to visualize against the ground truth.
            """
            st.markdown(explanation, unsafe_allow_html=True)
    else:
        st.warning("Queue size data not available for ground truth congestion comparison")
    
    # Create visualizations for actual vs predicted buffer occupancy
    st.subheader("Buffer Occupancy Prediction Comparison")
    
    # Figure 3: Ground Truth Buffer Occupancy  
    if 'queue_size' in downstream_df.columns:
        queue_size = downstream_df['queue_size'].values
        queue_size_filtered = queue_size[time_mask]
        queue_size_downsampled = queue_size_filtered[::downsample_step]
        
        # Calculate maximum queue size for scaling
        max_queue_size = max(1, np.max(queue_size_downsampled))
        
        fig_ground_truth = go.Figure()
        
        fig_ground_truth.add_trace(
            go.Scatter(
                x=time_array_downsampled, 
                y=queue_size_downsampled,
                name='Buffer Occupancy',
                fill='tozeroy',
                mode='none',
                fillcolor="rgba(255,100,100,0.5)"
            )
        )
        
        fig_ground_truth.update_layout(
            title="",  # Remove duplicate title
            xaxis_title="Time (s)",
            yaxis_title="Queue Size (packets)",
            height=500,
            template='plotly_white',
            xaxis=dict(
                title=dict(font=dict(size=32)),
                tickfont=dict(size=28)
            ),
            yaxis=dict(
                title=dict(font=dict(size=32)),
                tickfont=dict(size=28)
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                font=dict(size=24)
            )
        )
        
        st.plotly_chart(fig_ground_truth, use_container_width=True)
        figures_for_export['Ground_Truth_Buffer_Occupancy'] = fig_ground_truth
    else:
        st.warning("Queue size data not available for ground truth visualization")
    
    # Figure 4: Traditional Peak Detection
    st.subheader(f"Traditional Peak Detection (LW: {long_window_ms}ms, SW: {short_window_ms}ms, PT: {peak_tolerance}, CT: {count_threshold})")
    fig_traditional = go.Figure()
    
    # Add traditional peak detection
    fig_traditional.add_trace(
        go.Scatter(
            x=time_array_downsampled,
            y=ds_peak_diff_downsampled,
            name='Traditional Detection',
            fill='tozeroy',
            mode='none',
            fillcolor="rgba(255,165,0,0.5)"
        )
    )
    
    fig_traditional.update_layout(
        title="",  # Remove duplicate title
        xaxis_title="Time (s)",
        yaxis_title="Clipping Detected (0/1)",
        height=500,
        yaxis=dict(
            range=[0, 1.1], 
            title=dict(font=dict(size=32)),
            tickfont=dict(size=28)
        ),
        xaxis=dict(
            title=dict(font=dict(size=32)),
            tickfont=dict(size=28)
        ),
        template='plotly_white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=24)
        )
    )
    
    st.plotly_chart(fig_traditional, use_container_width=True)
    figures_for_export['Traditional_Peak_Detection'] = fig_traditional
    
    # Figure 5: LSTM Prediction
    if downstream_predictions_downsampled is not None:
        st.subheader("LSTM Prediction - Queue Probability")
        fig_lstm = go.Figure()
        
        fig_lstm.add_trace(
            go.Scatter(
                x=time_array_downsampled,
                y=downstream_predictions_downsampled,
                name='LSTM Prediction',
                fill='tozeroy',
                mode='none',
                fillcolor="rgba(100,255,100,0.5)"
            )
        )
        
        fig_lstm.update_layout(
            title="",  # Remove duplicate title
            xaxis_title="Time (s)",
            yaxis_title="Queue Probability",
            height=500,
            yaxis=dict(
                range=[0, 1], 
                title=dict(font=dict(size=32)),
                tickfont=dict(size=28)
            ),
            xaxis=dict(
                title=dict(font=dict(size=32)),
                tickfont=dict(size=28)
            ),
            template='plotly_white',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                font=dict(size=24)
            )
        )
        
        st.plotly_chart(fig_lstm, use_container_width=True)
        figures_for_export['LSTM_Prediction'] = fig_lstm
    else:
        st.warning("LSTM predictions not available")
    
    # Add peak detection visualization to export figures
    if 'peak_viz_fig' in locals() and peak_viz_fig is not None:
        figures_for_export['Detailed_Peak_Detection'] = peak_viz_fig
    
    # PDF Export Section
    st.subheader("Export Graphs as PDF")
    
    # Get experiment info for filename
    experiment_name = "unknown_experiment"
    subfolder = "unknown_subfolder"
    
    if 'file_name' in downstream_df.columns:
        experiment_name = str(downstream_df['file_name'].iloc[0])
    
    # Create download links for all figures
    if figures_for_export:
        st.markdown("### Download Individual Graphs:")
        download_links = save_all_figures_as_pdf(figures_for_export, experiment_name, subfolder)
        
        for link in download_links:
            st.markdown(link, unsafe_allow_html=True)
        
        st.markdown("---")
        time_filter_note = f"first {100}s" if limit_to_100s else "all available time"
        st.info(f"PDF downloads show {time_filter_note} of data. Toggle 'Limit PDFs to 100s' to change this.")
    
    # Calculate and display metrics if both methods are available
    if downstream_predictions is not None and 'queue_exists' in downstream_df.columns:
        st.subheader("Detection Performance Metrics")
        
        # Apply time filter to predictions for metrics
        queue_exists_for_metrics = queue_exists_filtered
        downstream_predictions_for_metrics = downstream_predictions_filtered if downstream_predictions_filtered is not None else downstream_predictions[time_mask]
        ds_peaking_for_metrics = ds_peak_diff_filtered
        
        # Convert predictions to binary for metrics calculation
        binary_downstream_predictions = (downstream_predictions_for_metrics >= 0.5).astype(int)
        
        # Calculate metrics for both methods
        traditional_cm, traditional_report = calculate_lstm_metrics(queue_exists_for_metrics, ds_peaking_for_metrics.astype(int))
        lstm_cm, lstm_report = calculate_lstm_metrics(queue_exists_for_metrics, binary_downstream_predictions)
        
        # Display confusion matrices
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Traditional Peak Detection**")
            traditional_cm_fig = plot_confusion_matrix(traditional_cm, "Traditional Detection")
            st.plotly_chart(traditional_cm_fig, use_container_width=True)
            figures_for_export['Traditional_Confusion_Matrix'] = traditional_cm_fig
            
        with col2:
            st.markdown("**LSTM Prediction**")
            lstm_cm_fig = plot_confusion_matrix(lstm_cm, "LSTM Prediction")
            st.plotly_chart(lstm_cm_fig, use_container_width=True)
            figures_for_export['LSTM_Confusion_Matrix'] = lstm_cm_fig
        
        # Display comparison table
        comparison = create_comparison_dataframe(traditional_report, lstm_report)
        if comparison is not None:
            st.subheader("Comparison of Detection Methods")
            st.dataframe(comparison)
        
        # Add explanatory text
        time_filter_note = f"first {100} seconds" if limit_to_100s else "entire dataset"
        st.markdown(f"""
        ### Understanding the Metrics
        - **Accuracy**: Overall correct predictions (both positive and negative)
        - **Precision**: When a method predicts a queue exists, how often is it correct?
        - **Recall**: Of all actual queue occurrences, how many were correctly detected?
        - **F1-Score**: Balanced measure of precision and recall
        
        ### Comparison of Methods
        - **Traditional Peak Detection**: Rule-based algorithm that identifies congestion based on throughput clipping patterns
        - **LSTM Model**: Machine learning approach that learns temporal patterns that may precede congestion
        
        **Note**: Metrics calculated on {time_filter_note} based on the "Limit PDFs to 100s" setting.
        """)
        
    # CSV Export Section
    st.subheader("Download Time Series Data")
    
    # Prepare the data for export
    queue_size_for_export = None
    if 'queue_size' in downstream_df.columns:
        queue_size_filtered = downstream_df['queue_size'].values[time_mask]
        queue_size_for_export = queue_size_filtered[::downsample_step]

    export_df = create_time_series_csv(
        time_array_downsampled,
        ds_filtered_downsampled, 
        queue_exists_downsampled,
        ds_peak_diff_downsampled,
        downstream_predictions_downsampled,
        queue_size_for_export
    )

    # Create filename
    experiment_name = "unknown_experiment"
    if 'file_name' in downstream_df.columns:
        experiment_name = str(downstream_df['file_name'].iloc[0])

    time_suffix = "_first_100s" if limit_to_100s else "_full_data"
    csv_filename = f"{experiment_name}_detection_comparison_data{time_suffix}.csv"

    # Convert DataFrame to CSV string
    csv_string = export_df.to_csv(index=False)

    # Create download button
    time_filter_note = f"first {100}s" if limit_to_100s else "all available time"
    st.download_button(
        label=f"📊 Download Time Series Data as CSV ({time_filter_note})",
        data=csv_string,
        file_name=csv_filename,
        mime='text/csv',
        help=f"Download the time series data ({time_filter_note}) used for visualization and confusion matrix calculations"
    )

    # Show preview of the data
    with st.expander("Preview of data to be downloaded"):
        st.dataframe(export_df.head(10))
        st.write(f"Total rows: {len(export_df)}")
        st.write(f"Columns: {', '.join(export_df.columns)}")
        st.write(f"Time range: {time_filter_note}")

# ===========================================
# Main Application
# ===========================================
def main():
    st.title("Network Throughput Analysis")
    
    try:
        # Load LSTM predictor at startup
        lstm_models = load_lstm_models()
        has_lstm = lstm_models is not None
        
        if has_lstm:
            st.sidebar.success(f"LSTM models loaded: {len(lstm_models)} models")
            for model_name, model_info in lstm_models.items():
                st.sidebar.info(f"Model: {model_name} - {os.path.basename(model_info['path'])}")
        else:
            st.sidebar.warning("LSTM models not found. LSTM prediction unavailable.")
        
        # Get available subfolders
        with st.spinner("Loading available data folders..."):
            subfolders = get_available_subfolders()
        
        if not subfolders:
            st.error(f"No subfolders found in {EXTRACTED_DATA_FOLDER} directory.")
            return
            
        # Add subfolder selector
        selected_subfolder = st.sidebar.selectbox(
            "Select Data Folder",
            options=subfolders,
            help="Choose which folder's data to analyze"
        )

        # Find all CSV files in the selected subfolder
        with st.spinner("Loading CSV files..."):
            csv_files = get_csv_files(selected_subfolder)
        
        if not csv_files:
            st.error(f"No CSV files found in '{os.path.join(EXTRACTED_DATA_FOLDER, selected_subfolder)}'")
            return

        # Group CSV files by base name (pairing upstream and downstream)
        experiments = {}
        for file in csv_files:
            if 'downstream' in file or 'upstream' in file:
                base_name = file.replace('_downstream.csv', '').replace('_upstream.csv', '')
                if base_name not in experiments:
                    experiments[base_name] = {
                        'downstream': None,
                        'upstream': None
                    }
                    
                if 'downstream' in file:
                    experiments[base_name]['downstream'] = file
                elif 'upstream' in file:
                    experiments[base_name]['upstream'] = file
        
        if not experiments:
            st.error("No downstream/upstream file pairs found. Files need '_downstream.csv' or '_upstream.csv' suffix.")
            return

        # Experiment selection
        selected_exp = st.sidebar.selectbox(
            "Select experiment:",
            options=list(experiments.keys())
        )

        # Page selection (data visualization or detection comparison)
        page = st.sidebar.radio(
            "Select Page",
            ["Data Visualization", "Detection Comparison"]
        )

        if selected_exp:
            exp_files = experiments[selected_exp]
            try:
                # Start timing
                start_time = time.time()
                
                # Load data
                with st.spinner("Loading and processing data..."):
                    downstream_df = None
                    upstream_df = None
                    
                    if exp_files['downstream']:
                        downstream_df = load_processed_data(selected_subfolder, exp_files['downstream'])
                    
                    if exp_files['upstream']:
                        upstream_df = load_processed_data(selected_subfolder, exp_files['upstream'])
                    
                    if downstream_df is None and upstream_df is None:
                        st.error("Failed to load any data files")
                        return
                
                # Check for feature files
                downstream_features_df = None
                if downstream_df is not None and exp_files['downstream']:
                    feature_file_downstream = exp_files['downstream'].replace('.csv', '_features.csv')
                    
                    # Try to load features from feature_data folder
                    with st.spinner("Looking for feature data..."):
                        downstream_features_df = load_processed_data(selected_subfolder, feature_file_downstream, use_features=True)
                        
                        if downstream_features_df is not None:
                            st.success(f"Found features file: {feature_file_downstream}")
                        else:
                            st.warning(f"Feature file not found: {feature_file_downstream}")
                
                # Show data range info
                max_time = 0
                if downstream_df is not None and 'relative_time' in downstream_df.columns:
                    max_time = max(max_time, downstream_df['relative_time'].max())
                
                if upstream_df is not None and 'relative_time' in upstream_df.columns:
                    max_time = max(max_time, upstream_df['relative_time'].max())
                
                if max_time > 0:
                    st.info(f"Dataset spans from 0 to {max_time:.1f} seconds.")
                
                # Show appropriate page based on selection
                if page == "Data Visualization":
                    # Ensure both dataframes exist for data visualization
                    if downstream_df is None:
                        downstream_df = pd.DataFrame({'relative_time': [0, 1], 'tx_bytes': [0, 0], 'rx_bytes': [0, 0]})
                        st.warning("No downstream data available. Showing placeholder data.")
                    
                    if upstream_df is None:
                        upstream_df = pd.DataFrame({'relative_time': [0, 1], 'tx_bytes': [0, 0], 'rx_bytes': [0, 0]})
                        st.warning("No upstream data available. Showing placeholder data.")
                        
                    data_visualization_page(downstream_df, upstream_df)
                else:  # Detection Comparison
                    if downstream_df is None:
                        st.error("Downstream data is required for detection comparison")
                        return
                        
                    detection_comparison_page(downstream_df, downstream_features_df, lstm_models if has_lstm else None)
                
                # Show processing time
                end_time = time.time()
                processing_time = end_time - start_time
                st.success(f"Analysis completed in {processing_time:.2f} seconds.")

            except Exception as e:
                st.error(f"Error processing data: {str(e)}")
                st.exception(e)

    except Exception as e:
        st.error(f"Application Error: {str(e)}")
        st.exception(e)
        
if __name__ == "__main__":
    main()