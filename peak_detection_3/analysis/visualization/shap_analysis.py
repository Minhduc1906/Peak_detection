import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import shap
import re
from sklearn.preprocessing import StandardScaler, LabelEncoder
from collections import defaultdict

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=50, num_layers=2, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 25)
        self.fc2 = nn.Linear(25, output_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        batch_size = x.size(0)
        device = x.device
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        return self.sigmoid(self.fc2(out))

# Model wrapper for SHAP
class LSTMModelWrapper:
    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names
        
    def __call__(self, x):
        # Reshape from 2D to 3D for LSTM
        num_samples = x.shape[0]
        num_features = len(self.feature_names)
        time_steps = x.shape[1] // num_features
        
        x_reshaped = x.reshape(num_samples, time_steps, num_features)
        x_tensor = torch.FloatTensor(x_reshaped).to(self.model.lstm.weight_ih_l0.device)
        
        with torch.no_grad():
            output = self.model(x_tensor).cpu().numpy()
        
        return output

# Helper functions
def create_sequences(X, y, time_steps=10):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

def preprocess_data(df, feature_names=None, time_steps=10, scaler=None):
    """Preprocess data for SHAP analysis using specified feature_names if provided"""
    # Handle missing values
    data_copy = df.copy()
    
    # First, explicitly drop categorical columns
    categorical_cols = ['direction', 'file_name', 'file_name_encoded', 'subfolder', 'testcase_with_limit']
    data_copy = data_copy.drop(columns=[col for col in categorical_cols if col in data_copy.columns])
    
    # Handle remaining non-numeric columns
    object_columns = data_copy.select_dtypes(include=['object']).columns
    if not object_columns.empty:
        print(f"Dropping non-numeric columns: {list(object_columns)}")
        data_copy = data_copy.drop(columns=object_columns)
    
    # Replace infinities with NaN then fill with max value
    for col in data_copy.columns:
        if data_copy[col].dtype.kind in 'if':  # Only process numeric columns
            data_copy[col] = data_copy[col].replace([float('inf'), float('-inf')], float('nan'))
    data_copy = data_copy.fillna(data_copy.max())
    
    # Select target variable
    y = data_copy['queue_exists'].values
    
    # If specific feature names are provided, use only those
    if feature_names:
        # Check if all required features exist in the data
        missing_features = [f for f in feature_names if f not in data_copy.columns]
        if missing_features:
            print(f"Warning: Some specified features are missing from data: {missing_features}")
            # Use only available features
            available_features = [f for f in feature_names if f in data_copy.columns]
            print(f"Using only available features: {available_features}")
            feature_cols = available_features
        else:
            print(f"Using specified features: {feature_names}")
            feature_cols = feature_names
    else:
        # Auto-detect features
        feature_cols = [c for c in data_copy.columns if c != 'queue_size' and  c != 'queue_exists' and c != 'detection_accuracy' 
                      and c != "time" and c != "relative_time" and c != 'network_limit']
        print(f"Auto-detected features: {feature_cols}")
    
    # Extract feature data
    X = data_copy[feature_cols].values
    
    # Double-check that X contains only numeric data
    if not np.issubdtype(X.dtype, np.number):
        print("WARNING: X contains non-numeric data. Converting to float...")
        print(f"X dtype: {X.dtype}")
        print(f"Sample X data: {X[:5]}")
        X = X.astype(float)
    
    # Scale features
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)
    
    # Create time sequences
    X_seq, y_seq = create_sequences(X_scaled, y, time_steps)
    return X_seq, y_seq, feature_cols, scaler

def load_model(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device)
    return checkpoint['model_state_dict'], checkpoint.get('time_steps', 10)

def compute_shap_values(model, X, feature_names, sample_size=100, background_size=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    # Create model wrapper and flatten input
    model_wrapper = LSTMModelWrapper(model, feature_names)
    num_samples, time_steps, num_features = X.shape
    X_flat = X.reshape(num_samples, time_steps * num_features)
    
    # Sample data for analysis
    if sample_size < num_samples:
        X_sample = X_flat[np.random.choice(num_samples, sample_size, replace=False)]
    else:
        X_sample = X_flat
    
    # Create background data
    if background_size < num_samples:
        background = X_flat[np.random.choice(num_samples, background_size, replace=False)]
    else:
        background = X_flat
    
    # Create feature names with time steps
    timestep_feature_names = []
    for t in range(time_steps):
        time_offset = time_steps - t
        for feature in feature_names:
            timestep_feature_names.append(f"{feature}_t-{time_offset-1}")
    
    # Compute SHAP values
    explainer = shap.KernelExplainer(model_wrapper, background)
    shap_values = explainer.shap_values(X_sample)
    
    return shap_values, timestep_feature_names, X_sample

def analyze_feature_importance(shap_values, feature_names, output_dir, time_steps=10):
    os.makedirs(output_dir, exist_ok=True)
    
    # Handle different SHAP value structures
    if isinstance(shap_values, list):
        shap_values_to_use = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    else:
        shap_values_to_use = shap_values
    
    # Handle 3D SHAP values
    if len(shap_values_to_use.shape) == 3:
        shap_values_to_use = shap_values_to_use.reshape(shap_values_to_use.shape[0], -1)
    
    # Calculate mean absolute SHAP values
    mean_abs_shap = np.abs(shap_values_to_use).mean(0)
    if mean_abs_shap.ndim > 1:
        mean_abs_shap = mean_abs_shap.flatten()
    
    # Ensure feature names and SHAP values have matching dimensions
    if len(feature_names) != len(mean_abs_shap):
        min_len = min(len(feature_names), len(mean_abs_shap))
        feature_names = feature_names[:min_len]
        mean_abs_shap = mean_abs_shap[:min_len]
    
    # Create DataFrame for per-time-step feature importance
    feature_importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance": mean_abs_shap
    }).sort_values("Importance", ascending=False)
    
    # Aggregate importance by base feature name
    base_importance = defaultdict(float)
    for i, feature in enumerate(feature_names):
        base_name = feature.split('_t-')[0]
        base_importance[base_name] += mean_abs_shap[i]
    
    agg_importance = pd.DataFrame({
        "Feature": list(base_importance.keys()),
        "Importance": list(base_importance.values())
    }).sort_values("Importance", ascending=False)
    
    # Save to CSV
    feature_importance.to_csv(os.path.join(output_dir, "feature_importance.csv"), index=False)
    agg_importance.to_csv(os.path.join(output_dir, "aggregated_feature_importance.csv"), index=False)
    
    # Plot aggregated feature importance
    try:
        plt.figure(figsize=(12, 8))
        top_n = min(10, len(agg_importance))
        plt.barh(agg_importance["Feature"][:top_n], agg_importance["Importance"][:top_n])
        plt.title('Aggregated Feature Importance')
        plt.xlabel('Mean |SHAP Value|')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "aggregated_feature_importance.png"))
        plt.close()
    except Exception as e:
        print(f"Error saving aggregated feature importance plot: {e}")
    
    return agg_importance

def main():
    # Setup
    results_dir = 'results'
    models_dir = os.path.join(results_dir, 'models')
    output_dir = os.path.join(results_dir, 'shap_analysis')
    finetune_output_dir = os.path.join(output_dir, 'finetune_model')
    combined_output_dir = os.path.join(output_dir, 'combined_model')
    
    # Create directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(finetune_output_dir, exist_ok=True)
    os.makedirs(combined_output_dir, exist_ok=True)
    
    # Load data
    data = pd.read_csv('combined_features.csv')
    downstream_data = data[data['direction'] == 'downstream'].copy()
    
    # Sample data for faster processing
    sample_count = 1000
    sampled_data = downstream_data.sample(n=min(sample_count, len(downstream_data)), random_state=42)
    
    # Load models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load finetune model
    finetune_model_path = os.path.join(models_dir, 'finetune_final_model.pt')
    if not os.path.exists(finetune_model_path):
        finetune_files = [f for f in os.listdir(models_dir) if f.startswith('finetune_step') and f.endswith('.pt')]
        if finetune_files:
            step_nums = [int(re.search(r'step(\d+)', f).group(1)) for f in finetune_files]
            highest_step = max(step_nums)
            finetune_model_path = os.path.join(models_dir, [f for f in finetune_files if f'step{highest_step}' in f][0])
    
    try:
        finetune_state_dict, finetune_time_steps = load_model(finetune_model_path)
    except KeyError as e:
        print(f"Error: Checkpoint missing key {e}. Using default time_steps=10.")
        finetune_state_dict = torch.load(finetune_model_path, map_location=device)
        finetune_time_steps = 10
    
    # Check for combined model
    combined_model_path = os.path.join(models_dir, 'combined_model_downstream_only.pt')
    has_combined = os.path.exists(combined_model_path)
    
    if has_combined:
        try:
            combined_state_dict, combined_time_steps = load_model(combined_model_path)
        except KeyError as e:
            print(f"Error: Checkpoint missing key {e}. Using default time_steps=10.")
            combined_state_dict = torch.load(combined_model_path, map_location=device)
            combined_time_steps = 10
    
    # Prepare data
    try:
        X_seq, y_seq, feature_names, scaler = preprocess_data(
            sampled_data, time_steps=finetune_time_steps
        )
    except Exception as e:
        print(f"Error in preprocessing data: {e}")
        return
    
    # Initialize finetune model
    finetune_model = LSTMModel(input_size=X_seq.shape[2])
    finetune_model.load_state_dict(finetune_state_dict)
    finetune_model.to(device)
    
    # Analyze finetune model
    print("Analyzing finetune model with SHAP...")
    
    # Use smaller sample for SHAP analysis
    shap_sample_size = min(100, len(X_seq))
    background_size = min(100, len(X_seq))
    
    # shap_sample_size = len(X_seq)
    # background_size = len(X_seq)
    
    try:
        finetune_shap_values, finetune_feature_names, X_finetune_sample = compute_shap_values(
            finetune_model, X_seq, feature_names, shap_sample_size, background_size
        )
    except Exception as e:
        print(f"Error computing SHAP values for finetune model: {e}")
        return
    
    # Create aggregated SHAP values and data
    finetune_shap_values_to_use = finetune_shap_values
    if isinstance(finetune_shap_values, list):
        finetune_shap_values_to_use = finetune_shap_values[1] if len(finetune_shap_values) > 1 else finetune_shap_values[0]
    if len(finetune_shap_values_to_use.shape) == 3:
        finetune_shap_values_to_use = finetune_shap_values_to_use.reshape(finetune_shap_values_to_use.shape[0], -1)
    
    # Aggregate SHAP values by base feature
    num_features = len(feature_names)
    num_samples = finetune_shap_values_to_use.shape[0]
    agg_shap_values = np.zeros((num_samples, num_features))
    for i, feature in enumerate(finetune_feature_names):
        base_name = feature.split('_t-')[0]
        base_idx = feature_names.index(base_name)
        agg_shap_values[:, base_idx] += finetune_shap_values_to_use[:, i]
    
    # Aggregate input data by averaging across time steps
    X_agg = X_finetune_sample.reshape(num_samples, finetune_time_steps, num_features).mean(axis=1)
    
    # Ensure dimensions match
    if agg_shap_values.shape[1] != len(feature_names):
        print(f"Warning: Aggregated SHAP values dimension ({agg_shap_values.shape[1]}) does not match feature names ({len(feature_names)}).")
        return
    
    # Compute base_values using the model wrapper
    model_wrapper = LSTMModelWrapper(finetune_model, feature_names)
    base_values = np.mean(model_wrapper(X_finetune_sample))
    
    # Create SHAP Explanation object for aggregated features
    finetune_explanation_agg = shap.Explanation(
        values=agg_shap_values,
        base_values=base_values,
        data=X_agg,
        feature_names=feature_names
    )
    
    # Generate aggregated waterfall plot
    try:
        plt.figure(figsize=(12, 8))
        shap.plots.waterfall(finetune_explanation_agg[0], max_display=10, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(finetune_output_dir, "aggregated_waterfall_plot.png"))
        plt.close()
    except Exception as e:
        print(f"Error saving aggregated waterfall plot for finetune model: {e}")
    
    # Generate aggregated beeswarm plot
    try:
        plt.figure(figsize=(12, 8))
        shap.plots.beeswarm(finetune_explanation_agg, max_display=10, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(finetune_output_dir, "aggregated_beeswarm_plot.png"))
        plt.close()
    except Exception as e:
        print(f"Error saving aggregated beeswarm plot for finetune model: {e}")
    
    # Generate aggregated bar plot
    try:
        plt.figure(figsize=(12, 8))
        shap.summary_plot(agg_shap_values, X_agg, feature_names=feature_names, 
                          plot_type="bar", show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(finetune_output_dir, "aggregated_summary_bar_plot.png"))
        plt.close()
    except Exception as e:
        print(f"Error saving aggregated bar plot for finetune model: {e}")
    
    # Analyze feature importance (already aggregated)
    try:
        finetune_agg_importance = analyze_feature_importance(
            finetune_shap_values, finetune_feature_names, 
            finetune_output_dir, time_steps=finetune_time_steps
        )
    except Exception as e:
        print(f"Error analyzing feature importance for finetune model: {e}")
        return
    
    # If combined model exists, analyze it too
    if has_combined:
        print("Analyzing combined model with SHAP...")
        
        # Prepare data if needed
        if combined_time_steps != finetune_time_steps:
            try:
                X_seq_combined, y_seq_combined, feature_names, _ = preprocess_data(
                    sampled_data, time_steps=combined_time_steps, scaler=scaler
                )
            except Exception as e:
                print(f"Error in preprocessing data for combined model: {e}")
                return
        else:
            X_seq_combined, y_seq_combined = X_seq, y_seq
        
        # Initialize combined model
        combined_model = LSTMModel(input_size=X_seq_combined.shape[2])
        combined_model.load_state_dict(combined_state_dict)
        combined_model.to(device)
        
        # Compute SHAP values
        try:
            combined_shap_values, combined_feature_names, X_combined_sample = compute_shap_values(
                combined_model, X_seq_combined, feature_names, shap_sample_size, background_size
            )
        except Exception as e:
            print(f"Error computing SHAP values for combined model: {e}")
            return
        
        # Create aggregated SHAP values and data
        combined_shap_values_to_use = combined_shap_values
        if isinstance(combined_shap_values, list):
            combined_shap_values_to_use = combined_shap_values[1] if len(combined_shap_values) > 1 else combined_shap_values[0]
        if len(combined_shap_values_to_use.shape) == 3:
            combined_shap_values_to_use = combined_shap_values_to_use.reshape(combined_shap_values_to_use.shape[0], -1)
        
        # Aggregate SHAP values by base feature
        agg_shap_values_combined = np.zeros((num_samples, num_features))
        for i, feature in enumerate(combined_feature_names):
            base_name = feature.split('_t-')[0]
            base_idx = feature_names.index(base_name)
            agg_shap_values_combined[:, base_idx] += combined_shap_values_to_use[:, i]
        
        # Aggregate input data
        X_agg_combined = X_combined_sample.reshape(num_samples, combined_time_steps, num_features).mean(axis=1)
        
        # Ensure dimensions match
        if agg_shap_values_combined.shape[1] != len(feature_names):
            print(f"Warning: Aggregated SHAP values dimension ({agg_shap_values_combined.shape[1]}) does not match feature names ({len(feature_names)}).")
            return
        
        # Compute base_values
        combined_model_wrapper = LSTMModelWrapper(combined_model, feature_names)
        combined_base_values = np.mean(combined_model_wrapper(X_combined_sample))
        
        # Create SHAP Explanation object for aggregated features
        combined_explanation_agg = shap.Explanation(
            values=agg_shap_values_combined,
            base_values=combined_base_values,
            data=X_agg_combined,
            feature_names=feature_names
        )
        
        # Generate aggregated waterfall plot
        try:
            plt.figure(figsize=(12, 8))
            shap.plots.waterfall(combined_explanation_agg[0], max_display=10, show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(combined_output_dir, "aggregated_waterfall_plot.png"))
            plt.close()
        except Exception as e:
            print(f"Error saving aggregated waterfall plot for combined model: {e}")
        
        # Generate aggregated beeswarm plot
        try:
            plt.figure(figsize=(12, 8))
            shap.plots.beeswarm(combined_explanation_agg, max_display=10, show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(combined_output_dir, "aggregated_beeswarm_plot.png"))
            plt.close()
        except Exception as e:
            print(f"Error saving aggregated beeswarm plot for combined model: {e}")
        
        # Generate aggregated bar plot
        try:
            plt.figure(figsize=(12, 8))
            shap.summary_plot(agg_shap_values_combined, X_agg_combined, feature_names=feature_names, 
                              plot_type="bar", show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(combined_output_dir, "aggregated_summary_bar_plot.png"))
            plt.close()
        except Exception as e:
            print(f"Error saving aggregated bar plot for combined model: {e}")
        
        # Analyze feature importance
        try:
            combined_agg_importance = analyze_feature_importance(
                combined_shap_values, combined_feature_names,
                combined_output_dir, time_steps=combined_time_steps
            )
        except Exception as e:
            print(f"Error analyzing feature importance for combined model: {e}")
            return
        
        # Compare models
        print("Comparing feature importance between models...")
        
        # Prepare comparison DataFrames
        finetune_df = finetune_agg_importance.set_index('Feature')
        finetune_df.columns = ['Finetune_Importance']
        
        combined_df = combined_agg_importance.set_index('Feature')
        combined_df.columns = ['Combined_Importance']
        
        # Join and normalize
        comparison_df = finetune_df.join(combined_df, how='outer').fillna(0).reset_index()
        finetune_sum = comparison_df['Finetune_Importance'].sum()
        combined_sum = comparison_df['Combined_Importance'].sum()
        
        comparison_df['Finetune_Pct'] = comparison_df['Finetune_Importance'] / finetune_sum * 100
        comparison_df['Combined_Pct'] = comparison_df['Combined_Importance'] / combined_sum * 100
        comparison_df['Pct_Difference'] = comparison_df['Finetune_Pct'] - comparison_df['Combined_Pct']
        
        # Sort by finetune percentage difference first, then save the original comparison
        try:
            comparison_df_orig = comparison_df.sort_values('Pct_Difference', key=abs, ascending=False)
            comparison_df_orig.to_csv(os.path.join(output_dir, "model_comparison.csv"), index=False)
        except Exception as e:
            print(f"Error saving model comparison CSV: {e}")
        
        # Plot comparison - sorted by finetune model importance (highest to lowest)
        try:
            plt.figure(figsize=(14, 8))
            
            # Sort by finetune model importance for plotting
            comparison_df_plot = comparison_df.sort_values('Finetune_Pct', ascending=False)
            
            top_n = min(10, len(comparison_df_plot))
            features = comparison_df_plot['Feature'][:top_n].tolist()
            finetune_vals = comparison_df_plot['Finetune_Pct'][:top_n].values
            combined_vals = comparison_df_plot['Combined_Pct'][:top_n].values
            
            x = np.arange(top_n)
            width = 0.35
            
            bars1 = plt.bar(x - width/2, finetune_vals, width, label='Finetune Model')
            bars2 = plt.bar(x + width/2, combined_vals, width, label='Combined Model')
            
            # Add value labels on top of bars
            for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
                # Finetune model values
                height1 = bar1.get_height()
                plt.text(bar1.get_x() + bar1.get_width()/2., height1 + 0.1,
                        f'{height1:.1f}%', ha='center', va='bottom', fontsize=8)
                
                # Combined model values
                height2 = bar2.get_height()
                plt.text(bar2.get_x() + bar2.get_width()/2., height2 + 0.1,
                        f'{height2:.1f}%', ha='center', va='bottom', fontsize=8)
            
            plt.xlabel('Features')
            plt.ylabel('Importance (%)')
            plt.title('Feature Importance Comparison (Ordered by Finetune Model)')
            plt.xticks(x, features, rotation=45, ha='right')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "model_importance_comparison.png"))
            plt.close()
        except Exception as e:
            print(f"Error saving model comparison plot: {e}")
    
    print("SHAP analysis complete! Results saved to:", output_dir)
    
    # Print top features
    print("\nTop 5 features for finetune model:")
    for i, row in finetune_agg_importance.head(5).iterrows():
        print(f"  {row['Feature']}: {row['Importance']:.6f}")
    
    if has_combined:
        print("\nTop 5 features for combined model:")
        for i, row in combined_agg_importance.head(5).iterrows():
            print(f"  {row['Feature']}: {row['Importance']:.6f}")
        
        print("\nFeatures with largest importance difference:")
        for i, row in comparison_df.head(5).iterrows():
            print(f"  {row['Feature']}: {row['Pct_Difference']:.2f}% (Finetune: {row['Finetune_Pct']:.2f}%, Combined: {row['Combined_Pct']:.2f}%)")

if __name__ == "__main__":
    main()