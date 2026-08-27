import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
import sklearn
import matplotlib.pyplot as plt
import os
import json
import re
import copy

# Create directory structure
results_dir = 'results'
for subdir in ['models', 'plots', 'metrics', 'finetune']:
    os.makedirs(os.path.join(results_dir, subdir), exist_ok=True)

# Helper functions
def extract_scenario_name(filename):
    """Extract scenario name from a complex filename"""
    base_name = filename.replace('.csv', '').replace('test_', '')
    parts = base_name.split('_')
    scenario_parts = []
    for part in parts:
        if any(keyword in part for keyword in ['limited', 'unlimited', 'mbps', 'load', 'randomized']):
            break
        scenario_parts.append(part)
    return '_'.join(scenario_parts)

def create_sequences(X, y, time_steps=10):
    """Create sequences for LSTM input"""
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

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
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        return self.sigmoid(self.fc2(out))

# Preprocess data function
def preprocess_data(df, include_filename_features=False, include_direction=False, 
                   include_network_limit=True, time_steps=10, scaler=None):
    """Preprocess data for LSTM model"""
    data_copy = df.copy()
    
    # Handle infinities and NaNs
    data_copy.replace([np.inf, -np.inf], np.nan, inplace=True)
    data_copy.fillna(data_copy.max(), inplace=True)
    
    # Drop unnecessary columns
    drop_cols = ['queue_size']
    if not include_filename_features:
        drop_cols.extend(['file_name', 'file_name_encoded'])
    else:
        drop_cols.append('file_name')
    
    if not include_direction and 'direction' in data_copy.columns:
        drop_cols.append('direction')
    
    if not include_network_limit and 'network_limit' in data_copy.columns:
        drop_cols.append('network_limit')
        
    # Drop columns if they exist
    for col in drop_cols:
        if col in data_copy.columns:
            data_copy = data_copy.drop(columns=[col])
            
    # One-hot encode direction if needed
    if include_direction and 'direction' in df.columns:
        enc_args = {'sparse_output': False} if int(sklearn.__version__.split('.')[0]) >= 1 else {'sparse': False}
        one_hot = OneHotEncoder(**enc_args).fit_transform(data_copy[['direction']])
        dir_cols = [f'direction_{i}' for i in range(one_hot.shape[1])]
        data_copy = data_copy.drop(columns=['direction'])
        data_copy = pd.concat([data_copy, pd.DataFrame(one_hot, columns=dir_cols, index=data_copy.index)], axis=1)
    
    # Normalize network_limit if included
    if include_network_limit and 'network_limit' in data_copy.columns:
        limits = data_copy['network_limit'].unique()
        if len(limits) > 1:
            max_limit = data_copy['network_limit'].max()
            min_limit = data_copy['network_limit'].min()
            if max_limit > min_limit:
                data_copy['network_limit'] = (data_copy['network_limit'] - min_limit) / (max_limit - min_limit)
    
    # Drop auxiliary columns
    for col in ['subfolder', 'testcase_with_limit']:
        if col in data_copy.columns:
            data_copy = data_copy.drop(columns=[col])
    
    # Extract features and target
    feature_cols = [c for c in data_copy.columns if c != 'queue_exists']
    X = data_copy[feature_cols].values
    y = data_copy['queue_exists'].values
    
    # Scale features
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)
    
    # Create sequences and split data
    X_seq, y_seq = create_sequences(X_scaled, y, time_steps)
    X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, feature_cols, scaler

# Training function
def train_and_evaluate_lstm(X_train, X_test, y_train, y_test, model_name, epochs=20, batch_size=32,
                          feature_names=None, existing_model=None, optimizer_state=None, finetune=False):
    """Train or fine-tune LSTM model"""
    # Convert to tensors and create data loaders
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)
    
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor), 
        batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor), 
        batch_size=batch_size, shuffle=False
    )
    
    # Determine device and initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_size = X_train.shape[2]
    
    if existing_model is None:
        model = LSTMModel(input_size=input_size).to(device)
    else:
        model = existing_model
    
    # Initialize optimizer
    criterion = nn.BCELoss()
    if optimizer_state is None:
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    else:
        optimizer = optim.Adam(model.parameters(), lr=0.0005)  # Reduced rate for fine-tuning
        optimizer.load_state_dict(optimizer_state)
    
    # Initialize metrics tracking
    train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []
    
    # Training loop
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            predicted = (outputs >= 0.5).float()
            train_total += y_batch.size(0)
            train_correct += (predicted == y_batch).sum().item()
        
        train_loss_avg = train_loss / len(train_loader)
        train_acc = train_correct / train_total
        train_losses.append(train_loss_avg)
        train_accuracies.append(train_acc)
        
        # Validation phase
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                
                val_loss += loss.item()
                predicted = (outputs >= 0.5).float()
                val_total += y_batch.size(0)
                val_correct += (predicted == y_batch).sum().item()
        
        val_loss_avg = val_loss / len(test_loader)
        val_acc = val_correct / val_total
        val_losses.append(val_loss_avg)
        val_accuracies.append(val_acc)
        
        print(f'{model_name} - Epoch {epoch+1}/{epochs}, Train: {train_acc:.4f}, Val: {val_acc:.4f}')
    
    # Prepare metrics
    display_name = model_name.replace("downstream_", "").replace("finetune_", "FT: ")
    metrics = {
        'model_name': model_name,
        'display_name': display_name,
        'train_accuracy': train_accuracies[-1],
        'val_accuracy': val_accuracies[-1],
        'train_loss': train_losses[-1],
        'val_loss': val_losses[-1],
        'feature_names': feature_names,
        'train_history': {'loss': train_losses, 'accuracy': train_accuracies},
        'val_history': {'loss': val_losses, 'accuracy': val_accuracies},
        'is_finetuned': finetune
    }
    
    # Save model and metrics
    save_dir = os.path.join(results_dir, 'finetune' if finetune else 'models')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'time_steps': X_train.shape[1]
    }, os.path.join(save_dir, f"{model_name}.pt"))
    
    # Save metrics separately
    with open(os.path.join(save_dir, f"{model_name}_metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Create plot
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_accuracies, 'b-', label='Train')
    plt.plot(val_accuracies, 'r-', label='Val')
    plt.title(f'{display_name} - Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_losses, 'b-', label='Train')
    plt.plot(val_losses, 'r-', label='Val')
    plt.title(f'{display_name} - Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'plots', f"{model_name}_plot.png"))
    plt.close()
    
    return model, optimizer.state_dict(), metrics

# Function for sequential fine-tuning
def train_sequentially_finetuned_model(test_cases, data, time_steps=10, epochs=20, batch_size=32):
    """Train a model sequentially with fine-tuning"""
    # Sort test cases for deterministic order
    sorted_cases = sorted(test_cases)
    finetune_metrics = []
    model, optimizer_state, scaler = None, None, None
    
    # Train on each test case sequentially
    for i, testcase_with_limit in enumerate(sorted_cases):
        print(f"\n\n{'='*50}\nFine-tuning Step {i+1}/{len(sorted_cases)}: {testcase_with_limit}\n{'='*50}")
        
        filename, network_limit = testcase_with_limit.rsplit('_', 1)
        
        # Filter data for this test case and network limit
        case_data = data[
            (data['file_name'] == filename) & 
            (data['network_limit'] == int(network_limit))
        ]
        
        # Skip if not enough data
        if len(case_data) <= time_steps + 10:
            print(f"Skipping {testcase_with_limit} - insufficient data ({len(case_data)} rows)")
            continue
        
        # Preprocess data for this test case
        scenario = extract_scenario_name(filename)
        step_name = f"finetune_step{i+1}_{scenario}_limit{network_limit}"
        
        X_train, X_test, y_train, y_test, feature_names, scaler = preprocess_data(
            case_data, include_network_limit=True, time_steps=time_steps, scaler=scaler
        )
        
        # Train or fine-tune the model
        model, optimizer_state, metrics = train_and_evaluate_lstm(
            X_train, X_test, y_train, y_test, model_name=step_name,
            epochs=epochs, batch_size=batch_size, feature_names=feature_names,
            existing_model=model, optimizer_state=optimizer_state, finetune=(i > 0)
        )
        
        # Add metadata to metrics
        metrics.update({
            'original_filename': filename,
            'network_limit': network_limit,
            'testcase_with_limit': testcase_with_limit,
            'finetune_step': i + 1
        })
        finetune_metrics.append(metrics)
        
        # Save checkpoints
        if i == 0 or i == len(sorted_cases) - 1:
            checkpoint_name = "finetune_baseline_model.pt" if i == 0 else "finetune_final_model.pt"
            torch.save({
                'model_state_dict': copy.deepcopy(model).state_dict(),
                'step': i + 1,
                'testcase_with_limit': testcase_with_limit
            }, os.path.join(results_dir, 'finetune', checkpoint_name))
    
    # Create performance plots
    if finetune_metrics:
        # Plot accuracy and loss across fine-tuning steps
        plt.figure(figsize=(12, 5))
        steps = [m['finetune_step'] for m in finetune_metrics]
        
        plt.subplot(1, 2, 1)
        plt.plot(steps, [m['val_accuracy'] for m in finetune_metrics], 'o-')
        plt.title('Validation Accuracy Across Fine-tuning Steps')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(steps, [m['val_loss'] for m in finetune_metrics], 'o-')
        plt.title('Validation Loss Across Fine-tuning Steps')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'finetune', "finetune_performance.png"))
        plt.close()
        
        # Save step mapping
        step_mapping = {
            i+1: {
                'filename': case.rsplit('_', 1)[0],
                'network_limit': case.rsplit('_', 1)[1],
                'scenario': extract_scenario_name(case.rsplit('_', 1)[0])
            } for i, case in enumerate(sorted_cases)
        }
        
        with open(os.path.join(results_dir, 'finetune', "finetune_steps_mapping.json"), 'w') as f:
            json.dump(step_mapping, f, indent=4)
    
    return model, finetune_metrics

# Main execution function
def main():
    time_steps = 10
    batch_size = 32
    epochs = 20
    all_metrics = []
    
    # Load and process data
    print("Loading data...")
    data = pd.read_csv('combined_features.csv')
    print(f"Loaded {len(data)} records")
    
    # Check/create network_limit column
    if 'network_limit' not in data.columns and 'subfolder' in data.columns:
        data['network_limit'] = data['subfolder'].apply(
            lambda x: int(re.search(r'(\d+)$', x).group(1)) if re.search(r'(\d+)$', x) else -1
        )
    elif 'network_limit' not in data.columns:
        data['network_limit'] = -1
    
    # Filter downstream data and create identifiers
    downstream_data = data[data['direction'] == 'downstream'].copy()
    downstream_data['testcase_with_limit'] = downstream_data.apply(
        lambda row: f"{row['file_name']}_{row['network_limit']}", axis=1
    )
    
    # Get unique combinations
    unique_testcases = downstream_data['testcase_with_limit'].unique()
    print(f"Found {len(unique_testcases)} unique test case + network limit combinations")
    
    # Encode filenames for combined models
    filename_encoder = LabelEncoder()
    data['file_name_encoded'] = filename_encoder.fit_transform(data['file_name'])
    downstream_data['file_name_encoded'] = filename_encoder.transform(downstream_data['file_name'])
    
    # 1. Train individual models
    print("\nTraining individual models...")
    for testcase in unique_testcases:
        filename, network_limit = testcase.rsplit('_', 1)
        case_data = downstream_data[
            (downstream_data['file_name'] == filename) & 
            (downstream_data['network_limit'] == int(network_limit))
        ]
        
        if len(case_data) <= time_steps + 10:
            continue
            
        scenario = extract_scenario_name(filename)
        model_name = f"downstream_{scenario}_limit{network_limit}"
        
        X_train, X_test, y_train, y_test, features, _ = preprocess_data(
            case_data, include_network_limit=True, time_steps=time_steps
        )
        
        _, _, metrics = train_and_evaluate_lstm(
            X_train, X_test, y_train, y_test, model_name=model_name,
            epochs=epochs, batch_size=batch_size, feature_names=features
        )
        
        metrics.update({
            'original_filename': filename,
            'network_limit': network_limit,
            'testcase_with_limit': testcase
        })
        all_metrics.append(metrics)
    
    # 2. Train sequentially fine-tuned model
    print("\nTraining sequentially fine-tuned model...")
    finetuned_model, finetune_metrics = train_sequentially_finetuned_model(
        unique_testcases, downstream_data, time_steps, epochs, batch_size
    )
    
    # 3. Train combined models for comparison
    print("\nTraining combined models for comparison...")
    
    # Combined model with all data
    X_train, X_test, y_train, y_test, features, _ = preprocess_data(
        data, include_filename_features=True, include_direction=True, 
        include_network_limit=True, time_steps=time_steps
    )
    
    _, _, combined_metrics = train_and_evaluate_lstm(
        X_train, X_test, y_train, y_test, model_name="combined_model_all_directions",
        epochs=epochs, batch_size=batch_size, feature_names=features
    )
    all_metrics.append(combined_metrics)
    
    # Combined model with downstream data only
    X_train, X_test, y_train, y_test, features, _ = preprocess_data(
        downstream_data, include_filename_features=True, include_direction=False,
        include_network_limit=True, time_steps=time_steps
    )
    
    _, _, downstream_metrics = train_and_evaluate_lstm(
        X_train, X_test, y_train, y_test, model_name="combined_model_downstream_only",
        epochs=epochs, batch_size=batch_size, feature_names=features
    )
    all_metrics.append(downstream_metrics)
    
    # Save summary
    with open(os.path.join(results_dir, 'models_summary.json'), 'w') as f:
        json.dump(all_metrics, f, indent=4)
    
    # Compare model performance
    if finetune_metrics:
        final_finetune = finetune_metrics[-1]
        
        # Calculate average individual model accuracy
        individual_accs = [m['val_accuracy'] for m in all_metrics if m['model_name'].startswith('downstream_')]
        avg_individual = sum(individual_accs) / max(len(individual_accs), 1)
        
        # Plot comparison
        plt.figure(figsize=(10, 6))
        model_types = ['Individual\nModels Avg', 'Sequential\nFine-tuning', 'Combined\nAll', 'Combined\nDownstream']
        accuracies = [
            avg_individual,
            final_finetune['val_accuracy'],
            combined_metrics['val_accuracy'],
            downstream_metrics['val_accuracy']
        ]
        
        plt.bar(model_types, accuracies, color=['lightblue', 'lightgreen', 'salmon', 'orange'])
        plt.title('Model Performance Comparison')
        plt.ylabel('Validation Accuracy')
        plt.ylim(0, 1.0)
        
        for i, v in enumerate(accuracies):
            plt.text(i, v + 0.02, f"{v:.3f}", ha='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "model_comparison.png"))
        plt.close()
    
    print("\nTraining complete. Results saved to results/ directory")

if __name__ == "__main__":
    main()