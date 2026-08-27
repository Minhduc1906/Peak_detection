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
import time
from datetime import datetime

# Create directory structure - MODIFIED: Added 'time' folder
results_dir = 'results'
for subdir in ['models', 'plots', 'metrics', 'time']:
    os.makedirs(os.path.join(results_dir, subdir), exist_ok=True)

# Global list to store timing data
training_times = []

# Helper functions
def extract_scenario_name(filename):
    base_name = filename.replace('.csv', '').replace('test_', '')
    parts = base_name.split('_')
    scenario_parts = []
    for part in parts:
        if any(keyword in part for keyword in ['limited', 'unlimited', 'mbps', 'load', 'randomized']):
            break
        scenario_parts.append(part)
    return '_'.join(scenario_parts)

def create_sequences(X, y, time_steps=10):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

def save_timing_data():
    """Save timing data to CSV file"""
    if training_times:
        timing_df = pd.DataFrame(training_times)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(results_dir, 'time', f'training_times_{timestamp}.csv')
        timing_df.to_csv(csv_path, index=False)
        print(f"\nTiming data saved to: {csv_path}")
        
        # Also save a summary CSV with just the latest run
        summary_path = os.path.join(results_dir, 'time', 'latest_training_times.csv')
        timing_df.to_csv(summary_path, index=False)
        print(f"Latest timing summary saved to: {summary_path}")

# LSTM Model
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

# EWC class with fixed Fisher computation
class EWC:
    def __init__(self, model, dataloader, device):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.params = {n: p.clone() for n, p in model.named_parameters() if p.requires_grad}
        self._compute_fisher()
    
    def _compute_fisher(self):
        self.fisher = {}
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                self.fisher[n] = torch.zeros_like(p)
        
        criterion = nn.BCELoss()
        self.model.train()  # Keep in training mode for CuDNN
        
        for x_batch, y_batch in self.dataloader:
            x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
            self.model.zero_grad()
            
            # Forward pass without gradients
            with torch.no_grad():
                outputs = self.model(x_batch)
            
            # Enable gradients for loss computation
            outputs = outputs.clone().requires_grad_(True)
            loss = criterion(outputs, y_batch)
            
            # Compute gradients
            loss.backward()
            
            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    self.fisher[n] += p.grad.data.pow(2) / len(self.dataloader.dataset)
    
    def penalty(self, model, lambda_ewc=1000.0):
        loss = 0
        for n, p in model.named_parameters():
            if n in self.params:
                loss += (self.fisher[n] * (p - self.params[n]).pow(2)).sum()
        return lambda_ewc * loss

def preprocess_data(df, include_filename_features=False, include_direction=False, 
                   include_network_limit=True, time_steps=10, scaler=None):
    data_copy = df.copy()
    data_copy.replace([np.inf, -np.inf], np.nan, inplace=True)
    data_copy.fillna(data_copy.max(), inplace=True)
    
    drop_cols = ['queue_size']
    if not include_filename_features:
        drop_cols.extend(['file_name', 'file_name_encoded'])
    else:
        drop_cols.append('file_name')
    
    if not include_direction and 'direction' in data_copy.columns:
        drop_cols.append('direction')
    
    if not include_network_limit and 'network_limit' in data_copy.columns:
        drop_cols.append('network_limit')
        
    for col in drop_cols:
        if col in data_copy.columns:
            data_copy = data_copy.drop(columns=[col])
            
    if include_direction and 'direction' in df.columns:
        enc_args = {'sparse_output': False} if int(sklearn.__version__.split('.')[0]) >= 1 else {'sparse': False}
        one_hot = OneHotEncoder(**enc_args).fit_transform(data_copy[['direction']])
        dir_cols = [f'direction_{i}' for i in range(one_hot.shape[1])]
        data_copy = data_copy.drop(columns=['direction'])
        data_copy = pd.concat([data_copy, pd.DataFrame(one_hot, columns=dir_cols, index=data_copy.index)], axis=1)
    
    if include_network_limit and 'network_limit' in data_copy.columns:
        limits = data_copy['network_limit'].unique()
        if len(limits) > 1:
            max_limit = data_copy['network_limit'].max()
            min_limit = data_copy['network_limit'].min()
            if max_limit > min_limit:
                data_copy['network_limit'] = (data_copy['network_limit'] - min_limit) / (max_limit - min_limit)
    
    for col in ['subfolder', 'testcase_with_limit']:
        if col in data_copy.columns:
            data_copy = data_copy.drop(columns=[col])
    
    feature_cols = [c for c in data_copy.columns if c != 'queue_exists' and c != 'detection_accuracy' and c != "time" and 
                    c != "relative_time" and c != 'network_limit']

    X = data_copy[feature_cols].values
    y = data_copy['queue_exists'].values
    
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)
    
    X_seq, y_seq = create_sequences(X_scaled, y, time_steps)
    X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42, shuffle=False)
    
    return X_train, X_test, y_train, y_test, feature_cols, scaler

def train_and_evaluate_lstm(X_train, X_test, y_train, y_test, model_name, epochs=20, batch_size=32,
                          feature_names=None, model=None, optimizer=None, ewc=None,
                          finetune=False, lambda_ewc=1000.0):
    
    # MODIFIED: Track training start time
    start_time = time.time()
    start_datetime = datetime.now()
    
    print(f"\nStarting training for {model_name} at {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    
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
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_size = X_train.shape[2]
    
    if model is None:
        model = LSTMModel(input_size=input_size).to(device)
    
    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    criterion = nn.BCELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                    factor=0.5, patience=3)
    
    train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []
    
    for epoch in range(epochs):
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            if finetune and ewc is not None:
                ewc_loss = ewc.penalty(model, lambda_ewc)
                loss += ewc_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            predicted = (outputs >= 0.5).float()
            train_total += y_batch.size(0)
            train_correct += (predicted == y_batch).sum().item()
        
        train_loss_avg = train_loss / len(train_loader)
        train_acc = train_correct / train_total
        train_losses.append(train_loss_avg)
        train_accuracies.append(train_acc)
        
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
        
        scheduler.step(val_loss_avg)
        
        print(f'{model_name} - Epoch {epoch+1}/{epochs}, Train: {train_acc:.4f}, Val: {val_acc:.4f}')
    
    # MODIFIED: Track training end time and calculate duration
    end_time = time.time()
    end_datetime = datetime.now()
    duration_seconds = end_time - start_time
    duration_minutes = duration_seconds / 60.0
    
    print(f"Completed training for {model_name} at {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Training duration: {duration_minutes:.2f} minutes ({duration_seconds:.2f} seconds)")
    
    # MODIFIED: Store timing data
    timing_data = {
        'model_name': model_name,
        'start_time': start_datetime.strftime('%Y-%m-%d %H:%M:%S'),
        'end_time': end_datetime.strftime('%Y-%m-%d %H:%M:%S'),
        'duration_seconds': round(duration_seconds, 2),
        'duration_minutes': round(duration_minutes, 2),
        'epochs': epochs,
        'batch_size': batch_size,
        'training_samples': len(X_train),
        'validation_samples': len(X_test),
        'final_train_accuracy': round(train_accuracies[-1], 4),
        'final_val_accuracy': round(val_accuracies[-1], 4),
        'is_finetuned': finetune,
        'lambda_ewc': lambda_ewc if finetune else None,
        'device': str(device)
    }
    training_times.append(timing_data)
    
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
        'is_finetuned': finetune,
        'training_duration_seconds': duration_seconds,
        'training_duration_minutes': duration_minutes
    }
    
    # Always save models to models folder and metrics to metrics folder
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'time_steps': X_train.shape[1],
        'training_time': timing_data
    }, os.path.join(results_dir, 'models', f"{model_name}.pt"))
    
    # Save metrics JSON to metrics folder
    with open(os.path.join(results_dir, 'metrics', f"{model_name}_metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=4)
    
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
    
    return model, optimizer, train_loader, metrics

def train_sequentially_finetuned_model(test_cases, data, time_steps=10, epochs=20, batch_size=32, lambda_ewc=1000.0):
    sorted_cases = sorted(test_cases)
    finetune_metrics = []
    model, optimizer, prev_loader, scaler = None, None, None, None
    
    # MODIFIED: Track overall fine-tuning start time
    overall_start_time = time.time()
    overall_start_datetime = datetime.now()
    print(f"\nStarting sequential fine-tuning process at {overall_start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    
    for i, testcase_with_limit in enumerate(sorted_cases):
        print(f"\n\n{'='*50}\nFine-tuning Step {i+1}/{len(sorted_cases)}: {testcase_with_limit}\n{'='*50}")
        
        filename, network_limit = testcase_with_limit.rsplit('_', 1)
        case_data = data[
            (data['file_name'] == filename) & 
            (data['network_limit'] == int(network_limit))
        ]
        
        if len(case_data) <= time_steps + 10:
            print(f"Skipping {testcase_with_limit} - insufficient data ({len(case_data)} rows)")
            continue
        
        scenario = extract_scenario_name(filename)
        step_name = f"finetune_step{i+1}_{scenario}_limit{network_limit}"
        
        X_train, X_test, y_train, y_test, feature_names, scaler = preprocess_data(
            case_data, include_network_limit=True, time_steps=time_steps, scaler=scaler
        )
        
        ewc = None
        if i > 0 and prev_loader is not None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            ewc = EWC(model, prev_loader, device)
        
        model, optimizer, train_loader, metrics = train_and_evaluate_lstm(
            X_train, X_test, y_train, y_test,
            model_name=step_name,
            epochs=epochs,
            batch_size=batch_size,
            feature_names=feature_names,
            model=model,
            optimizer=optimizer,
            ewc=ewc,
            finetune=(i > 0),
            lambda_ewc=lambda_ewc
        )
        
        prev_loader = train_loader
        
        metrics.update({
            'original_filename': filename,
            'network_limit': network_limit,
            'testcase_with_limit': testcase_with_limit,
            'finetune_step': i + 1
        })
        finetune_metrics.append(metrics)
        
        # Changed to save checkpoint models to models folder
        if i == 0 or i == len(sorted_cases) - 1:
            checkpoint_name = "finetune_baseline_model.pt" if i == 0 else "finetune_final_model.pt"
            torch.save({
                'model_state_dict': copy.deepcopy(model).state_dict(),
                'step': i + 1,
                'testcase_with_limit': testcase_with_limit
            }, os.path.join(results_dir, 'models', checkpoint_name))
    
    # MODIFIED: Track overall fine-tuning end time
    overall_end_time = time.time()
    overall_end_datetime = datetime.now()
    overall_duration_seconds = overall_end_time - overall_start_time
    overall_duration_minutes = overall_duration_seconds / 60.0
    
    print(f"\nCompleted sequential fine-tuning process at {overall_end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total fine-tuning duration: {overall_duration_minutes:.2f} minutes ({overall_duration_seconds:.2f} seconds)")
    
    # MODIFIED: Store overall fine-tuning timing data
    overall_timing = {
        'model_name': 'sequential_finetuning_overall',
        'start_time': overall_start_datetime.strftime('%Y-%m-%d %H:%M:%S'),
        'end_time': overall_end_datetime.strftime('%Y-%m-%d %H:%M:%S'),
        'duration_seconds': round(overall_duration_seconds, 2),
        'duration_minutes': round(overall_duration_minutes, 2),
        'epochs': epochs,
        'batch_size': batch_size,
        'total_steps': len(sorted_cases),
        'successful_steps': len(finetune_metrics),
        'lambda_ewc': lambda_ewc,
        'device': str(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    }
    training_times.append(overall_timing)
    
    if finetune_metrics:
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
        plt.savefig(os.path.join(results_dir, 'plots', "finetune_performance.png"))
        plt.close()
        
        step_mapping = {
            i+1: {
                'filename': case.rsplit('_', 1)[0],
                'network_limit': case.rsplit('_', 1)[1],
                'scenario': extract_scenario_name(case.rsplit('_', 1)[0])
            } for i, case in enumerate(sorted_cases)
        }
        
        # Changed to save step mapping to metrics folder
        with open(os.path.join(results_dir, 'metrics', "finetune_steps_mapping.json"), 'w') as f:
            json.dump(step_mapping, f, indent=4)
    
    return model, finetune_metrics

def main():
    # MODIFIED: Track overall execution start time
    main_start_time = time.time()
    main_start_datetime = datetime.now()
    print(f"Starting main execution at {main_start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    
    time_steps = 10
    batch_size = 32
    epochs = 20
    all_metrics = []
    lambda_ewc = 1000.0
    
    print("Loading data...")
    data = pd.read_csv('combined_features.csv')
    print(f"Loaded {len(data)} records")
    
    if 'network_limit' not in data.columns and 'subfolder' in data.columns:
        data['network_limit'] = data['subfolder'].apply(
            lambda x: int(re.search(r'(\d+)$', x).group(1)) if re.search(r'(\d+)$', x) else -1
        )
    elif 'network_limit' not in data.columns:
        data['network_limit'] = -1
    
    downstream_data = data[data['direction'] == 'downstream'].copy()
    downstream_data['testcase_with_limit'] = downstream_data.apply(
        lambda row: f"{row['file_name']}_{row['network_limit']}", axis=1
    )
    
    unique_testcases = downstream_data['testcase_with_limit'].unique()
    print(f"Found {len(unique_testcases)} unique test case + network limit combinations")
    
    filename_encoder = LabelEncoder()
    data['file_name_encoded'] = filename_encoder.fit_transform(data['file_name'])
    downstream_data['file_name_encoded'] = filename_encoder.transform(downstream_data['file_name'])
    
    print("\nTraining sequentially fine-tuned model with EWC...")
    finetuned_model, finetune_metrics = train_sequentially_finetuned_model(
        unique_testcases,
        downstream_data,
        time_steps,
        epochs,
        batch_size,
        lambda_ewc=lambda_ewc
    )
    
    print("\nTraining combined models for comparison...")
    X_train, X_test, y_train, y_test, features, _ = preprocess_data(
        downstream_data, include_filename_features=False, include_direction=False,
        include_network_limit=True, time_steps=time_steps
    )
    
    _, _, _, downstream_metrics = train_and_evaluate_lstm(
        X_train, X_test, y_train, y_test,
        model_name="combined_model_downstream_only",
        epochs=epochs,
        batch_size=batch_size,
        feature_names=features
    )
    all_metrics.append(downstream_metrics)
    
    # Changed to save models summary to metrics folder
    with open(os.path.join(results_dir, 'metrics', 'models_summary.json'), 'w') as f:
        json.dump(all_metrics, f, indent=4)
    
    if finetune_metrics:
        final_finetune = finetune_metrics[-1]
        individual_accs = [m['val_accuracy'] for m in all_metrics if m['model_name'].startswith('downstream_')]
        avg_individual = sum(individual_accs) / max(len(individual_accs), 1)
        
        plt.figure(figsize=(10, 6))
        model_types = ['Individual\nModels Avg', 'Sequential\nFine-tuning', 'Combined\nDownstream']
        accuracies = [
            avg_individual,
            final_finetune['val_accuracy'],
            downstream_metrics['val_accuracy']
        ]
        
        plt.bar(model_types, accuracies, color=['lightblue', 'lightgreen', 'salmon'])
        plt.title('Model Performance Comparison')
        plt.ylabel('Validation Accuracy')
        plt.ylim(0, 1.0)
        
        for i, v in enumerate(accuracies):
            plt.text(i, v + 0.02, f"{v:.3f}", ha='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "model_comparison.png"))
        plt.close()
    
    # MODIFIED: Track overall execution end time and save timing data
    main_end_time = time.time()
    main_end_datetime = datetime.now()
    main_duration_seconds = main_end_time - main_start_time
    main_duration_minutes = main_duration_seconds / 60.0
    
    print(f"\nCompleted main execution at {main_end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total execution duration: {main_duration_minutes:.2f} minutes ({main_duration_seconds:.2f} seconds)")
    
    # Add overall execution timing
    overall_execution_timing = {
        'model_name': 'overall_execution',
        'start_time': main_start_datetime.strftime('%Y-%m-%d %H:%M:%S'),
        'end_time': main_end_datetime.strftime('%Y-%m-%d %H:%M:%S'),
        'duration_seconds': round(main_duration_seconds, 2),
        'duration_minutes': round(main_duration_minutes, 2),
        'total_models_trained': len(training_times),
        'device': str(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    }
    training_times.append(overall_execution_timing)
    
    # MODIFIED: Save all timing data to CSV
    save_timing_data()
    
    print("\nTraining complete. Results saved to results directory")
    print(f"Training time data saved to results/time/")

if __name__ == "__main__":
    main()