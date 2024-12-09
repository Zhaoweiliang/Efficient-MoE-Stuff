import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from LR_test import LoRaMoELinear
import matplotlib.pyplot as plt
import numpy as np
import time
import pandas as pd
import torch.nn.functional as F
import math

class LoRaMoELinear(nn.Module):
    def __init__(
        self, 
        in_features, 
        out_features, 
        num_experts=4, 
        top_k=1, 
        rank=8, 
        bias=False,
        alpha=1.0,
        dropout=0.1,
        capacity_factor=1.5,
        load_balance_coef=0.01
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_experts = num_experts
        self.top_k = top_k
        self.rank = rank
        self.capacity_factor = capacity_factor
        self.load_balance_coef = load_balance_coef
        self.alpha = alpha / self.rank
        
        # Main weight matrix
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        
        # LoRA parameters for each expert
        self.lora_A = nn.Parameter(torch.empty(num_experts, in_features, rank))
        self.lora_B = nn.Parameter(torch.empty(num_experts, rank, out_features))
        
        # Gating network
        self.gate = nn.Sequential(
            nn.Linear(in_features, 4 * num_experts),
            nn.LayerNorm(4 * num_experts),
            nn.ReLU(),
            nn.Linear(4 * num_experts, num_experts)
        )
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
        
        nn.init.normal_(self.lora_A, mean=0.0, std=1.0 / math.sqrt(self.rank))
        nn.init.zeros_(self.lora_B)
        
        for module in self.gate.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _compute_capacity(self, batch_size):
        return int(self.capacity_factor * batch_size / self.num_experts)

    def forward(self, x):
        orig_shape = x.shape
        if len(orig_shape) > 2:
            x = x.view(-1, self.in_features)
            
        batch_size = x.size(0)
        
        # Compute gates and routing
        gates = self.gate(x)
        routing_weights = F.softmax(gates, dim=-1)
        top_k_gates, top_k_indices = torch.topk(gates, self.top_k, dim=-1)
        top_k_gates = F.softmax(top_k_gates, dim=-1)
        
        # Compute load balancing loss
        fraction_to_experts = routing_weights.mean(dim=0)
        ideal_fraction = torch.ones_like(fraction_to_experts) / self.num_experts
        load_balancing_loss = torch.sum((fraction_to_experts - ideal_fraction) ** 2)
        self.load_balancing_loss = self.load_balance_coef * load_balancing_loss
        
        main_out = F.linear(x, self.weight, self.bias)
        expert_out = torch.zeros_like(main_out)
        
        capacity = self._compute_capacity(batch_size)
        
        for k in range(self.top_k):
            expert_idx = top_k_indices[:, k]
            expert_gate = top_k_gates[:, k].unsqueeze(-1)
            token_counts = torch.bincount(expert_idx, minlength=self.num_experts)
            mask = token_counts <= capacity

            for expert_id in range(self.num_experts):
                if not mask[expert_id]:
                    continue
                
                expert_mask = expert_idx == expert_id
                expert_inputs = x[expert_mask]
                expert_gates = expert_gate[expert_mask]
                
                # Apply capacity constraint
                if expert_inputs.size(0) > capacity:
                    expert_inputs = expert_inputs[:capacity]
                    expert_gates = expert_gates[:capacity]
                
                if expert_inputs.size(0) == 0:
                    continue
                
                expert_A = self.lora_A[expert_id]
                expert_B = self.lora_B[expert_id]
                
                tmp = F.linear(expert_inputs, expert_A.t())
                expert_output = F.linear(tmp, expert_B.t())
                
                expert_output = self.dropout(expert_output)
                expert_output = self.alpha * expert_output
                
                # Accumulate expert output
                expert_out[expert_mask.nonzero(as_tuple=True)[0][:capacity]] += expert_gates * expert_output
        
        output = main_out + expert_out
        
        if len(orig_shape) > 2:
            output = output.view(*orig_shape[:-1], self.out_features)
            
        return output

class ConvNet(nn.Module):
    def __init__(self, linear_layer_type='standard', num_experts=6, rank=4, hidden_dim=256,moe_hidden_dim=64):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.1)
        
        input_dim = 64 * 6 * 6  # 2304
        
        if linear_layer_type == 'standard':
            self.fc1 = nn.Linear(input_dim, hidden_dim)
        else:
    
            self.fc1 = LoRaMoELinear(
                input_dim, 
                moe_hidden_dim,  # 64
                num_experts=num_experts,
                rank=rank,
                dropout=0.1,
                capacity_factor=1.5,
                load_balance_coef=0.01,
                alpha=16.0  
            )
            
            self.fc1_match = LoRaMoELinear(
                moe_hidden_dim,  # 64
                hidden_dim,      # 256
                num_experts=num_experts,
                rank=rank,
                dropout=0.1,
                capacity_factor=1.5,
                load_balance_coef=0.01,
                alpha=16.0
            )
            
        self.fc2 = nn.Linear(hidden_dim, 10)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 6 * 6)
        x = self.dropout(x)
        
        if hasattr(self, 'fc1_match'):
            x = F.relu(self.fc1(x))      # fc1: 2304 -> 64
            x = self.dropout(x)
            x = F.relu(self.fc1_match(x)) # fc1_match: 64 -> 256
        else:
            x = F.relu(self.fc1(x))      # fc1: 2304 -> 256
            
        x = self.dropout(x)
        x = self.fc2(x)                  # fc2: 256 -> 10
        return x
    

def print_param_count(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    
    # Print parameters for each layer
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        print(f"{name}: {params:,} parameters")

# Test the parameter counts
standard_model = ConvNet('standard')
moe_model = ConvNet('lora_moe')

print("\nStandard Model:")
print_param_count(standard_model)
print("\nMoE+LoRA Model:")
print_param_count(moe_model)
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

import json
import os
from datetime import datetime

def save_results(results_df, model_type, num_experts=None, rank=None):
    """Save training results to a JSON file"""
    # Create a directory for results if it doesn't exist
    os.makedirs('training_results', exist_ok=True)
    
    # Convert DataFrame to dict for saving
    results_dict = results_df.to_dict('records')
    
    # Create filename based on configuration
    if model_type == 'standard':
        filename = 'standard_results.json'
    else:
        filename = f'moe_results_E{num_experts}_R{rank}.json'
    
    filepath = os.path.join('training_results', filename)
    
    # Save results with timestamp
    save_data = {
        'timestamp': datetime.now().isoformat(),
        'results': results_dict
    }
    
    with open(filepath, 'w') as f:
        json.dump(save_data, f)

def load_results(model_type, num_experts=None, rank=None):
    """Load training results if they exist"""
    if model_type == 'standard':
        filepath = os.path.join('training_results', 'standard_results.json')
    else:
        filepath = os.path.join('training_results', f'moe_results_E{num_experts}_R{rank}.json')
    
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            data = json.load(f)
            return pd.DataFrame(data['results'])
    return None

def train_and_evaluate(model_type='standard', num_experts=4, rank=8, epochs=12, force_retrain=False):
    # Try to load existing results for standard model
    if model_type == 'standard' and not force_retrain:
        existing_results = load_results(model_type)
        if existing_results is not None:
            print("Loading existing standard model results...")
            return existing_results
    
    # Original training code
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True)
    testloader = DataLoader(testset, batch_size=128)
    
    model = ConvNet(model_type, num_experts, rank).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    
    param_count = count_params(model)
    results = []
    
    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_acc = 100. * correct / total
        train_loss = running_loss / len(trainloader)
        
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        test_acc = 100. * correct / total
        epoch_time = time.time() - start_time
        
        results.append({
            'model_type': model_type,
            'num_experts': num_experts,
            'rank': rank,
            'epoch': epoch + 1,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'train_loss': train_loss,
            'time': epoch_time,
            'params': param_count
        })
        
        print(f'Epoch {epoch+1}: Train Acc={train_acc:.2f}%, Test Acc={test_acc:.2f}%, Time={epoch_time:.2f}s')
    
    results_df = pd.DataFrame(results)
    
    # Save results
    save_results(results_df, model_type, num_experts, rank)
    
    return results_df

def compare_expert_counts(expert_counts=[2, 4, 6, 8], rank=4, epochs=5):
    results_list = []
    
    # First get baseline results
    print("Getting baseline model results...")
    baseline_results = train_and_evaluate('standard', force_retrain=False)
    results_list.append(baseline_results)
    
    # Train models with different numbers of experts
    for num_experts in expert_counts:
        print(f'\nTraining LoRaMoE model with {num_experts} experts, rank {rank}...')
        lora_results = train_and_evaluate('lora_moe', num_experts=num_experts, rank=rank, epochs=epochs)
        results_list.append(lora_results)
    
    all_results = pd.concat(results_list, ignore_index=True)
    
    # Create comparison plots
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Parameter Count vs Number of Experts
    plt.subplot(131)
    expert_params = all_results[all_results['model_type'] == 'lora_moe'].groupby('num_experts')['params'].mean()
    baseline_params = all_results[all_results['model_type'] == 'standard']['params'].iloc[0]
    
    plt.plot(expert_params.index, expert_params.values, marker='o', label='LoRaMoE')
    plt.axhline(y=baseline_params, color='r', linestyle='--', label='Standard')
    plt.title('Parameter Count vs Experts')
    plt.xlabel('Number of Experts')
    plt.ylabel('Parameters')
    plt.legend()
    
    # Plot 2: Training Time vs Number of Experts
    plt.subplot(132)
    expert_times = all_results[all_results['model_type'] == 'lora_moe'].groupby('num_experts')['time'].mean()
    baseline_time = all_results[all_results['model_type'] == 'standard']['time'].mean()
    
    plt.plot(expert_times.index, expert_times.values, marker='o', label='LoRaMoE')
    plt.axhline(y=baseline_time, color='r', linestyle='--', label='Standard')
    plt.title('Training Time vs Experts')
    plt.xlabel('Number of Experts')
    plt.ylabel('Seconds per Epoch')
    plt.legend()
    
    # Plot 3: Final Test Accuracy vs Number of Experts
    plt.subplot(133)
    expert_acc = all_results[all_results['model_type'] == 'lora_moe'].groupby('num_experts')['test_acc'].last()
    baseline_acc = all_results[all_results['model_type'] == 'standard']['test_acc'].iloc[-1]
    
    plt.plot(expert_acc.index, expert_acc.values, marker='o', label='LoRaMoE')
    plt.axhline(y=baseline_acc, color='r', linestyle='--', label='Standard')
    plt.title('Final Test Accuracy vs Experts')
    plt.xlabel('Number of Experts')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("\nResults Summary:")
    summary = all_results.groupby(['model_type', 'num_experts']).agg({
        'params': 'first',
        'test_acc': ['last', 'mean'],
        'time': 'mean'
    }).round(4)
    print(summary)
    
    return all_results

if __name__ == '__main__':
    # Test with different numbers of experts
    results = compare_expert_counts(expert_counts=[2, 4, 6,8,10], rank=8, epochs=10)