import math
import os
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. THE PRUNABLE LINEAR LAYER
# ==========================================
class PrunableLinear(nn.Module):
    """
    A custom PyTorch linear layer that learns to prune its own weights
    during training by using a learnable gate for every weight parameter.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super(PrunableLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Standard weight parameter
        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        
        # Gate scores parameter for the custom gating mechanism
        self.gate_scores = nn.Parameter(torch.empty((out_features, in_features)))
        
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
            
        self.temperature = 1.0
            
        self.reset_parameters()
        
    def reset_parameters(self) -> None:
        # Standard Kaiming uniform initialization for linear weights
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # Initialize gate scores to a large positive value (3.0) so gates start near 1
        nn.init.constant_(self.gate_scores, 3.0)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply sigmoid with temperature scaling
        gates = torch.sigmoid(self.gate_scores / self.temperature)
        # Multiply weights by their corresponding gate score
        pruned_weights = self.weight * gates
        # Perform standard linear operation
        return F.linear(x, pruned_weights, self.bias)

# ==========================================
# 2. THE NEURAL NETWORK ARCHITECTURE
# ==========================================
class SelfPruningNet(nn.Module):
    def __init__(self, input_dim=3072, num_classes=10):
        super(SelfPruningNet, self).__init__()
        self.flatten = nn.Flatten()
        
        self.fc1 = PrunableLinear(input_dim, 512)
        self.relu1 = nn.ReLU()
        self.fc2 = PrunableLinear(512, 256)
        self.relu2 = nn.ReLU()
        self.fc3 = PrunableLinear(256, 128)
        self.relu3 = nn.ReLU()
        self.fc4 = PrunableLinear(128, num_classes)
        
    def forward(self, x):
        x = self.flatten(x)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        x = self.fc4(x)
        return x
        
    def set_temperature(self, t: float):
        """Sets the temperature value across all PrunableLinear layers dynamically."""
        for module in self.modules():
            if isinstance(module, PrunableLinear):
                module.temperature = t
        
    def get_all_gates(self):
        """Returns a single 1D tensor containing all current gate values (after sigmoid)."""
        gates = []
        for module in self.modules():
            if isinstance(module, PrunableLinear):
                gate_vals = torch.sigmoid(module.gate_scores / module.temperature)
                gates.append(gate_vals.view(-1))
        if len(gates) > 0:
            return torch.cat(gates)
        return torch.tensor([])

    def get_sparsity(self, threshold=0.1):
        """Calculates the overall network sparsity level."""
        all_gates = self.get_all_gates()
        if len(all_gates) == 0:
            return 0.0
        pruned_count = (all_gates < threshold).sum().item()
        total_count = all_gates.numel()
        return (pruned_count / total_count) * 100.0

    def get_layer_sparsity(self, threshold=0.1):
        """Returns the sparsity breakdown per PrunableLinear layer."""
        layer_stats = {}
        layer_idx = 1
        for name, module in self.named_modules():
            if isinstance(module, PrunableLinear):
                gates = torch.sigmoid(module.gate_scores / module.temperature)
                pruned_count = (gates < threshold).sum().item()
                total_count = gates.numel()
                layer_stats[f"Layer_{layer_idx}"] = (pruned_count / total_count) * 100.0
                layer_idx += 1
        return layer_stats

# ==========================================
# 3. TRAINING LOOP
# ==========================================
def train_model(lambda_val: float, epochs: int = 10, batch_size: int = 128, device: str = 'cuda'):
    print(f"\n[{'='*40}]\nTraining with lambda = {lambda_val}\n[{'='*40}]")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    # Using num_workers=0 to ensure compatibility across all systems (e.g. Windows)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = SelfPruningNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    current_temp = 1.0
    
    for epoch in range(epochs):
        model.set_temperature(current_temp)
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Classification Loss
            classification_loss = criterion(outputs, labels)
            
            # Custom Sparsity Loss (L1 on gates)
            all_gates = model.get_all_gates()
            sparsity_loss = torch.sum(all_gates)
            
            # Total Loss Formulation
            loss = classification_loss + lambda_val * sparsity_loss
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        scheduler.step()
        
        # Test evaluation at epoch end
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for data in testloader:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
        
        train_acc = 100 * correct / total
        test_acc = 100 * test_correct / test_total
        sparsity = model.get_sparsity()
        mean_gate = model.get_all_gates().mean().item()
        min_gate = model.get_all_gates().min().item()
        print(f"Epoch {epoch+1:02d}/{epochs} | Loss: {running_loss/len(trainloader):.4f} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}% | Sparsity: {sparsity:.2f}% | Mean Gate: {mean_gate:.4f} | Min Gate: {min_gate:.4f} | Temp: {current_temp:.2f}")

        # Slowly reduce temperature per epoch to allow gradients to focus sparsity aggressively over time
        current_temp = max(0.5, current_temp * 0.95)

    model_path = f"single_model_lambda_{lambda_val}.pth"
    torch.save(model.state_dict(), model_path)
    return test_acc, sparsity, model_path

# ==========================================
# 4. EVALUATION & PLOTTING
# ==========================================
def evaluate_and_plot(results, best_model_path, best_lambda):
    print("\n--- Generating Plots ---")
    # Trade-off Plot
    fig, ax1 = plt.subplots(figsize=(10, 6))
    color = 'tab:red'
    ax1.set_xlabel('Lambda value (log scale)')
    ax1.set_xscale('log')
    ax1.set_ylabel('Test Accuracy (%)', color=color)
    lambdas = [r['Lambda'] for r in results]
    accs = [r['Accuracy'] for r in results]
    ax1.plot(lambdas, accs, marker='o', color=color, linewidth=2, label='Accuracy')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(50, 65)   # zoomed in — actual range is 56–57%

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Sparsity Level (%)', color=color)
    sparsities = [r['Sparsity'] for r in results]
    ax2.plot(lambdas, sparsities, marker='s', color=color, linewidth=2, label='Sparsity')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(85, 100)  # zoomed in — actual range is 92–99%

    fig.tight_layout()
    plt.title('Accuracy vs. Sparsity Trade-off')
    plt.grid(True, alpha=0.3)
    plt.savefig('single_tradeoff_plot.png', dpi=300)
    plt.close()

    # Load best model for histogram — weights_only=False suppresses PyTorch 2.x FutureWarning
    model_best = SelfPruningNet()
    model_best.load_state_dict(torch.load(best_model_path, map_location='cpu', weights_only=False))
    
    # Gate Histogram Plot — use full (0,1) range so the spike near 0 is always visible
    all_gates = model_best.get_all_gates().detach().numpy()
    pruned_pct = (all_gates < 0.5).mean() * 100
    active_pct = 100 - pruned_pct
    plt.figure(figsize=(10, 6))
    plt.hist(all_gates, bins=100, color='purple', alpha=0.75, range=(0, 1),
             edgecolor='black', linewidth=0.2)
    plt.xlim(0, 1)   # force full range so spike near 0 is never clipped
    plt.axvline(x=0.5, color='red', linestyle='--', linewidth=1.5,
                label=f'Sparsity threshold (0.5)  |  {pruned_pct:.1f}% pruned')
    plt.title(f'Gate Value Distribution  \u03bb={best_lambda}\n'
              f'{pruned_pct:.1f}% gates pruned (< 0.5)  \u2502  {active_pct:.1f}% active')
    plt.xlabel('Gate Value  [sigmoid(gate_score / T)]')
    plt.ylabel('Number of weights  (log scale)')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('single_gate_histogram.png', dpi=300)
    plt.close()
    
    # Save results table to CSV for the report
    with open('results_summary.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['Lambda', 'Accuracy', 'Sparsity'])
        writer.writeheader()
        for r in results:
            writer.writerow({'Lambda': r['Lambda'], 'Accuracy': round(r['Accuracy'], 2), 'Sparsity': round(r['Sparsity'], 2)})
    
    print("Saved 'single_tradeoff_plot.png', 'single_gate_histogram.png', and 'results_summary.csv'")


# ==========================================
# 5. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Run for at least 20 epochs to allow gates time to slowly collapse to 0
    EPOCHS = 20
    lambda_values = [0.0001, 0.0003, 0.0005]
    results = []
    
    for l in lambda_values:
        test_acc, sparsity, path = train_model(l, epochs=EPOCHS, batch_size=128, device=device)
        results.append({
            "Lambda": l, "Accuracy": test_acc, "Sparsity": sparsity, "path": path
        })
            
    best_result = max(results, key=lambda x: x["Accuracy"])
    best_model_path = best_result["path"]
    best_lambda = best_result["Lambda"]
            
    print("\n--- Final Summary ---")
    print(f"{'Lambda':<10} | {'Test Accuracy (%)':<20} | {'Sparsity Level (%)':<20}")
    print("-" * 55)
    for r in results:
        print(f"{r['Lambda']:<10} | {r['Accuracy']:<20.2f} | {r['Sparsity']:<20.2f}")
    
    evaluate_and_plot(results, best_model_path, best_lambda)
