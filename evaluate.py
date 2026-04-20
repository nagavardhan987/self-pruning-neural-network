import torch
import matplotlib.pyplot as plt
import numpy as np
from model import SelfPruningNet
import csv

def evaluate_models():
    lambda_values = [0.0001, 0.001, 0.01]
    results = []
    
    # 1. Load results from CSV if exists
    try:
        with open('training_results.csv', 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                results.append({
                    'Lambda': float(row['Lambda']),
                    'Accuracy': float(row['Test Accuracy (%)']),
                    'Sparsity': float(row['Sparsity Level (%)'])
                })
    except FileNotFoundError:
        print("training_results.csv not found. Please run train.py first.")
        return

    # Plot 1: Trade-off plot (Dual axis)
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:red'
    ax1.set_xlabel('Lambda value (log scale)')
    ax1.set_xscale('log')
    ax1.set_ylabel('Test Accuracy (%)', color=color)
    lambdas = [r['Lambda'] for r in results]
    accs = [r['Accuracy'] for r in results]
    ax1.plot(lambdas, accs, marker='o', color=color, linewidth=2, label='Accuracy')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(0, 100)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('Sparsity Level (%)', color=color)
    sparsities = [r['Sparsity'] for r in results]
    ax2.plot(lambdas, sparsities, marker='s', color=color, linewidth=2, label='Sparsity')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0, 100)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.title('Accuracy vs. Sparsity Trade-off')
    plt.grid(True, alpha=0.3)
    plt.savefig('tradeoff_plot.png', dpi=300)
    plt.close()

    # Load the best model (usually lambda=0.001) for detailed plots
    best_lambda = 0.001
    model_path = f"model_lambda_{best_lambda}.pth"
    model = SelfPruningNet()
    try:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    except FileNotFoundError:
        print(f"Model file {model_path} not found.")
        return

    # Plot 2: Histogram of gate values
    all_gates = model.get_all_gates().detach().numpy()
    plt.figure(figsize=(10, 6))
    plt.hist(all_gates, bins=50, color='purple', alpha=0.7)
    plt.title(f'Distribution of Gate Values (λ={best_lambda})')
    plt.xlabel('Gate Value')
    plt.ylabel('Frequency (log scale)')
    plt.yscale('log') # Log scale helps see the active nodes clearly
    plt.grid(True, alpha=0.3)
    plt.savefig('gate_histogram.png', dpi=300)
    plt.close()
    
    # Plot 3: Per-layer sparsity
    layer_sparsity = model.get_layer_sparsity()
    layers = list(layer_sparsity.keys())
    sparsities = list(layer_sparsity.values())
    
    plt.figure(figsize=(10, 6))
    plt.bar(layers, sparsities, color='teal')
    plt.title(f'Sparsity per Layer (λ={best_lambda})')
    plt.ylabel('Sparsity (%)')
    plt.ylim(0, 100)
    plt.grid(axis='y', alpha=0.3)
    for i, v in enumerate(sparsities):
        plt.text(i, v + 1, f'{v:.1f}%', ha='center')
    plt.savefig('layer_sparsity.png', dpi=300)
    plt.close()

    print("Generated evaluation plots: tradeoff_plot.png, gate_histogram.png, layer_sparsity.png")

if __name__ == "__main__":
    evaluate_models()
