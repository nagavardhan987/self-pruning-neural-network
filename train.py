import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import SelfPruningNet
import csv

def train_model(lambda_val: float, epochs: int = 15, batch_size: int = 128, device: str = 'cuda'):
    print(f"\nTraining with lambda = {lambda_val}")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    model = SelfPruningNet().to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            
            # Standard classification loss
            classification_loss = criterion(outputs, labels)
            
            # Custom sparsity loss: L1 norm of all gate values
            # Since gates are from sigmoid (0 to 1), L1 norm is just the sum
            all_gates = model.get_all_gates()
            sparsity_loss = torch.sum(all_gates)
            
            # Total Loss formulation
            loss = classification_loss + lambda_val * sparsity_loss
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        scheduler.step()
        
        # Evaluate at epoch end
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
        print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss/len(trainloader):.4f} - Train Acc: {train_acc:.2f}% - Test Acc: {test_acc:.2f}% - Sparsity: {sparsity:.2f}%")

    model_path = f"model_lambda_{lambda_val}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Saved model to {model_path}")
    return test_acc, sparsity

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    lambda_values = [0.0001, 0.001, 0.01]
    results = []
    
    # We train fewer epochs to not hog the GPU eternally, typically 10-15 is enough to see sparsity.
    for l in lambda_values:
        test_acc, sparsity = train_model(l, epochs=20, batch_size=128, device=device)
        results.append({"Lambda": l, "Test Accuracy (%)": test_acc, "Sparsity Level (%)": sparsity})
        
    print("\n--- Final Summary ---")
    print(f"{'Lambda':<10} | {'Test Accuracy (%)':<20} | {'Sparsity Level (%)':<20}")
    print("-" * 55)
    for r in results:
        print(f"{r['Lambda']:<10} | {r['Test Accuracy (%)']:<20.2f} | {r['Sparsity Level (%)']:<20.2f}")
    
    with open('training_results.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["Lambda", "Test Accuracy (%)", "Sparsity Level (%)"])
        writer.writeheader()
        writer.writerows(results)
