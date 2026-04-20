import torch
import torch.nn as nn
from prunable_layer import PrunableLinear

class SelfPruningNet(nn.Module):
    def __init__(self, input_dim=3072, num_classes=10):
        super(SelfPruningNet, self).__init__()
        self.flatten = nn.Flatten()
        
        # A simple Feed-Forward Network architecture for CIFAR-10
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
        
    def get_all_gates(self):
        """
        Returns a single 1D tensor containing all current gate values (after sigmoid)
        from all PrunableLinear layers in the network.
        """
        gates = []
        for module in self.modules():
            if isinstance(module, PrunableLinear):
                # We need the values, not the raw scores
                gate_vals = torch.sigmoid(module.gate_scores)
                gates.append(gate_vals.view(-1))
        
        if len(gates) > 0:
            return torch.cat(gates)
        return torch.tensor([])

    def get_sparsity(self, threshold=1e-2):
        """
        Calculates the overall network sparsity level.
        Returns the percentage of weights whose specific gate is under the threshold.
        """
        all_gates = self.get_all_gates()
        if len(all_gates) == 0:
            return 0.0
            
        pruned_count = (all_gates < threshold).sum().item()
        total_count = all_gates.numel()
        return (pruned_count / total_count) * 100.0

    def get_layer_sparsity(self, threshold=1e-2):
        """
        Returns the sparsity breakdown per PrunableLinear layer.
        """
        layer_stats = {}
        layer_idx = 1
        for name, module in self.named_modules():
            if isinstance(module, PrunableLinear):
                gates = torch.sigmoid(module.gate_scores)
                pruned_count = (gates < threshold).sum().item()
                total_count = gates.numel()
                layer_stats[f"Layer_{layer_idx}"] = (pruned_count / total_count) * 100.0
                layer_idx += 1
        return layer_stats
