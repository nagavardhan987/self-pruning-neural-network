import math
import torch
import torch.nn as nn
import torch.nn.functional as F

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
            
        self.reset_parameters()
        
    def reset_parameters(self) -> None:
        # Standard Kaiming uniform initialization for linear weights
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
        # Initialize gate scores to a positive value (e.g. 2.0) so gates start near 1
        nn.init.constant_(self.gate_scores, 2.0)
        
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply sigmoid to strictly bound the gate values between 0 and 1
        gates = torch.sigmoid(self.gate_scores)
        
        # Multiply weights by their corresponding gate score
        pruned_weights = self.weight * gates
        
        # Perform standard linear operation
        return F.linear(x, pruned_weights, self.bias)
