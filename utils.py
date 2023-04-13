import torch
import torch.nn as nn
import math

# To compute the posterior predictive probability
# The factor lambda is often set to either pi/8 or 3/pi^2
def mean_field_logits(logits, pred_cov, mean_field_factor):
    logits_scale = torch.sqrt(1.0 + torch.diag(pred_cov) * mean_field_factor)
    logits = logits / logits_scale.unsqueeze(-1)
    return logits

def RandomFourierFeatures(in_dim, out_dim):
    # Returns a linear layer whose parameters are frozen
    lin = nn.Linear(in_dim, out_dim)
    
    nn.init.normal_(lin.weight, mean=0.0, std=1.0) # Based on paper
    lin.weight.requires_grad = False # Freeze the weights
    
    nn.init.uniform_(lin.bias, a=0.0, b=2.0 * math.pi) # Based on paper
    lin.bias.requires_grad = False # Freeze the biases
    
    return lin
