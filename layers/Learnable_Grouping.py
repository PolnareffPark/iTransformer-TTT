import torch
import torch.nn as nn
import torch.nn.functional as F

class LearnableGrouping(nn.Module):
    """
    Learnable Variate Grouping Layer
    assigns N variates into G groups using a soft-assignment mechanism.
    """
    def __init__(self, n_vars, n_groups, d_model, dropout=0.1):
        super(LearnableGrouping, self).__init__()
        self.n_vars = n_vars
        self.n_groups = n_groups
        self.d_model = d_model
        
        # Group assignment scoring layer
        # Maps embedded variate (d_model) to group scores (n_groups)
        self.score_projection = nn.Linear(d_model, n_groups)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        x: (B, N, E) where N is number of variates, E is d_model
        returns:
            group_representatives: (B, G, E)
            assignment_weights: (B, N, G)
        """
        B, N, E = x.shape
        G = self.n_groups
        
        # 1. Compute assignment scores for each variate: (B, N, G)
        # We use the embedded representation of each variate to decide its group.
        logits = self.score_projection(x) # (B, N, G)
        weights = F.softmax(logits, dim=-1) # (B, N, G)
        weights = self.dropout(weights)
        
        # 2. Compute group representatives (soft aggregation): (B, G, E)
        # weights.transpose(-1, -2): (B, G, N)
        # x: (B, N, E)
        group_representatives = torch.matmul(weights.transpose(-1, -2), x) # (B, G, E)
        
        # Optional: Normalize group representatives if needed
        # group_representatives = group_representatives / (weights.sum(dim=1, keepdim=True).transpose(-1, -2) + 1e-6)
        
        return group_representatives, weights

class VariateReconstruction(nn.Module):
    """
    Broadcasts group-level features back to individual variates
    """
    def __init__(self, d_model):
        super(VariateReconstruction, self).__init__()
        # Optional: refinement layer after broadcasting
        self.refinement = nn.Linear(d_model, d_model)
        
    def forward(self, group_features, weights):
        """
        group_features: (B, G, E)
        weights: (B, N, G)
        returns: (B, N, E)
        """
        # Broadcast group features back using assignment weights
        # (B, N, G) @ (B, G, E) -> (B, N, E)
        x_reconstructed = torch.matmul(weights, group_features)
        return self.refinement(x_reconstructed)
