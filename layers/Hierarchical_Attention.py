import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import sqrt
from layers.Learnable_Grouping import LearnableGrouping, VariateReconstruction

class HierarchicalAttention(nn.Module):
    def __init__(self, n_vars, num_groups, d_model, attention_dropout=0.1, output_attention=False):
        super(HierarchicalAttention, self).__init__()
        self.num_groups = num_groups
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        
        self.grouping = LearnableGrouping(n_vars, num_groups, d_model, dropout=attention_dropout)
        self.reconstruction = VariateReconstruction(d_model)
        self.inner_attention = FullAttention(attention_dropout, output_attention)
        
    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        """
        queries: (B, N, H, D)
        Note: Hierarchical logic for Inverted dimension (N = variates)
        """
        B, N, H, D = queries.shape
        E = H * D # Total embedding dim
        
        # Reshape to (B, N, E) for grouping logic
        q_v = queries.reshape(B, N, E)
        k_v = keys.reshape(B, N, E)
        v_v = values.reshape(B, N, E)
        
        # 1. Learnable Grouping: (B, G, E)
        # We use queries to decide grouping (can also use a separate context)
        group_reps, assignment_weights = self.grouping(q_v) # reps: (B, G, E), weights: (B, N, G)
        
        # 2. Inter-group (Global) Attention
        # Transform group_reps back to (B, G, H, D) for multi-head attention
        q_global = group_reps.view(B, self.num_groups, H, D)
        k_global = k_v.view(B, N, H, D).mean(dim=1, keepdim=True).expand(-1, self.num_groups, -1, -1) # Simplified global context
        v_global = v_v.view(B, N, H, D).mean(dim=1, keepdim=True).expand(-1, self.num_groups, -1, -1)
        
        # Global Attention between groups (or group-to-all)
        # For efficiency, we can do self-attention among group_reps
        out_global_reps, attn_global = self.inner_attention(q_global, q_global, q_global, None) # (B, G, H, D)
        
        # 3. Reconstruction (Broadcasting back to N variates)
        # out_global_reps: (B, G, E)
        out_global_reps = out_global_reps.reshape(B, self.num_groups, E)
        out_v = self.reconstruction(out_global_reps, assignment_weights) # (B, N, E)
        
        # 4. Intra-group (Local) Refinement
        # Residual connection and final reshape
        out = out_v.reshape(B, N, H, D)
        
        if self.output_attention:
            return out, attn_global
        else:
            return out, None

class FullAttention(nn.Module):
    """Simplified Multi-head Attention logic for internal use"""
    def __init__(self, dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.output_attention = output_attention

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, D = queries.shape
        scale = 1. / sqrt(D)
        
        scores = torch.einsum("blhd,bshd->bhls", queries, keys)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)
        
        return V.contiguous(), A
