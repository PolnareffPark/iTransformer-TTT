import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

class HierarchicalAttention(nn.Module):
    """
    Hybrid Hierarchical Attention (Phase 12)
    Logic: Local Direct Interaction + Global Bridge
    Complexity: O(N^2/G + G^2)
    """
    def __init__(self, n_vars, num_groups, d_model, attention_dropout=0.1, output_attention=False):
        super(HierarchicalAttention, self).__init__()
        self.num_groups = num_groups
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        
        # Local Attention (Intra-group) - Shared across groups to save parameters
        self.local_attn = MultiHeadAttention(attention_dropout)
        
        # Global Attention (Inter-group) - Between group representatives
        self.global_attn = MultiHeadAttention(attention_dropout)
        
        # Integration layers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        """
        queries/keys/values: (B, N, H, D)
        """
        B, N, H, D = queries.shape
        G = self.num_groups
        
        # 1. Padding if N is not divisible by G
        pad_len = (G - (N % G)) % G
        if pad_len > 0:
            queries = F.pad(queries, (0, 0, 0, 0, 0, pad_len))
            keys = F.pad(keys, (0, 0, 0, 0, 0, pad_len))
            values = F.pad(values, (0, 0, 0, 0, 0, pad_len))
        
        N_padded = N + pad_len
        M = N_padded // G # Number of variates per group
        
        # --- [Step 1: Local Attention (Intra-group)] ---
        # Reshape to (B*G, M, H, D)
        q_local = queries.view(B, G, M, H, D).transpose(1, 0).reshape(G * B, M, H, D)
        k_local = keys.view(B, G, M, H, D).transpose(1, 0).reshape(G * B, M, H, D)
        v_local = values.view(B, G, M, H, D).transpose(1, 0).reshape(G * B, M, H, D)
        
        # Perform self-attention within each group: O(G * M^2) = O(N^2/G)
        out_local, _ = self.local_attn(q_local, k_local, v_local) # (G*B, M, H, D)
        
        # Reshape back to (B, G, M, H, D)
        out_local = out_local.view(G, B, M, H, D).transpose(0, 1)
        
        # --- [Step 2: Global Attention (Inter-group)] ---
        # Pool local features to get group representatives: (B, G, H, D)
        # Using mean pooling for simplicity and robustness
        group_reps = out_local.mean(dim=2) # (B, G, H, D)
        
        # Global interaction among G groups: O(G^2)
        out_global, attn_global = self.global_attn(group_reps, group_reps, group_reps) # (B, G, H, D)
        
        # --- [Step 3: Integration] ---
        # Broadcast global context back to all variates in the group
        # (B, G, 1, H, D) expand to (B, G, M, H, D)
        out_global = out_global.unsqueeze(2).expand(-1, -1, M, -1, -1)
        
        # Composite feature: Local + Global (Residual)
        out = out_local + self.dropout(out_global)
        
        # Reshape back to (B, N_padded, H, D) and strip padding
        out = out.reshape(B, N_padded, H, D)
        if pad_len > 0:
            out = out[:, :N, :, :]
            
        if self.output_attention:
            return out, attn_global
        else:
            return out, None

class MultiHeadAttention(nn.Module):
    """Refined Multi-head Attention logic"""
    def __init__(self, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values):
        # queries: (B, L, H, D), keys: (B, S, H, D)
        B, L, H, D = queries.shape
        scale = 1. / sqrt(D)
        
        # Attention scores: (B, H, L, S)
        scores = torch.einsum("blhd,bshd->bhls", queries, keys)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)
        
        return V.contiguous(), A
