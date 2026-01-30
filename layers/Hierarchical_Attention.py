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
    def __init__(self, n_vars, num_groups, d_model, n_heads=8, attention_dropout=0.1, output_attention=False, 
                 pooling='statistical'):
        super(HierarchicalAttention, self).__init__()
        self.num_groups = num_groups
        self.output_attention = output_attention
        self.pooling = pooling
        self.dropout = nn.Dropout(attention_dropout)
        
        # Local Attention (Intra-group)
        self.local_attn = MultiHeadAttention(attention_dropout)
        
        # Global Attention (Inter-group) 
        self.global_attn = MultiHeadAttention(attention_dropout)
        
        # Learnable Pooling (optional)
        if pooling == 'learnable':
            # Initialize query per head
            D = d_model // n_heads
            self.query_gen = nn.Parameter(torch.randn(1, 1, 1, D)) 
        
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
        if self.pooling == 'statistical':
            # Statistical Pooling: Capture more information than just mean
            # (B, G, H, D)
            rep_mean = out_local.mean(dim=2) 
            rep_max = out_local.max(dim=2)[0]
            rep_std = out_local.std(dim=2)
            
            # Multi-Token Bridge: Attend to all statistical aspects
            # Shape: (B, 3*G, H, D)
            group_reps = torch.cat([rep_mean, rep_max, rep_std], dim=1)
            
            # Global interaction among 3*G tokens: O((3G)^2)
            out_global_combined, attn_global = self.global_attn(group_reps, group_reps, group_reps) 
            
            # Split global context back into 3 parts and aggregate (sum)
            g_mean, g_max, g_std = torch.chunk(out_global_combined, 3, dim=1)
            out_global = g_mean + g_max + g_std
            
        elif self.pooling == 'learnable':
            # Learnable Pooling: Using a learnable query to extract features per group
            # query_gen: (1, 1, 1, D) -> (B*G, 1, H, D)
            # q_pool = self.query_gen.expand(B*G, -1, H, -1)
            # But simpler: Use the weighted average of variates as representives
            # For now, let's implement a simple attention-based pooling
            q_pool = self.query_gen.expand(G * B, 1, H, D)
            # Attend to local variates: (G*B, 1, H, D)
            rep_learnable, _ = self.local_attn(q_pool, q_local, v_local)
            group_reps = rep_learnable.view(G, B, 1, H, D).transpose(0, 1).reshape(B, G, H, D)
            
            out_global, attn_global = self.global_attn(group_reps, group_reps, group_reps)
            
        else: # 'mean' (Baseline)
            group_reps = out_local.mean(dim=2)
            out_global, attn_global = self.global_attn(group_reps, group_reps, group_reps)
        
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
