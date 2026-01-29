import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import sqrt

class HierarchicalAttention(nn.Module):
    def __init__(self, num_groups, attention_dropout=0.1, output_attention=False):
        super(HierarchicalAttention, self).__init__()
        self.num_groups = num_groups
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        
    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        # queries: (B, N, H, D) where N is number of variates
        B, N, H, D = queries.shape
        G = self.num_groups
        
        # 1. Padding if N is not divisible by G
        pad_len = (G - (N % G)) % G
        if pad_len > 0:
            queries = F.pad(queries, (0, 0, 0, 0, 0, pad_len))
            keys = F.pad(keys, (0, 0, 0, 0, 0, pad_len))
            values = F.pad(values, (0, 0, 0, 0, 0, pad_len))
        
        N_padded = N + pad_len
        M = N_padded // G # Variates per group
        
        # 2. Reshape for Intra-group Attention
        # (B, G, M, H, D)
        q_grouped = queries.view(B, G, M, H, D)
        k_grouped = keys.view(B, G, M, H, D)
        v_grouped = values.view(B, G, M, H, D)
        
        # Intra-group Attention: (B*G, H, M, D)
        q_intra = q_grouped.transpose(2, 3).reshape(B * G, H, M, D)
        k_intra = k_grouped.transpose(2, 3).reshape(B * G, H, M, D)
        v_intra = v_grouped.transpose(2, 3).reshape(B * G, H, M, D)
        
        scale = 1. / sqrt(D)
        
        # scores: (B*G, H, M, M)
        scores_intra = torch.matmul(q_intra, k_intra.transpose(-1, -2)) * scale
        attn_intra = self.dropout(torch.softmax(scores_intra, dim=-1))
        out_intra = torch.matmul(attn_intra, v_intra) # (B*G, H, M, D)
        
        # 3. Inter-group (Global) Attention
        # First, pool to get group representatives: (B, G, H, D)
        q_global = q_grouped.mean(dim=2) # (B, G, H, D)
        k_global = k_grouped.mean(dim=2)
        v_global = v_grouped.mean(dim=2)
        
        # Transpose for attention: (B, H, G, D)
        q_global = q_global.transpose(1, 2)
        k_global = k_global.transpose(1, 2)
        v_global = v_global.transpose(1, 2)
        
        scores_global = torch.matmul(q_global, k_global.transpose(-1, -2)) * scale
        attn_global = self.dropout(torch.softmax(scores_global, dim=-1))
        out_global = torch.matmul(attn_global, v_global) # (B, H, G, D)
        
        # 4. Combine (Residual-like sum/expand)
        # out_global: (B, H, G, D) -> (B, G, H, D) -> (B, G, M, H, D)
        out_global = out_global.transpose(1, 2).unsqueeze(2).expand(B, G, M, H, D)
        
        # out_intra: (B*G, H, M, D) -> (B, G, H, M, D) -> (B, G, M, H, D)
        out_intra = out_intra.view(B, G, H, M, D).transpose(2, 3)
        
        # Sum of Local + Global
        out = out_intra + out_global
        
        # 5. Reshape back to (B, N, H, D)
        out = out.reshape(B, N_padded, H, D)
        if pad_len > 0:
            out = out[:, :N, :, :]
            
        if self.output_attention:
            return out, attn_intra # returning only local attn for now
        else:
            return out, None
