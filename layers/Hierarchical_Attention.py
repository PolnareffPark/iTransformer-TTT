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
                 pooling='statistical', use_variable_resolution=True, use_interaction_bridge=True,
                 use_global_interact=True, partition_strategy='softmax', dynamic_tokens_per_group=1):
        super(HierarchicalAttention, self).__init__()
        self.num_groups = num_groups
        self.d_model = d_model
        self.output_attention = output_attention
        self.pooling = pooling
        self.use_variable_resolution = use_variable_resolution
        self.use_interaction_bridge = use_interaction_bridge
        self.use_global_interact = use_global_interact
        self.partition_strategy = partition_strategy
        self.dynamic_tokens_per_group = dynamic_tokens_per_group
        self.dropout = nn.Dropout(attention_dropout)
        
        # Local Attention (Intra-group) - Used for fixed grouping or refinement
        self.local_attn = MultiHeadAttention(attention_dropout)
        
        # Global Attention (Inter-group) 
        self.global_attn = MultiHeadAttention(attention_dropout)
        
        # Dynamic Grouping (Phase 17)
        if pooling == 'dynamic':
            # 1. Dynamic Partitioning: Learnable mapping
            self.score_projection = nn.Linear(d_model, num_groups * dynamic_tokens_per_group)
            self.temp = nn.Parameter(torch.ones(1) * 0.1) 
            
            # 2. Variable Resolution: Salience Gate
            if use_variable_resolution:
                self.salience_gate = nn.Linear(d_model, 1)
            
            # 3. Cross-Interaction Bridge: Gated Integration
            if use_interaction_bridge:
                self.interaction_gate = nn.Linear(d_model * 2, d_model)
            self.refinement = nn.Linear(d_model, d_model)
        
        # Learnable Pooling (optional)
        if pooling == 'learnable':
            D = d_model // n_heads
            self.query_gen = nn.Parameter(torch.randn(1, 1, 1, D)) 
        
        # Integration layers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
    def _partition(self, logits):
        """Helper for different partitioning strategies"""
        temp = torch.abs(self.temp) + 1e-4
        
        if self.partition_strategy == 'gumbel':
            # Gumbel-Softmax for hard categorical assignment with gradients
            return F.gumbel_softmax(logits, tau=temp, hard=True, dim=-1)
        
        elif self.partition_strategy == 'topk':
            # Keep only top-k groups, zero out others (sharp sparsity)
            k = max(1, self.num_groups // 4) # Heuristic: keep 25%
            values, indices = torch.topk(logits, k, dim=-1)
            mask = torch.zeros_like(logits).scatter_(-1, indices, 1.0)
            weights = torch.softmax(logits / temp, dim=-1)
            return weights * mask / (weights * mask).sum(dim=-1, keepdim=True)
            
        else: # 'softmax' (standard)
            return torch.softmax(logits / temp, dim=-1)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        """
        queries/keys/values: (B, N, H, D)
        """
        B, N, H, D = queries.shape
        G = self.num_groups
        K = self.dynamic_tokens_per_group
        
        if self.pooling == 'dynamic':
            # --- [Phase 17+: Multi-Token Dynamic Grouping] ---
            x_raw = queries.reshape(B, N, H * D)
            v_raw = values.reshape(B, N, H * D)
            
            # 1. Salience Gate
            if self.use_variable_resolution:
                salience = torch.sigmoid(self.salience_gate(x_raw))
            else:
                salience = torch.zeros((B, N, 1), device=x_raw.device)
            
            # 2. Dynamic Partitioning with Multi-Token support
            # logits: (B, N, G*K)
            logits = self.score_projection(x_raw)
            weights = self._partition(logits) # (B, N, G*K)
            
            # 3. Dynamic Aggregation (Global Nodes: G*K)
            # group_reps: (B, G*K, d_model)
            group_reps_raw = torch.matmul(weights.transpose(-1, -2), v_raw)
            # Normalize by weight sum (Stability)
            weight_sum = weights.sum(dim=1, keepdim=True).transpose(-1, -2) + 1e-4
            group_reps_raw = group_reps_raw / weight_sum
            
            group_reps = group_reps_raw.reshape(B, G * K, H, D)
            
            # 4. Global Attention: Inter-Token Interaction
            if self.use_global_interact:
                out_global, attn_global = self.global_attn(group_reps, group_reps, group_reps)
                out_global_raw = out_global.reshape(B, G * K, H * D)
                out_context = torch.matmul(weights, out_global_raw)
            else:
                # Skip global interaction, contextual bias is zero
                out_context = torch.zeros((B, N, H * D), device=x_raw.device)

            # 6. Cross-Interaction Bridge (Gated Integration)
            if self.use_interaction_bridge:
                combined = torch.cat([out_context, v_raw], dim=-1)
                interaction = torch.sigmoid(self.interaction_gate(combined))
                gate = interaction * (1.0 - salience) 
                out_raw = gate * out_context + (1.0 - gate) * v_raw
            else:
                # Fallback to simple addition/residual if bridge is off
                out_raw = out_context + v_raw
            
            out = self.refinement(out_raw).reshape(B, N, H, D)
            
            if self.output_attention:
                res = {'attn': attn_global, 'salience': salience}
                if self.use_interaction_bridge:
                    res['interaction'] = interaction
                return out, res
            else:
                return out, None

        # --- [Original Hierarchical Flow (Fixed Grouping)] ---
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
        if self.use_global_interact:
            if self.pooling == 'statistical':
                # Statistical Pooling: Capture more information than just mean
                rep_mean = out_local.mean(dim=2) 
                rep_max = out_local.max(dim=2)[0]
                rep_std = out_local.std(dim=2)
                
                group_reps = torch.cat([rep_mean, rep_max, rep_std], dim=1)
                out_global_combined, attn_global = self.global_attn(group_reps, group_reps, group_reps) 
                
                g_mean, g_max, g_std = torch.chunk(out_global_combined, 3, dim=1)
                out_global = g_mean + g_max + g_std
                
            elif self.pooling == 'learnable':
                q_pool = self.query_gen.expand(G * B, 1, H, D)
                rep_learnable, _ = self.local_attn(q_pool, q_local, v_local)
                group_reps = rep_learnable.view(G, B, 1, H, D).transpose(0, 1).reshape(B, G, H, D)
                
                out_global, attn_global = self.global_attn(group_reps, group_reps, group_reps)
                
            else: # 'mean' (Baseline)
                group_reps = out_local.mean(dim=2)
                out_global, attn_global = self.global_attn(group_reps, group_reps, group_reps)
            
            # --- [Step 3: Integration] ---
            out_global = out_global.unsqueeze(2).expand(-1, -1, M, -1, -1)
            out = out_local + self.dropout(out_global)
        else:
            # Skip Step 2 & 3, just use local info
            out = out_local
            attn_global = None
        
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
