import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import AttentionLayer
from layers.Embed import DataEmbedding_inverted
from layers.Hierarchical_Attention import HierarchicalAttention
from layers.Learnable_Grouping import LearnableGrouping, VariateReconstruction
import numpy as np


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
        
        # VG-iT Specific
        self.num_groups = getattr(configs, 'num_groups', 8)
        self.use_learnable_grouping = getattr(configs, 'use_learnable_grouping', False)
        self.pooling = getattr(configs, 'pooling', 'statistical')
        self.use_shuffling = getattr(configs, 'use_shuffling', 0)
        self.variate_indices = None # Will be initialized in forecast based on N

        # Embedding
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        self.class_strategy = configs.class_strategy
        
        # Optional: Learnable Grouping Layer
        if self.use_learnable_grouping:
            self.grouping_layer = LearnableGrouping(configs.enc_in, self.num_groups, configs.d_model, configs.dropout)
            self.reconstruction_layer = VariateReconstruction(configs.d_model)

        # Encoder with Hierarchical Attention
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                         HierarchicalAttention(configs.enc_in, self.num_groups, configs.d_model,
                                            n_heads=configs.n_heads,
                                            attention_dropout=configs.dropout,
                                            output_attention=configs.output_attention,
                                            pooling=self.pooling,
                                            use_variable_resolution=getattr(configs, 'use_variable_resolution', True),
                                            use_interaction_bridge=getattr(configs, 'use_interaction_bridge', True),
                                            use_global_interact=getattr(configs, 'use_global_interact', 1),
                                            partition_strategy=getattr(configs, 'partition_strategy', 'softmax'),
                                            dynamic_tokens_per_group=getattr(configs, 'dynamic_tokens_per_group', 1)), 
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        B, T, N = x_enc.shape
        
        # Ablation 1: Shuffling variate order
        if self.use_shuffling:
            if self.variate_indices is None or self.variate_indices.shape[0] != N:
                # Use a fixed seed for shuffling based on the existing seed + a constant
                g = torch.Generator()
                g.manual_seed(42) # Fixed seed for the shuffle itself to be consistent across batches
                self.variate_indices = torch.randperm(N, generator=g).to(x_enc.device)
                self.inverse_indices = torch.argsort(self.variate_indices)
            
            x_enc = x_enc[:, :, self.variate_indices]
            if self.use_norm:
                stdev = stdev[:, :, self.variate_indices]
                means = means[:, :, self.variate_indices]

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        
        # Project and Filter covariates (if any)
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N]
        
        # De-Normalize while in current order (shuffled or original)
        if self.use_norm:
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        # Restore original order if shuffled (Ablation 1)
        if self.use_shuffling:
            dec_out = dec_out[:, :, self.inverse_indices]

        return dec_out, attns

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out, attns = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]
