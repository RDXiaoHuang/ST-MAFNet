import math

import torch
from torch import Tensor
from torch.nn import functional as F
from torch import nn


class AdaptiveGraphAttention(nn.Module):
    def __init__(self, d_model, num_nodes, node_dim=64, num_heads=4, dropout=0.1):
        super(AdaptiveGraphAttention, self).__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.num_nodes = num_nodes
        self.scale = math.sqrt(self.d_k)
        
        self.node_emb1 = nn.Parameter(torch.randn(num_nodes, node_dim))
        self.node_emb2 = nn.Parameter(torch.randn(num_nodes, node_dim))
        nn.init.xavier_uniform_(self.node_emb1)
        nn.init.xavier_uniform_(self.node_emb2)
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout)
        
    def get_adaptive_adj(self):
        adj = torch.softmax(torch.relu(self.node_emb1 @ self.node_emb2.T), dim=-1)
        return adj
        
    def forward(self, x):
        B, N, D = x.shape
        
        adj = self.get_adaptive_adj()
        
        Q = self.q_proj(x).view(B, N, self.num_heads, self.d_k).permute(0, 2, 1, 3)
        K = self.k_proj(x).view(B, N, self.num_heads, self.d_k).permute(0, 2, 1, 3)
        V = self.v_proj(x).view(B, N, self.num_heads, self.d_k).permute(0, 2, 1, 3)
        
        attn_scores = (Q @ K.transpose(-2, -1)) / self.scale
        
        adj_expanded = adj.unsqueeze(0).unsqueeze(0)
        attn_scores = attn_scores + torch.log(adj_expanded + 1e-8)
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        out = attn_weights @ V
        out = out.permute(0, 2, 1, 3).contiguous().view(B, N, D)
        out = self.out_proj(out)
        out = self.dropout(out)
        
        return out


class AdaptiveGraphAttentionLayer(nn.Module):
    def __init__(self, d_model, num_nodes, node_dim=64, nhead=4, dim_feedforward=None, dropout=0.1):
        super(AdaptiveGraphAttentionLayer, self).__init__()
        
        dim_feedforward = dim_feedforward or 4 * d_model
        
        self.self_attn = AdaptiveGraphAttention(d_model, num_nodes, node_dim, nhead, dropout)
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src: Tensor) -> Tensor:
        src2 = self.self_attn(src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        src2 = self.linear2(self.dropout(F.gelu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src


class AdaptiveGraphAttentionEncoder(nn.Module):
    def __init__(self, d_model, num_nodes, out_dim, num_layers=2, nhead=4, 
                 dim_feedforward=None, dropout=0.1, node_dim=64):
        super(AdaptiveGraphAttentionEncoder, self).__init__()
        
        self.layers = nn.ModuleList([
            AdaptiveGraphAttentionLayer(d_model, num_nodes, node_dim, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim)
        )
        self.norm = nn.LayerNorm(out_dim)
        
    def forward(self, src: Tensor) -> Tensor:
        x = src
        for layer in self.layers:
            x = layer(x)
        
        output = self.output_proj(x)
        output = self.norm(output)
        return output
