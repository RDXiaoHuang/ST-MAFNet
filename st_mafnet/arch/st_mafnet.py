import torch
import torch.nn as nn
import torch.nn.functional as F

from .adap_attention import AdaptiveGraphAttentionEncoder


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=None, dropout=0.2):
        super().__init__()
        output_dim = output_dim or input_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        self.residual_proj = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()

    def forward(self, x):
        out = self.fc2(self.dropout(self.act(self.fc1(x))))
        return out + self.residual_proj(x)


class FusionModel(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers=2, dropout=0.2):
        super().__init__()
        layers = []
        for i in range(num_layers):
            if i == num_layers - 1:
                layers.append(MLP(input_dim, input_dim, output_dim, dropout=dropout))
            else:
                layers.append(MLP(input_dim, input_dim, input_dim, dropout=dropout))
        self.fusion_model = nn.Sequential(*layers)

    def forward(self, *features):
        fusion_input = torch.cat(features, dim=-1)
        return self.fusion_model(fusion_input)


class TConvLayer(nn.Module):
    def __init__(self, features=64, kernel_size=4, dropout=0.1):
        super(TConvLayer, self).__init__()
        self.conv = nn.Conv2d(features, features, (1, kernel_size))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.pad(x, (1, 0, 0, 0))
        out = self.dropout(self.relu(self.conv(x)))
        return out + x[..., -out.size(-1):]


class MultiScaleTemporalEncoder(nn.Module):
    def __init__(self, in_dim=64, hidden_dim=64, input_len=12, num_scales=4, dropout=0.1, kernel_sizes=None):

        super(MultiScaleTemporalEncoder, self).__init__()
        
        self.num_scales = num_scales
        self.hidden_dim = hidden_dim
        
        if in_dim != hidden_dim:
            self.start_conv = nn.Conv2d(in_dim, hidden_dim, kernel_size=(1, 1))
        else:
            self.start_conv = None
        
        if kernel_sizes is None:
            kernel_size = int(input_len / num_scales + 1)
            self.kernel_sizes = [kernel_size] * num_scales
        elif isinstance(kernel_sizes, int):
            self.kernel_sizes = [kernel_sizes] * num_scales
        elif isinstance(kernel_sizes, (list, tuple)):
            assert len(kernel_sizes) == num_scales, \
                f"Length of kernel_sizes ({len(kernel_sizes)}) must equal num_scales ({num_scales})"
            self.kernel_sizes = list(kernel_sizes)
        else:
            raise ValueError(f"kernel_sizes must be None, int, or list/tuple, got {type(kernel_sizes)}")
        
        self.tcn_layers = nn.ModuleList([
            TConvLayer(features=hidden_dim, kernel_size=self.kernel_sizes[i], dropout=dropout)
            for i in range(num_scales)
        ])

    def forward(self, x):
        if self.start_conv is not None:
            x = self.start_conv(x)
        
        scale_features = []
        for i, tcn_layer in enumerate(self.tcn_layers):
            x = tcn_layer(x)
            scale_feat = x[..., -1].permute(0, 2, 1)
            scale_features.append(scale_feat)
        
        return scale_features


class Graph_Projection(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.act_fn = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        return x + self.fc2(x)


class DualGraphEncoder(nn.Module):
    def __init__(self, num_nodes, node_dim, dropout=0.2, if_forward_graph=True, if_backward_graph=True):
        super().__init__()
        self.num_nodes = num_nodes
        self.node_dim = node_dim
        self.if_forward_graph = if_forward_graph
        self.if_backward_graph = if_backward_graph
        
        if if_forward_graph:
            self.forward_graph = Graph_Projection(input_dim=num_nodes, hidden_dim=node_dim, dropout=dropout)
        if if_backward_graph:
            self.backward_graph = Graph_Projection(input_dim=num_nodes, hidden_dim=node_dim, dropout=dropout)

    def forward(self, supports, batch_size):
        device = next(self.parameters()).device
        
        if self.if_forward_graph:
            adj_forward = supports[0].to(device)
            forward_graph = self.forward_graph(adj_forward.unsqueeze(0))
            forward_graph_emb = forward_graph.expand(batch_size, -1, -1)
        else:
            forward_graph_emb = None
        
        if self.if_backward_graph:
            adj_backward = supports[1].to(device)
            backward_graph = self.backward_graph(adj_backward.unsqueeze(0))
            backward_graph_emb = backward_graph.expand(batch_size, -1, -1)
        else:
            backward_graph_emb = None
        
        return forward_graph_emb, backward_graph_emb


class ST_Multi_Fusion(nn.Module):
    def __init__(self, encoder_dim, time_emb_dim, node_dim, adaptive_dim, output_dim, 
                 num_layers, dropout, first, temporal_dim=64,
                 if_adaptive_graph=True, if_forward_graph=True, if_backward_graph=True):
        super(ST_Multi_Fusion, self).__init__()
        
        self.first = first
        self.if_adaptive_graph = if_adaptive_graph
        self.if_forward_graph = if_forward_graph
        self.if_backward_graph = if_backward_graph

        base_dim = time_emb_dim + temporal_dim
        if if_forward_graph or if_backward_graph:
            base_dim += node_dim
        
        if not first:
            base_dim += output_dim
        
        if if_forward_graph:
            self.forward_proj = nn.Sequential(
                nn.Linear(base_dim, output_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )
        
        if if_backward_graph:
            self.backward_proj = nn.Sequential(
                nn.Linear(base_dim, output_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )
        
        adaptive_base_dim = time_emb_dim
        if if_adaptive_graph:
            adaptive_base_dim += adaptive_dim
        
        if not first:
            adaptive_base_dim += output_dim
            
        self.adaptive_proj = nn.Sequential(
            nn.Linear(adaptive_base_dim, output_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        fusion_count = 1  
        if if_forward_graph:
            fusion_count += 1
        if if_backward_graph:
            fusion_count += 1
        fusion_input_dim = output_dim * fusion_count
        self.fusion_model = FusionModel(fusion_input_dim, output_dim, num_layers=num_layers, dropout=dropout)

    def forward(self, time_emb, forward_graph_emb, backward_graph_emb, 
                temporal_feat, adaptive_emb, pred_emb=None):
        base_feat = torch.cat([time_emb, temporal_feat], dim=-1)

        if self.first or pred_emb is None or len(pred_emb) == 0:
            if self.if_forward_graph and forward_graph_emb is not None:
                forward_input = torch.cat([base_feat, forward_graph_emb], dim=-1)
            if self.if_backward_graph and backward_graph_emb is not None:
                backward_input = torch.cat([base_feat, backward_graph_emb], dim=-1)

            adaptive_input_list = [time_emb]
            if self.if_adaptive_graph and adaptive_emb is not None:
                adaptive_input_list.append(adaptive_emb)
            adaptive_input = torch.cat(adaptive_input_list, dim=-1)
        else:
            pred = pred_emb[0]
            if self.if_forward_graph and forward_graph_emb is not None:
                forward_input = torch.cat([base_feat, forward_graph_emb, pred], dim=-1)
            if self.if_backward_graph and backward_graph_emb is not None:
                backward_input = torch.cat([base_feat, backward_graph_emb, pred], dim=-1)
            adaptive_input_list = [time_emb]
            if self.if_adaptive_graph and adaptive_emb is not None:
                adaptive_input_list.append(adaptive_emb)
            adaptive_input_list.append(pred)
            adaptive_input = torch.cat(adaptive_input_list, dim=-1)
        
        fusion_list = []
        if self.if_forward_graph and forward_graph_emb is not None:
            h_forward = self.forward_proj(forward_input)
            fusion_list.append(h_forward)
        if self.if_backward_graph and backward_graph_emb is not None:
            h_backward = self.backward_proj(backward_input)
            fusion_list.append(h_backward)
        
        h_adaptive = self.adaptive_proj(adaptive_input)
        fusion_list.append(h_adaptive)
        
        output = self.fusion_model(*fusion_list)
        
        return output


class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, num_layers, dropout):
        super(Decoder, self).__init__()
        
        layers = []
        current_dim = input_dim
        
        for i in range(num_layers):
            if i == num_layers - 1:
                layers.append(MLP(current_dim, hidden_dim, out_dim, dropout=dropout))
            else:
                layers.append(MLP(current_dim, hidden_dim, hidden_dim, dropout=dropout))
                current_dim = hidden_dim
        
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class STMAFNet(nn.Module):
    def __init__(
        self,
        num_nodes,
        supports=None,
        in_channels=1,
        hidden_dim=64,
        skip_channels=256,
        input_len=12,
        out_dim=12,
        num_scales=4,
        decoder_layers=2,
        fusion_layers=2,
        dropout=0.2,
        node_dim=64,
        adp_dim=128,
        time_of_day_size=288,
        day_of_week_size=7,
        temp_dim_tid=32,
        temp_dim_diw=32,
        if_time_in_day=True,
        if_day_in_week=True,
        adaptive_nhead=4,
        adaptive_layers=2,
        adaptive_embedding_dim=64,
        if_adaptive_graph=True,
        if_forward_graph=True,
        if_backward_graph=True,
        if_adp_emb=True,
        if_multi_scale_anchor=True,
        kernel_sizes=None,
        **kwargs
    ):

        super(STMAFNet, self).__init__()

        self.num_nodes = num_nodes
        self.input_len = input_len
        self.out_dim = out_dim
        self.supports = supports
        self.hidden_dim = hidden_dim
        self.if_multi_scale_anchor = if_multi_scale_anchor
        self.num_scales = num_scales if if_multi_scale_anchor else 1
        
        self.if_adaptive_graph = if_adaptive_graph
        self.if_forward_graph = if_forward_graph
        self.if_backward_graph = if_backward_graph
        self.if_adp_emb = if_adp_emb

        self.if_time_in_day = if_time_in_day
        self.if_day_in_week = if_day_in_week
        self.time_of_day_size = time_of_day_size
        self.day_of_week_size = day_of_week_size

        self.adaptive_embedding_dim = adaptive_embedding_dim if if_adp_emb else 0
        if if_adp_emb:
            self.adaptive_embedding = nn.Parameter(torch.empty(num_nodes, adaptive_embedding_dim))
            nn.init.xavier_uniform_(self.adaptive_embedding)
        
        self.input_proj = nn.Conv2d(in_channels, hidden_dim, kernel_size=(1, 1))

        if if_time_in_day:
            self.tod_embedding = nn.Embedding(time_of_day_size, temp_dim_tid)
        if if_day_in_week:
            self.dow_embedding = nn.Embedding(day_of_week_size, temp_dim_diw)

        self.time_emb_dim = 0
        if if_time_in_day:
            self.time_emb_dim += temp_dim_tid
        if if_day_in_week:
            self.time_emb_dim += temp_dim_diw

        if if_forward_graph or if_backward_graph:
            self.graph_encoder = DualGraphEncoder(num_nodes, node_dim, dropout, if_forward_graph, if_backward_graph)

        self.temporal_encoder = MultiScaleTemporalEncoder(
            in_dim=hidden_dim,
            hidden_dim=hidden_dim,
            input_len=input_len,
            num_scales=num_scales,
            dropout=dropout
        )

        if if_adaptive_graph:
            self.adaptive_attentions = nn.ModuleList([
                AdaptiveGraphAttentionEncoder(
                    d_model=hidden_dim + self.adaptive_embedding_dim,
                    num_nodes=num_nodes,
                    out_dim=adp_dim,
                    num_layers=adaptive_layers,
                    nhead=adaptive_nhead,
                    dim_feedforward=4 * hidden_dim,
                    dropout=dropout,
                    node_dim=node_dim
                )
                for _ in range(num_scales)
            ])

        self.bottlenecks = nn.ModuleList([
            ST_Multi_Fusion(
                encoder_dim=hidden_dim,
                time_emb_dim=self.time_emb_dim,
                node_dim=node_dim,
                adaptive_dim=adp_dim,
                output_dim=skip_channels,
                num_layers=fusion_layers,
                dropout=dropout,
                first=(i == self.num_scales - 1),
                temporal_dim=hidden_dim,
                if_adaptive_graph=if_adaptive_graph,
                if_forward_graph=if_forward_graph,
                if_backward_graph=if_backward_graph
            )
            for i in range(self.num_scales)
        ])

        self.decoder = Decoder(
            input_dim=skip_channels * self.num_scales,
            hidden_dim=skip_channels * 2,
            out_dim=out_dim,
            num_layers=decoder_layers,
            dropout=dropout
        )

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def get_time_embedding(self, input_data):
        batch_size, input_steps, num_nodes, num_features = input_data.shape
        time_emb_list = []

        if self.if_time_in_day and num_features > 1:
            tod_idx = (input_data[:, -1, :, 1] * self.time_of_day_size).long().clamp(0, self.time_of_day_size - 1)
            tod_emb = self.tod_embedding(tod_idx)
            time_emb_list.append(tod_emb)

        if self.if_day_in_week and num_features > 2:
            dow_idx = (input_data[:, -1, :, 2] * self.day_of_week_size).long().clamp(0, self.day_of_week_size - 1)
            dow_emb = self.dow_embedding(dow_idx)
            time_emb_list.append(dow_emb)

        if time_emb_list:
            return torch.cat(time_emb_list, dim=-1)
        return torch.zeros(batch_size, num_nodes, self.time_emb_dim, device=input_data.device)

    def forward(self, history_data, future_data=None, batch_seen=None, epoch=None, train=True, **kwargs):
        batch_size = history_data.size(0)
        
        time_emb = self.get_time_embedding(history_data)
        
        if self.if_forward_graph or self.if_backward_graph:
            forward_graph_emb, backward_graph_emb = self.graph_encoder(self.supports, batch_size)
        else:
            forward_graph_emb, backward_graph_emb = None, None

        x = history_data[..., 0:1].permute(0, 3, 2, 1)
        
        x = self.input_proj(x)
        
        scale_features = self.temporal_encoder(x)
        
        if self.if_adp_emb:
            adp_emb = self.adaptive_embedding.unsqueeze(0).expand(batch_size, -1, -1)
        
        scale_outputs = [None] * self.num_scales
        pred_emb = []
        
        for i in range(self.num_scales - 1, -1, -1):
            temporal_feat = scale_features[i]  # [B, N, hidden_dim]
            
            if self.if_adp_emb:
                input_x = torch.cat([temporal_feat, adp_emb], dim=-1)  # [B, N, hidden_dim + adp_dim]
            else:
                input_x = temporal_feat  # [B, N, hidden_dim]
            
            if self.if_adaptive_graph:
                adaptive_graph_emb = self.adaptive_attentions[i](input_x)
            else:
                adaptive_graph_emb = None
            
            pred_emb_input = pred_emb if (self.if_multi_scale_anchor and len(pred_emb) > 0) else []
            
            predict = self.bottlenecks[i](
                time_emb, forward_graph_emb, backward_graph_emb,
                temporal_feat, adaptive_graph_emb, pred_emb_input
            )
            scale_outputs[i] = predict
            if self.if_multi_scale_anchor:
                pred_emb = [predict]
        fused = torch.cat(scale_outputs, dim=-1)
        output = self.decoder(fused)
        
        prediction = output.permute(0, 2, 1).unsqueeze(-1)

        return prediction
