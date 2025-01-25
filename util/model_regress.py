

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATv2Conv
from torch.nn import Linear
from torch_geometric.nn.conv import HypergraphConv
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter

from torch_geometric.experimental import disable_dynamic_shapes
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import scatter, softmax



class HypergraphConv(MessagePassing):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_attention: bool = False,
        attention_mode: str = 'node',
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0,
        bias: bool = True,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(flow='source_to_target', node_dim=0, **kwargs)

        assert attention_mode in ['node', 'edge']

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_attention = use_attention
        self.attention_mode = attention_mode

        if self.use_attention:
            self.heads = heads
            self.concat = concat
            self.negative_slope = negative_slope
            self.dropout = dropout
            self.lin = Linear(in_channels, heads * out_channels, bias=False,
                              weight_initializer='glorot')
            self.att = Parameter(torch.empty(1, heads, 2 * out_channels))
        else:
            self.heads = 1
            self.concat = True
            self.lin = Linear(in_channels, out_channels, bias=False,
                              weight_initializer='glorot')

        if bias and concat:
            self.bias = Parameter(torch.empty(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()


    def reset_parameters(self):
        super().reset_parameters()
        self.lin.reset_parameters()
        if self.use_attention:
            glorot(self.att)
        zeros(self.bias)



    def forward(self, x: Tensor, hyperedge_index: Tensor,
                hyperedge_weight: Optional[Tensor] = None,
                hyperedge_attr: Optional[Tensor] = None,
                num_edges: Optional[int] = None) -> Tensor:
        r"""Runs the forward pass of the module.

        Args:
            x (torch.Tensor): Node feature matrix
                :math:`\mathbf{X} \in \mathbb{R}^{N \times F}`.
            hyperedge_index (torch.Tensor): The hyperedge indices, *i.e.*
                the sparse incidence matrix
                :math:`\mathbf{H} \in {\{ 0, 1 \}}^{N \times M}` mapping from
                nodes to edges.
            hyperedge_weight (torch.Tensor, optional): Hyperedge weights
                :math:`\mathbf{W} \in \mathbb{R}^M`. (default: :obj:`None`)
            hyperedge_attr (torch.Tensor, optional): Hyperedge feature matrix
                in :math:`\mathbb{R}^{M \times F}`.
                These features only need to get passed in case
                :obj:`use_attention=True`. (default: :obj:`None`)
            num_edges (int, optional) : The number of edges :math:`M`.
                (default: :obj:`None`)
        """
        num_nodes = x.size(0)

        if num_edges is None:
            num_edges = 0
            if hyperedge_index.numel() > 0:
                num_edges = int(hyperedge_index[1].max()) + 1

        if hyperedge_weight is None:
            hyperedge_weight = x.new_ones(num_edges)

        x = self.lin(x)

        alpha = None
        if self.use_attention:
            assert hyperedge_attr is not None
            x = x.view(-1, self.heads, self.out_channels)
            hyperedge_attr = self.lin(hyperedge_attr)
            hyperedge_attr = hyperedge_attr.view(-1, self.heads,
                                                 self.out_channels)
            x_i = x[hyperedge_index[0]]
            x_j = hyperedge_attr[hyperedge_index[1]]
            alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
            alpha = F.leaky_relu(alpha, self.negative_slope)
            if self.attention_mode == 'node':
                alpha = softmax(alpha, hyperedge_index[1], num_nodes=num_edges)
            else:
                alpha = softmax(alpha, hyperedge_index[0], num_nodes=num_nodes)
            alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        D = scatter(hyperedge_weight[hyperedge_index[1]], hyperedge_index[0],
                    dim=0, dim_size=num_nodes, reduce='sum')
        D = 1.0 / D
        D[D == float("inf")] = 0

        B = scatter(x.new_ones(hyperedge_index.size(1)), hyperedge_index[1],
                    dim=0, dim_size=num_edges, reduce='sum')
        B = 1.0 / B
        B[B == float("inf")] = 0

        out = self.propagate(hyperedge_index, x=x, norm=B, alpha=alpha,
                             size=(num_nodes, num_edges))
        out = self.propagate(hyperedge_index.flip([0]), x=out, norm=D,
                             alpha=alpha, size=(num_edges, num_nodes))

        if self.concat is True:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias

        return out


    def message(self, x_j: Tensor, norm_i: Tensor, alpha: Tensor) -> Tensor:
        H, F = self.heads, self.out_channels

        out = norm_i.view(-1, 1, 1) * x_j.view(-1, H, F)

        if alpha is not None:
            out = alpha.view(-1, self.heads, 1) * out

        return out

import torch
import torch.nn as nn
import torch.nn.functional as F

class Net_encoder(nn.Module):
    def __init__(self, input_size):
        super(Net_encoder, self).__init__()
        self.input_size = input_size

        self.encoder = nn.Sequential(
            nn.Linear(self.input_size, 64),
            # nn.ReLU(),
            # nn.LayerNorm(64)
        )

    def forward(self, data):
        data = data.float().view(-1, self.input_size)
        embedding = self.encoder(data)
        return embedding

class DualAttention(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super(DualAttention, self).__init__()
        self.query_layer = nn.Linear(in_dim, hidden_dim)
        self.key_layer = nn.Linear(in_dim, hidden_dim)
        self.value_layer = nn.Linear(in_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, in_dim)

    def forward(self, x1, x2):
        # 
        Q1 = self.query_layer(x1)
        K1 = self.key_layer(x1)
        V1 = self.value_layer(x1)

        attention_weights1 = F.softmax(torch.bmm(Q1, K1.transpose(1, 2)) / (K1.size(-1) ** 0.5), dim=-1)
        self_attention_out = torch.bmm(attention_weights1, V1)

        # 
        Q2 = self.query_layer(x2)
        K2 = self.key_layer(x2)
        V2 = self.value_layer(x2)

        attention_weights2 = F.softmax(torch.bmm(Q2, K2.transpose(1, 2)) / (K2.size(-1) ** 0.5), dim=-1)
        cross_attention_out = torch.bmm(attention_weights2, V2)

        # 
        out = self_attention_out + cross_attention_out
        out = self.fc_out(out)

        return out

class Net_cell(nn.Module):
    def __init__(self, num_of_class):
        super(Net_cell, self).__init__()
        self.cell = nn.Sequential(
            nn.Linear(64, num_of_class)
        )

        self.gnn = GNN(num_of_class)

        # 
        self.attention = DualAttention(in_dim=64, hidden_dim=512)

    def forward(self, embedding, edge_index):
        cell_prediction = self.cell(embedding)
        embedding = embedding.unsqueeze(1)

        # 
        hyperedge_attr = torch.randn(embedding.size(0), 64).to(embedding.device)
        hyperedge_attr = hyperedge_attr.unsqueeze(1)        
        attention_output = self.attention(embedding, hyperedge_attr)

        # 
        cell_prediction = self.gnn(attention_output, edge_index, hyperedge_attr)
        return cell_prediction

class GNN(nn.Module):
    def __init__(self, num_of_class):
        super(GNN, self).__init__()

        self.hconv = HypergraphConv(64, 1, use_attention=True)
        self.num_heads =1
        self.head_convs = nn.ModuleList([HypergraphConv(1,1) for _ in range(self.num_heads)])
        self.out = nn.Linear(1* self.num_heads, num_of_class)


    def forward(self, x, edge_index, hyperedge_attr):
        h = self.hconv(x, edge_index, hyperedge_attr=hyperedge_attr)
        out = torch.cat([conv(h, edge_index, hyperedge_attr=hyperedge_attr) for conv in self.head_convs], dim=1)
        embedding = torch.relu(out)
        z = self.out(embedding)
        return z
 
