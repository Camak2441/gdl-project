from typing import List

import torch
import torch_geometric
from layers import IntraQueryGATConv


class QsGat(torch.nn.Module):
    """
    :param in_dim: Input node feature dimension.
    :param query_dim: Query embedding dimension.
    :param hidden_dims: List of per-head hidden dimensions for intermediate layers.
    :param out_dim: Output dimension (typically 1 for node scoring).
    :param edge_dim: Edge feature dimension.
    :param heads: Number of attention heads.
    """

    def __init__(
        self,
        in_dim: int,
        query_dim: int,
        edge_dim: int,
        hidden_dims: List[int],
        out_dim: int,
        heads: int = 1,
        multi: bool = False,
    ):
        super().__init__()

        self.in_channels = in_dim
        self.hidden_channels = hidden_dims
        self.out_channels = out_dim

        l_in_dims = [in_dim] + hidden_dims
        l_out_dims = hidden_dims + [out_dim]
        layers = []

        for i in range(len(l_in_dims)):
            is_last = i + 1 == len(l_in_dims)
            layers.append(
                IntraQueryGATConv(
                    in_dim=l_in_dims[i] if i == 0 else l_in_dims[i] * heads,
                    edge_dim=edge_dim,
                    query_dim=query_dim,
                    out_dim=l_out_dims[i],
                    heads=1 if is_last else heads,
                    concat=not is_last,
                )
            )

        self.layers = torch.nn.ModuleList(layers)
        self.multi = multi
        if multi:
            self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()

    def forward(self, x, edge_index, edge_attr, query, batch):
        for i, layer in enumerate(self.layers):
            x = layer(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                query=query,
                batch=batch,
            )
            if i + 1 == len(self.layers):
                if self.multi:
                    x = self.sigmoid(x).squeeze(-1)
                else:
                    x = torch_geometric.utils.softmax(x.squeeze(-1), batch)
            else:
                x = self.relu(x)
        return x
