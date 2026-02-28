from typing import List

import torch
import torch_geometric
from torch_geometric.nn.conv import GATConv


class Gat(torch.nn.Module):

    def __init__(
        self,
        in_dim: int,
        edge_dim: int,
        hidden_dims: List[int],
        out_dim: int,
        heads=1,
        multi: bool = False,
        out_act: bool = True,
    ):
        super().__init__()

        self.in_channels = in_dim
        self.hidden_channels = hidden_dims
        self.out_channels = out_dim
        layers = []

        l_in_dims = [in_dim] + hidden_dims
        l_out_dims = hidden_dims + [out_dim]

        for i in range(len(l_in_dims)):
            if i == 0:
                layers.append(
                    GATConv(
                        in_channels=l_in_dims[i],
                        out_channels=l_out_dims[i],
                        edge_dim=edge_dim,
                        heads=heads,
                    )
                )
            elif i + 1 < len(l_in_dims):
                layers.append(
                    GATConv(
                        in_channels=l_in_dims[i] * heads,
                        out_channels=l_out_dims[i],
                        edge_dim=edge_dim,
                        heads=heads,
                    )
                )
            else:
                layers.append(
                    GATConv(
                        in_channels=l_in_dims[i] * heads,
                        out_channels=l_out_dims[i],
                        edge_dim=edge_dim,
                        heads=1,
                    )
                )

        self.layers = torch.nn.ModuleList(layers)

        self.out_act = out_act
        if out_act:
            self.multi = multi
            if multi:
                self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()

    def forward(self, x, edge_index, edge_attr, batch):
        for id, layer in enumerate(self.layers):
            x = layer(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
            )
            if id + 1 == len(self.layers):
                if self.out_act:
                    if self.multi:
                        x = self.sigmoid(x).squeeze(-1)
                    else:
                        x = torch_geometric.utils.softmax(x.squeeze(-1), batch)
            else:
                x = self.relu(x)
        return x
