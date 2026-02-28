from typing import List

import torch
import torch_geometric

from layers.query_message_passing import QueryMessagePassing


class QueryMpn(torch.nn.Module):

    def __init__(
        self,
        in_dim: int,
        edge_dim: int,
        query_dim: int,
        hidden_dims: List[int],
        out_dim: int,
        multi: bool = False,
    ):
        super().__init__()

        self.in_channels = in_dim
        self.hidden_channels = hidden_dims
        self.out_channels = out_dim
        layers = []

        l_in_dims = [in_dim] + hidden_dims
        l_out_dims = hidden_dims + [out_dim]

        for i in range(len(l_in_dims)):
            layers.append(
                QueryMessagePassing(
                    in_dim=l_in_dims[i],
                    edge_dim=edge_dim,
                    query_dim=query_dim,
                    out_dim=l_out_dims[i],
                )
            )

        self.layers = torch.nn.ModuleList(layers)

        self.multi = multi
        if multi:
            self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()

    def forward(self, x, edge_index, edge_attr, query, batch):
        for id, layer in enumerate(self.layers):
            x = layer(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                query=query[batch],
            )
            if id + 1 == len(self.layers):
                if self.multi:
                    x = self.sigmoid(x).squeeze(-1)
                else:
                    x = torch_geometric.utils.softmax(x.squeeze(-1), batch)
            else:
                x = self.relu(x)
        return x
