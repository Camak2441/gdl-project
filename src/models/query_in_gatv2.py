import torch
import torch_geometric
from torch_geometric.nn.conv import GATv2Conv
from layers import QueryGATv2Conv


class QueryInGATv2(torch.nn.Module):

    def __init__(self, in_dim, query_dim, hidden_dims, out_dim, edge_dim, heads=1):
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
                    QueryGATv2Conv(
                        in_dim=l_in_dims[i],
                        query_dim=query_dim,
                        out_dim=l_out_dims[i],
                        edge_dim=edge_dim,
                        heads=heads,
                    )
                )
            elif i + 1 < len(l_in_dims):
                layers.append(
                    GATv2Conv(
                        in_channels=l_in_dims[i] * heads,
                        out_channels=l_out_dims[i],
                        edge_dim=edge_dim,
                        heads=heads,
                    )
                )
            else:
                layers.append(
                    GATv2Conv(
                        in_channels=l_in_dims[i] * heads,
                        out_channels=l_out_dims[i],
                        edge_dim=edge_dim,
                        heads=1,
                    )
                )

        self.layers = torch.nn.ModuleList(layers)

        self.relu = torch.nn.ReLU()

    def forward(self, x, edge_index, edge_attr, query, batch):
        x = self.layers[0](
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            query=query,
            batch=batch,
        )
        x = self.relu(x)
        for id, layer in enumerate(self.layers[1:]):
            x = layer(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
            )
            if id + 2 == len(self.layers):
                x = torch_geometric.utils.softmax(x.squeeze(), batch)
            else:
                x = self.relu(x)
        return x
