import torch
import torch_geometric
from layers import QueryGATConv


class QueryGAT(torch.nn.Module):

    def __init__(self, in_dim, query_dim, hidden_dims, out_dim, edge_dim, heads=1):
        super().__init__()

        self.in_channels = in_dim
        self.hidden_channels = hidden_dims
        self.out_channels = out_dim
        layers = []

        l_in_dims = [in_dim] + hidden_dims
        l_out_dims = hidden_dims + [out_dim]

        for i in range(len(l_in_dims)):
            layers.append(
                QueryGATConv(
                    in_dim=l_in_dims[i],
                    query_dim=query_dim,
                    out_dim=l_out_dims[i],
                    edge_dim=edge_dim,
                    heads=heads,
                )
            )

        self.layers = torch.nn.ModuleList(layers)

        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=0)

    def forward(self, x, edge_index, edge_attr, query, batch):
        for id, layer in enumerate(self.layers):
            x = layer(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                query=query,
                batch=batch,
            )
            if id + 1 == len(self.layers):
                x = torch_geometric.utils.softmax(x.squeeze(), batch)
            else:
                x = self.relu(x)
        return x
