import torch
from torch_geometric.nn.conv import GATConv


class QueryGATConv(torch.nn.Module):
    def __init__(
        self,
        in_dim,
        query_dim,
        out_dim,
        edge_dim=None,
        heads=1,
        concat=True,
    ):
        super().__init__()
        self.gat_conv: GATConv = GATConv(
            in_channels=in_dim + query_dim,
            out_channels=out_dim,
            edge_dim=edge_dim,
            heads=heads,
            concat=concat,
        )
        self.in_dim = in_dim
        self.query_dim = query_dim
        self.out_dim = out_dim
        self.edge_dim = edge_dim

    def forward(self, x, edge_index, edge_attr, query, batch):
        """
        Docstring for forward

        :param self: Description
        :param x: node features in shape [num_nodes, num_node_features]
        :param edge_index: Description
        :param edge_attr: Description
        :param query: Description
        :param batch: batch indices in shape [num_nodes]
        """
        x = torch.cat(
            [
                x,
                query[batch],
            ],
            dim=1,
        )
        return self.gat_conv(x=x, edge_index=edge_index, edge_attr=edge_attr)
