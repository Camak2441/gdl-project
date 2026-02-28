import torch
from torch_geometric.nn.conv import GATv2Conv


class QueryGATv2Conv(torch.nn.Module):
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
        self.gat_conv: GATv2Conv = GATv2Conv(
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
        :param x: Node features [num_nodes, in_dim]
        :param edge_index: Edge index [2, num_edges]
        :param edge_attr: Edge features [num_edge, edge_dim]
        :param query: Query [num graphs, query_dim]
        :param batch: Batch indices [num_nodes]
        :returns: Updated node features [num_nodes, out_dim].
        """
        x = torch.cat(
            [
                x,
                query[batch],
            ],
            dim=1,
        )
        return self.gat_conv(x=x, edge_index=edge_index, edge_attr=edge_attr)
