import torch_geometric
import torch
import torch.nn as nn
import torch.nn.functional as F


class QueryMessagePassing(torch_geometric.nn.MessagePassing):

    def __init__(
        self,
        in_dim: int,
        edge_dim: int,
        query_dim: int,
        out_dim: int,
        heads: int = 1,
    ):
        super().__init__(aggr="add", node_dim=0)

        self.in_dim = in_dim
        self.query_dim = query_dim
        self.out_dim = out_dim
        self.heads = heads

        self.query_proj_mlp = None
        if query_dim != in_dim:
            self.query_proj_mlp = nn.Sequential(
                nn.Linear(query_dim, in_dim),
                nn.BatchNorm1d(in_dim),
                nn.ReLU(),
                nn.Linear(in_dim, in_dim),
                nn.BatchNorm1d(in_dim),
            )
        self.message_mlp = nn.Sequential(
            nn.Linear(in_dim * 2 + edge_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
        )

        self.update_mlp = nn.Sequential(
            nn.Linear(in_dim + out_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
            nn.BatchNorm1d(out_dim),
        )

    def forward(self, x, edge_index, edge_attr, query):
        """
        :param x: Node features [num_nodes, in_dim].
        :param edge_index: Edge indices [2, num_edges].
        :param edge_attr: Edge attributes [num_edges, edge_dim]
        :param query: Query embeddings [num_graphs, query_dim].
        :returns: Updated node features [num_nodes, out_dim].
        """

        if self.query_proj_mlp is not None:
            query = self.query_proj_mlp(query)

        out = self.propagate(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            query=query,
        )

        return out

    def message(self, x_i, x_j, edge_attr, query_i):
        return self.message_mlp(
            torch.cat([x_i * query_i, x_j * query_i, edge_attr], dim=-1)
        )

    def update(self, aggr_out, x):
        return self.update_mlp(torch.cat([x, aggr_out], dim=-1))
