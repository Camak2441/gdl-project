import torch
import torch.nn as nn


class AddQueryVirtualNode(torch.nn.Module):
    def __init__(self, query_dim, in_dim, query_mlp: bool = True):
        """
        Adds the query as a virtual node connected to all nodes in the graph.

        :param query_dim: dimensionality of the query embedding
        :param node_dim: dimensionality of the node features
        """
        super().__init__()
        self.query_dim = query_dim
        self.node_dim = in_dim

        if query_dim != in_dim or query_mlp:
            self.proj = nn.Sequential(
                nn.Linear(query_dim, in_dim),
                nn.BatchNorm1d(in_dim),
                nn.ReLU(),
                nn.Linear(in_dim, in_dim),
                nn.BatchNorm1d(in_dim),
            )
        else:
            self.proj = None

    def forward(self, x, edge_index, query, batch):
        """
        :param x: node features [num_nodes, node_dim]
        :param edge_index: edge indices [2, num_edges]
        :param query: query embeddings [num_graphs, query_dim]
        :param batch: batch indices [num_nodes]
        :return: (x_aug, edge_index_aug, batch_aug) with one virtual node per graph
        """
        num_nodes = x.size(0)
        num_graphs = query.size(0)

        vn_features = (
            self.proj(query) if self.proj is not None else query
        )  # [num_graphs, node_dim]

        x_aug = torch.cat([x, vn_features], dim=0)  # [num_nodes + num_graphs, node_dim]

        # Virtual node index for each real node: num_nodes + graph_idx
        vn_idx = num_nodes + batch  # [num_nodes]
        real_idx = torch.arange(num_nodes, device=x.device)  # [num_nodes]

        vn_to_real = torch.stack([vn_idx, real_idx], dim=0)  # [2, num_nodes]
        real_to_vn = torch.stack([real_idx, vn_idx], dim=0)  # [2, num_nodes]

        edge_index_aug = torch.cat([edge_index, vn_to_real, real_to_vn], dim=1)

        vn_batch = torch.arange(num_graphs, device=x.device)
        batch_aug = torch.cat([batch, vn_batch], dim=0)

        return x_aug, edge_index_aug, batch_aug
