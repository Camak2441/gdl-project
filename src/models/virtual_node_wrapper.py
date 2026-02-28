import torch
import torch_geometric
from layers import AddQueryVirtualNode


class VirtualNodeWrapper(torch.nn.Module):
    @staticmethod
    def get_args(
        model,
        in_dim: int,
        query_dim: int,
        edge_dim: int,
        query_mlp: bool = False,
        multi: bool = False,
    ):
        return {
            "in_dim": in_dim,
            "edge_dim": edge_dim,
            "multi": multi,
            "out_act": False,
        }

    def __init__(
        self,
        model,
        in_dim: int,
        query_dim: int,
        edge_dim: int,
        query_mlp: bool = False,
        multi: bool = False,
    ):
        """
        Wraps a model by prepending a virtual node layer that injects the query
        into the graph as a virtual node, then strips it from the output.

        :param model: the inner model with signature forward(x, edge_index, edge_attr, query, batch)
        :param query_dim: dimensionality of the query embedding
        :param node_dim: dimensionality of the node features
        """
        super().__init__()
        self.model = model
        self.virtual_node = AddQueryVirtualNode(
            query_dim=query_dim, in_dim=in_dim, query_mlp=query_mlp
        )

        self.multi = multi
        if multi:
            self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x, edge_index, edge_attr, query, batch):
        """
        :param x: node features [num_nodes, node_dim]
        :param edge_index: edge indices [2, num_edges]
        :param edge_attr: edge features [num_edges, edge_dim] or None
        :param query: query embeddings [num_graphs, query_dim]
        :param batch: batch indices [num_nodes]
        """
        num_nodes = x.size(0)

        x_aug, edge_index_aug, batch_aug = self.virtual_node(
            x, edge_index, query, batch
        )

        # Pad edge_attr with zeros for the 2 * num_nodes new virtual node edges
        if edge_attr is not None:
            vn_edge_attr = torch.zeros(
                2 * num_nodes,
                edge_attr.size(-1),
                device=edge_attr.device,
                dtype=edge_attr.dtype,
            )
            edge_attr_aug = torch.cat([edge_attr, vn_edge_attr], dim=0)
        else:
            edge_attr_aug = None

        out = self.model(x_aug, edge_index_aug, edge_attr_aug, batch_aug)

        # Strip virtual node outputs (appended at the end, one per graph)
        out = out[:num_nodes]

        if self.multi:
            out = self.sigmoid(out).squeeze(-1)
        else:
            out = torch_geometric.utils.softmax(out.squeeze(-1), batch)

        return out
