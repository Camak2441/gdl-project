from typing import List

import torch
import torch.nn.functional as F
import torch_geometric
from layers import IntraQueryGATConv, InterQueryConv


class QsGnn(torch.nn.Module):
    """
    :param in_dim: Input node feature dimension.
    :param query_dim: Query embedding dimension.
    :param hidden_dims: List of per-head hidden dimensions; each entry produces one intra+inter block.
    :param out_dim: Output dimension (typically 1 for node scoring).
    :param edge_dim: Edge feature dimension.
    :param heads: Number of attention heads for intermediate layers.
    """

    def __init__(
        self,
        in_dim: int,
        query_dim: int,
        edge_dim: int,
        hidden_dims: List[int],
        heads: int = 1,
        multi: bool = False,
    ):
        super().__init__()

        self.in_channels = in_dim
        self.hidden_channels = hidden_dims
        self.out_channels = query_dim

        l_in_dims = [in_dim] + hidden_dims
        l_out_dims = hidden_dims + [query_dim]

        intra_layers = []
        inter_layers = []

        for i in range(len(l_in_dims)):
            is_last = i + 1 == len(l_in_dims)
            in_d = l_in_dims[i] if i == 0 else l_in_dims[i] * heads
            out_d = l_out_dims[i]
            h = 1 if is_last else heads

            intra_layers.append(
                IntraQueryGATConv(
                    in_dim=in_d,
                    edge_dim=edge_dim,
                    query_dim=query_dim,
                    out_dim=out_d,
                    heads=h,
                    concat=not is_last,
                )
            )

            if not is_last:
                inter_layers.append(
                    InterQueryConv(
                        in_dim=out_d * h,
                        query_dim=query_dim,
                        out_dim=out_d,
                        heads=h,
                        concat=True,
                    )
                )

        self.intra_layers = torch.nn.ModuleList(intra_layers)
        self.inter_layers = torch.nn.ModuleList(inter_layers)
        self.multi = multi
        if multi:
            self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()

    def forward(self, x, edge_index, edge_attr, query, batch):
        for i, intra in enumerate(self.intra_layers):
            x = intra(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                query=query,
                batch=batch,
            )

            if i + 1 == len(self.intra_layers):
                x = F.cosine_similarity(x, query[batch])
                if self.multi:
                    x = self.sigmoid(x).squeeze(-1)
                else:
                    x = torch_geometric.utils.softmax(x.squeeze(-1), batch)
            else:
                x = self.relu(x)
                x = self.inter_layers[i](
                    x=x,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    query=query,
                    batch=batch,
                )
                x = self.relu(x)

        return x
