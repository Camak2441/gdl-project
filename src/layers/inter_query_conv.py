import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax


class InterQueryConv(MessagePassing):
    """
    Inter-level query-guided convolution from QSGNN (arXiv:2510.11541).
    """

    def __init__(
        self,
        in_dim: int,
        query_dim: int,
        out_dim: int,
        heads: int = 1,
        dropout: float = 0.0,
        concat: bool = True,
    ):
        super().__init__(aggr="add", node_dim=0)

        self.in_dim = in_dim
        self.query_dim = query_dim
        self.out_dim = out_dim
        self.heads = heads
        self.concat = concat

        proj_dim = out_dim

        # Node projections for the cross-level pair embedding
        self.W_t = nn.Linear(in_dim, proj_dim, bias=False)  # target node
        self.W_s = nn.Linear(in_dim, proj_dim, bias=False)  # source node

        # gamma projections: query-to-pair cosine similarity (shared across heads)
        self.W_gamma_q = nn.Linear(query_dim, proj_dim, bias=False)
        self.W_gamma_k = nn.Linear(2 * proj_dim, proj_dim, bias=False)

        # Value projection
        self.W_v = nn.Linear(in_dim, heads * out_dim, bias=False)

        out_channels = heads * out_dim if concat else out_dim
        self.bias = nn.Parameter(torch.zeros(out_channels))

        self.dropout = nn.Dropout(p=dropout)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_t.weight)
        nn.init.xavier_uniform_(self.W_s.weight)
        nn.init.xavier_uniform_(self.W_gamma_q.weight)
        nn.init.xavier_uniform_(self.W_gamma_k.weight)
        nn.init.xavier_uniform_(self.W_v.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x, edge_index, edge_attr, query, batch):
        """
        :param x: Node features [num_nodes, in_dim].
        :param edge_index: Edge indices [2, num_edges].
        :param edge_attr: Edge attributes (unused; kept for API compatibility).
        :param query: Query embeddings [num_graphs, query_dim].
        :param batch: Batch indices [num_nodes].
        :returns: Updated node features [num_nodes, heads*out_dim] or [num_nodes, out_dim].
        """
        q = query[batch]  # [num_nodes, query_dim]

        x_t = self.W_t(x)  # [N, proj_dim]
        x_s = self.W_s(x)  # [N, proj_dim]
        q_gamma = self.W_gamma_q(q)  # [N, proj_dim]
        x_v = self.W_v(x).view(-1, self.heads, self.out_dim)  # [N, H, D]

        out = self.propagate(
            edge_index,
            x_t=x_t,
            x_s=x_s,
            q_gamma=q_gamma,
            x_v=x_v,
        )  # [N, H, D]

        if self.concat:
            out = out.view(-1, self.heads * self.out_dim)
        else:
            out = out.mean(dim=1)

        return out + self.bias

    def message(self, x_t_i, x_s_j, q_gamma_i, x_v_j, index):
        # p_ij: cross-level pair embedding [E, 2*proj_dim]
        p = torch.cat([x_t_i, x_s_j], dim=-1)

        # gamma: query-to-pair cosine similarity, broadcast across heads [E, 1]
        gamma = F.cosine_similarity(q_gamma_i, self.W_gamma_k(p), dim=-1)  # [E]
        gamma = gamma.unsqueeze(-1)  # [E, 1]

        attn = softmax(gamma, index)  # [E, 1]
        attn = self.dropout(attn)

        return attn.unsqueeze(-1) * x_v_j  # [E, H, D]  (attn [E,1,1] broadcasts)
