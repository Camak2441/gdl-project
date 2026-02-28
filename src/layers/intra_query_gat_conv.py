import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax


class IntraQueryGATConv(MessagePassing):
    """
    Intra-level query-guided GAT convolution from QSGNN (arXiv:2510.11541).
    """

    def __init__(
        self,
        in_dim: int,
        edge_dim: int,
        query_dim: int,
        out_dim: int,
        heads: int = 1,
        dropout: float = 0.0,
        concat: bool = True,
    ):
        super().__init__(aggr="add", node_dim=0)

        self.in_dim = in_dim
        self.edge_dim = edge_dim
        self.query_dim = query_dim
        self.out_dim = out_dim
        self.heads = heads
        self.concat = concat

        # alpha projections: per-head node-to-node cosine similarity
        self.W_alpha_q = nn.Linear(in_dim, heads * out_dim, bias=False)
        self.W_alpha_k = nn.Linear(in_dim, heads * out_dim, bias=False)

        # beta projections: query-to-edge cosine similarity (shared across heads)
        self.W_beta_q = nn.Linear(query_dim, out_dim, bias=False)
        self.W_beta_k = nn.Linear(2 * in_dim + edge_dim, out_dim, bias=False)

        # Value projection
        self.W_v = nn.Linear(in_dim, heads * out_dim, bias=False)

        out_channels = heads * out_dim if concat else out_dim
        self.bias = nn.Parameter(torch.zeros(out_channels))

        self.dropout = nn.Dropout(p=dropout)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_alpha_q.weight)
        nn.init.xavier_uniform_(self.W_alpha_k.weight)
        nn.init.xavier_uniform_(self.W_beta_q.weight)
        nn.init.xavier_uniform_(self.W_beta_k.weight)
        nn.init.xavier_uniform_(self.W_v.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x, edge_index, edge_attr, query, batch):
        """
        :param x: Node features [num_nodes, in_dim].
        :param edge_index: Edge indices [2, num_edges].
        :param edge_attr: Edge attributes [num_edges, edge_dim].
        :param query: Query embeddings [num_graphs, query_dim].
        :param batch: Batch indices [num_nodes].
        :returns: Updated node features [num_nodes, heads*out_dim] or [num_nodes, out_dim].
        """
        q = query[batch]  # [num_nodes, query_dim]

        x_aq = self.W_alpha_q(x).view(-1, self.heads, self.out_dim)  # [N, H, D]
        x_ak = self.W_alpha_k(x).view(-1, self.heads, self.out_dim)  # [N, H, D]
        q_beta = self.W_beta_q(q)  # [N, D]
        x_v = self.W_v(x).view(-1, self.heads, self.out_dim)  # [N, H, D]

        out = self.propagate(
            edge_index,
            x_aq=x_aq,
            x_ak=x_ak,
            q_beta=q_beta,
            x_v=x_v,
            x=x,
            edge_attr=edge_attr,
        )  # [N, H, D]

        if self.concat:
            out = out.view(-1, self.heads * self.out_dim)
        else:
            out = out.mean(dim=1)

        return out + self.bias

    def message(self, x_aq_i, x_ak_j, q_beta_i, x_v_j, x_i, x_j, edge_attr, index):
        # alpha: per-head node-to-node cosine similarity [E, H]
        alpha = F.cosine_similarity(x_aq_i, x_ak_j, dim=-1)  # [E, H]

        # beta: query-to-edge cosine similarity, broadcast across heads [E, 1]
        edge_ctx = torch.cat([x_i, x_j, edge_attr], dim=-1)  # [E, 2*in_dim + edge_dim]
        beta = F.cosine_similarity(q_beta_i, self.W_beta_k(edge_ctx), dim=-1)  # [E]
        beta = beta.unsqueeze(-1)  # [E, 1]

        # Normalised dual attention
        attn = softmax(alpha + beta, index)  # [E, H]
        attn = self.dropout(attn)

        return attn.unsqueeze(-1) * x_v_j  # [E, H, D]
