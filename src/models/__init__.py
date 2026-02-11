from models.query_gat import QueryGAT


def load_model(in_dim, query_dim, edge_dim, out_dim, model_name: str):
    if model_name.startswith("query_gat_"):
        layer_dims = [int(dim) for dim in model_name[len("query_gat_") :].split(",")]
        return QueryGAT(
            in_dim=in_dim,
            query_dim=query_dim,
            edge_dim=edge_dim,
            hidden_dims=layer_dims,
            out_dim=out_dim,
        )
