from torch_geometric.data import Data


class QueryData(Data):
    def __init__(self, x, pos, edge_index, edge_attr, query, y=None, qtype=None):
        super().__init__(x=x, pos=pos, edge_index=edge_index, edge_attr=edge_attr, y=y)
        self.query = query
        self.qtype = qtype

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key == "query":
            return None
        if key == "qtype":
            return 0
        return super().__cat_dim__(key, value, *args, **kwargs)
