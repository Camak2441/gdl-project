from models import get_model_name, load_model

EDGE_ENCODER = "all_minilm_l6v2"
NODE_ENCODER = "all_minilm_l6v2"
QUERY_ENCODER = "all_minilm_l6v2"

models = ["qigat", "qsgnn", "vngnn", "vngnn2", "qmpn"]

for model_shorthand in models:
    model_name = get_model_name(
        model_shorthand,
        node_encoder=NODE_ENCODER,
        edge_encoder=EDGE_ENCODER,
        query_encoder=QUERY_ENCODER,
        multi=False,
    )
    model = load_model(model_name)
    print(
        model_shorthand + ":", sum(p.numel() for p in model.parameters()), "parameters"
    )
