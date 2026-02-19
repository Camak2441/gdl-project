from sentence_transformers import SentenceTransformer


SENTENCE_TRANSFORMERS = {}


VALID_SENTENCE_TRANSFORMERS = {"all-MiniLM-L6-v2"}


def load_sentence_transformer(name, device):
    if name + device in SENTENCE_TRANSFORMERS:
        return SENTENCE_TRANSFORMERS[name + device]
    elif name in VALID_SENTENCE_TRANSFORMERS:
        SENTENCE_TRANSFORMERS[name + device] = SentenceTransformer(
            "all-MiniLM-L6-v2", device=device
        )
        return SENTENCE_TRANSFORMERS[name + device]
    raise Exception(f"Unknown sentence transfomer {name}")
