from sentence_transformers import SentenceTransformer

from consts import DEVICE


ALL_MINILM_L6_V2 = SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE)
