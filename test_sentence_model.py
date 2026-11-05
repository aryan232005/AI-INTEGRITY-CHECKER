import socket

_original = socket.getaddrinfo

def ipv4_only(host, port, family=0, type=0, proto=0, flags=0):
    return _original(host, port, socket.AF_INET, type, proto, flags)

socket.getaddrinfo = ipv4_only

from sentence_transformers import SentenceTransformer

print("Downloading/loading Sentence Transformer model...")

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device="cpu"
)

embeddings = model.encode(["This is a test of AI Integrity Checker."])

print("MODEL LOADED SUCCESSFULLY")
print("Embedding shape:", embeddings.shape)
