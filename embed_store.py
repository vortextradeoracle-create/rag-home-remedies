from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os

model = SentenceTransformer("all-MiniLM-L6-v2")

def load_docs(folder="data"):
    docs = []
    for file in os.listdir(folder):
        with open(os.path.join(folder, file), "r", encoding="utf-8") as f:
            docs.append(f.read())
    return docs


def create_vector_store():
    docs = load_docs()

    embeddings = model.encode(docs)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))

    return index, docs


if __name__ == "__main__":
    index, docs = create_vector_store()
    print("Vector DB created with docs:", len(docs))