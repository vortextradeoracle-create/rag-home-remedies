from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer(
"sentence-transformers/all-MiniLM-L6-v2"
)

def compute_retrieval_score(query, docs):

```
if not docs:
    return 0.0

query_embedding = model.encode([query])

scores = []

for doc in docs:

    doc_embedding = model.encode(
        [doc.page_content[:1000]]
    )

    similarity = cosine_similarity(
        query_embedding,
        doc_embedding
    )[0][0]

    scores.append(similarity)

return round(sum(scores) / len(scores), 3)
```
