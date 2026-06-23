from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

print("Loading grounding model...")

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

print("Grounding model ready!")


def compute_grounding(answer, docs):

    if not docs:
        return 0.0

    answer_embedding = embedding_model.encode([answer])

    scores = []

    for doc in docs:

        doc_embedding = embedding_model.encode(
            [doc.page_content[:1000]]
        )

        sim = cosine_similarity(
            answer_embedding,
            doc_embedding
        )[0][0]

        scores.append(sim)

    # STEP 1: sort best matches first
    scores.sort(reverse=True)

    # STEP 2: only take TOP 3 chunks (IMPORTANT FIX)
    top_scores = scores[:3]

    # STEP 3: weighted average (focus on best evidence)
    score = sum(top_scores) / len(top_scores)

    return round(float(score), 2)


def ask_question(question, retriever, llm_fn):

    if hasattr(retriever, "invoke"):
        docs = retriever.invoke(question)
    else:
        docs = retriever.get_relevant_documents(question)

    if not docs:
        return "❌ No information found."

    context = ""

    for i, doc in enumerate(docs):
        context += f"Source {i+1}:\n{doc.page_content}\n\n"

    prompt = f"""
You are a medical home remedy assistant.

Answer ONLY using the provided information.

Do not use outside knowledge.

If the answer is not present in the provided sources, reply exactly:

"I could not find that information in the knowledge base."

Context:

{context}

Question:
{question}

Answer:
"""

    answer = llm_fn(prompt)

    grounding_score = compute_grounding(answer, docs)

    if grounding_score >= 0.70:
        grounding_label = "✔ Well Grounded"
    elif grounding_score >= 0.50:
        grounding_label = "⚠ Partially Grounded"
    else:
        grounding_label = "❌ Weak Grounding"

    citations = []

    for i, doc in enumerate(docs):

        remedy = doc.metadata.get("remedy_name", "Unknown")
        source_file = doc.metadata.get("source_file", "unknown")
        chunk_id = doc.metadata.get("chunk_id", "-")

        citations.append(
            f"[{i+1}] {remedy} | {source_file} | chunk={chunk_id}"
        )

    sources_text = "\n".join(citations)

    return f"""
{answer}

---

📌 Sources Used:

{sources_text}

📊 Retrieved Chunks: {len(docs)}

🧠 Grounding Score: {grounding_score}
{grounding_label}
"""