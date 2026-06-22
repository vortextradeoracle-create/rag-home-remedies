def compute_retrieval_score(query, docs):

    if not docs:
        return 0.0

    query_tokens = set(query.lower().split())
    score = 0

    for doc in docs:
        doc_tokens = set(doc.page_content.lower().split())
        overlap = len(query_tokens & doc_tokens)
        score += overlap / (len(query_tokens) + 1)

    return round(score / len(docs), 3)