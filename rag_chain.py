def ask_question(question, retriever, llm_fn):

    docs = retriever.invoke(question)

    if not docs:
        return "Not found in knowledge base.", 0.0

    context = ""
    sources = []

    for doc in docs:
        text = doc.page_content.replace("\n", " ")
        text = " ".join(text.split())  # clean spacing

        context += f"{text}\n"
        sources.append(text[:150])

    prompt = f"""
You are a STRICT home remedy assistant.

RULES:
- Use ONLY the given context.
- Answer in 5-7 short lines only.
- First line = remedy
- Second line = usage
- Third line = safety note (if any)
- Do NOT add extra information.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

    answer = llm_fn(prompt).strip()

    return answer, sources