import gradio as gr

from retriever import get_retriever
from rag_chain import ask_question
from llm import generate_answer

retriever = get_retriever()


def chat(user_input):

    docs = retriever.invoke(user_input)

    # retrieval score
    query_tokens = set(user_input.lower().split())

    score = 0
    for d in docs:
        doc_tokens = set(d.page_content.lower().split())
        score += len(query_tokens & doc_tokens) / (len(query_tokens) + 1)

    score = round(score / len(docs), 3) if docs else 0.0

    answer, sources = ask_question(user_input, retriever, generate_answer)

    # format sources cleanly
    formatted_sources = "\n".join(
        [f"[{i+1}] {s}" for i, s in enumerate(sources)]
    )

    return answer, formatted_sources, score


with gr.Blocks() as demo:

    gr.Markdown("# 🌿 Home Remedies RAG (Qwen2.5-3B)")

    inp = gr.Textbox(label="Ask a question")

    out1 = gr.Textbox(label="Answer")
    out2 = gr.Textbox(label="Sources")
    out3 = gr.Textbox(label="Retrieval Score")

    btn = gr.Button("Submit")

    btn.click(chat, inputs=inp, outputs=[out1, out2, out3])

demo.launch()