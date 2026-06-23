import gradio as gr
from retriever import get_retriever
from rag_chain import ask_question
from llm import generate_answer

print("Loading retriever...")
retriever = get_retriever()
print("Retriever ready!")

def chat(user_input):
    return ask_question(
user_input,
retriever,
generate_answer
)

demo = gr.Interface(
fn=chat,
inputs=gr.Textbox(
lines=2,
placeholder="Ask a home remedy question..."
),
outputs=gr.Markdown(),
title="🌿 Home Remedies RAG (Qwen2.5-3B)",
theme="soft"
)

demo.launch()
