# 🌿 Home Remedies RAG using LangChain + Qwen2.5-3B

A Retrieval-Augmented Generation (RAG) system that answers home remedy questions using a custom knowledge base.

## Features

- LangChain RAG Pipeline
- FAISS Vector Database
- HuggingFace Embeddings
- Qwen2.5-3B-Instruct LLM
- GPU Inference (RTX 4060)
- Grounded Responses
- Source Citations
- Gradio Frontend

## Project Structure

```text
rag_home_remedies/
│
├── data/
├── vectorstore/
├── ingest.py
├── retriever.py
├── rag_chain.py
├── llm.py
├── app_gradio.py
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Build Vector Store

```bash
python ingest.py
```

## Run Application

```bash
python app_gradio.py
```

## Example Query

```
remedy for fever
```

## Example Output

```
Giloy decoction helps reduce fever naturally.
Drink warm once daily.
```

## Tech Stack

- Python
- LangChain
- FAISS
- HuggingFace
- Qwen2.5-3B
- Gradio