import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

print(">>> ingest.py started")

def main():

    DATA_PATH = "data"
    DB_PATH = "vectorstore"

    if not os.path.exists(DATA_PATH):
        print("❌ data folder missing")
        return

    files = [f for f in os.listdir(DATA_PATH) if f.endswith(".txt")]
    print(f"📄 Files found: {len(files)}")

    docs = []
    for f in files:
        path = os.path.join(DATA_PATH, f)
        loader = TextLoader(path, encoding="utf-8")
        docs.extend(loader.load())

    print(f"📚 Documents loaded: {len(docs)}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=350,
        chunk_overlap=70
    )

    chunks = splitter.split_documents(docs)
    print(f"✂️ Chunks created: {len(chunks)}")

    print("🧠 Creating embeddings...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.from_documents(chunks, embeddings)

    os.makedirs(DB_PATH, exist_ok=True)
    db.save_local(DB_PATH)

    print("✅ Vector DB saved at:", os.path.abspath(DB_PATH))


if __name__ == "__main__":
    main()