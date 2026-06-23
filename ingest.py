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

    for file_name in files:

        path = os.path.join(DATA_PATH, file_name)

        loader = TextLoader(path, encoding="utf-8")

        loaded_docs = loader.load()

        for doc in loaded_docs:

            doc.metadata["source_file"] = file_name

            doc.metadata["remedy_name"] = (
                file_name.replace(".txt", "")
                .replace("_", " ")
                .title()
            )

        docs.extend(loaded_docs)

    print(f"📚 Documents loaded: {len(docs)}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(docs)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i

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