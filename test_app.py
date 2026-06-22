from embed_store import build_vector_store
from retriever import Retriever

def run_test():
    print("🚀 Building vector store...")

    model, index, documents, metadata = build_vector_store()

    print("\n🚀 Initializing retriever...")
    retriever = Retriever(model, index, documents, metadata)

    # test query
    query = "home remedy for headache"

    print(f"\n🔍 Testing query: {query}")

    results = retriever.search(query, k=2)

    print("\n📌 Top Results:\n")

    for i, res in enumerate(results):
        print(f"Rank {i+1}")
        print("Source:", res["source"])
        print("Score:", res["score"])
        print("Text:", res["text"][:150])
        print("-" * 50)

    print("\n✅ TEST COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    run_test()