from retriever import get_retriever

retriever = get_retriever()

query = "home remedy for cold"

docs = retriever.invoke(query)

for i, d in enumerate(docs):
    print(f"\n--- Result {i+1} ---")
    print(d.page_content)