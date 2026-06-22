import streamlit as st
from rag_chain import ask_question

st.set_page_config(page_title="Home Remedies RAG", layout="centered")

st.title("🌿 Home Remedies RAG Assistant")

query = st.text_input("Ask your health question:")

if st.button("Get Answer"):

    if query.strip() == "":
        st.warning("Please enter a question")
    else:
        answer, context, sources = ask_question(query)

        st.subheader("🧠 Answer")
        st.write(answer)

        st.subheader("📚 Retrieved Context (Grounding)")
        st.write(context)

        st.subheader("🔗 Citations")
        for s in sources:
            st.write("•", s)