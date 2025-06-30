import streamlit as st
from pdf_ingestion import *
from ollama import chat
from answer import *

st.title("PDF RAG with Granite Vision & ChromaDB")
prompt = st.chat_input(
    "Say something and/or attach an image",
)

uploaded_file=st.file_uploader("Drop your pdf",type="pdf")
# Save the uploaded file to a temporary directory
if uploaded_file:
    vectorstore=vector_db(uploaded_file)
    st.success(f"✅ PDF processed and embedded successfully! You can now ask questions about the document.")

# Chat interface
if prompt and prompt.strip():
    with st.spinner("Retrieving relevant information..."):
        context = retrieve(prompt,vectorstore)

    with st.spinner("Generating answer..."):
        answer = generate_answer(prompt, context)

    st.write("**Answer:**", answer)

    # Show retrieved context (optional)
    with st.expander("Show retrieved context"):
        st.text(context)

