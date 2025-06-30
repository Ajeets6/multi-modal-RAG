import streamlit as st
from pdf_ingestion import *
from ollama import chat
from answer import *

st.title("PDF RAG with Granite Vision & ChromaDB")

uploaded_file=st.file_uploader("Drop your pdf",type="pdf")
if uploaded_file:
    vectorstore=vector_db(uploaded_file)
    st.success(f"✅ PDF processed and embedded successfully! You can now ask questions about the document.")

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        if 'vectorstore' in locals() and vectorstore is not None:
            with st.spinner("Retrieving relevant information..."):
                context= retrieve(prompt, vectorstore)

            with st.spinner("Generating answer..."):
                answer = generate_answer(prompt, context)


        st.write(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})

