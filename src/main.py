import streamlit as st

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from ollama import chat
import tempfile
import fitz
# Function to extract text from PDF using PyMuPDF
def extract_text_from_pdf(file_path):
    """Extract text from PDF using PyMuPDF"""
    doc = fitz.open(file_path)
    documents = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text()

        # Create Document object for each page
        document = Document(
            page_content=text,
            metadata={
                "page": page_num + 1,
                "source": str(file_path)
            }
        )
        documents.append(document)

    doc.close()
    return documents

prompt = st.chat_input(
    "Say something and/or attach an image",
)
import os
import tempfile
from pathlib import Path
uploaded_file=st.file_uploader("Drop your pdf",type="pdf")
# Save the uploaded file to a temporary directory
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = Path(temp_file.name)

    # Extract text from PDF using PyMuPDF
    docs = extract_text_from_pdf(temp_file_path)

    # Clean up the temporary file after processing
    os.remove(temp_file_path)

    splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=1500)
    chunks = splitter.split_documents(docs)

    # Embed the chunks and store them in Chroma DB
    embeddings = OllamaEmbeddings(model="granite-embedding:latest")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )

# Retrieve the most relevant chunks from the vectorstore
def retrieve(query: str) -> str:
    results = vectorstore.similarity_search(query, k=3)
    return "\n\n".join([doc.page_content for doc in results])

def generate_answer(query: str, context: str) -> str:
    response = chat(
        model="granite3.2-vision:latest",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Answer questions based on the provided context from the document."},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}
        ]
    )
    return response["message"]["content"]

st.title("PDF RAG with Granite Vision & ChromaDB")

# Chat interface
if prompt and prompt.strip():
    with st.spinner("Retrieving relevant information..."):
        context = retrieve(prompt)

    with st.spinner("Generating answer..."):
        answer = generate_answer(prompt, context)

    st.write("**Answer:**", answer)

    # Show retrieved context (optional)
    with st.expander("Show retrieved context"):
        st.text(context)

# Display status when PDF is uploaded
if uploaded_file:
    st.success(f"✅ PDF processed and embedded successfully! You can now ask questions about the document.")
    st.info(f"📄 Document contains {len(chunks)} text chunks")
