import tempfile
import fitz
import os
import tempfile
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
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
def vector_db(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = Path(temp_file.name)

    # Extract text from PDF using PyMuPDF
    docs = extract_text_from_pdf(temp_file_path)

    # Clean up the temporary file after processing
    os.remove(temp_file_path)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=400)
    chunks = splitter.split_documents(docs)

    # Embed the chunks and store them in Chroma DB
    embeddings = OllamaEmbeddings(model="granite-embedding:latest")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    return vectorstore