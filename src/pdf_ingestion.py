import fitz  # PyMuPDF
import tempfile
import os
from pathlib import Path
import base64
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from ollama import chat

def generate_image_description(image_path: str) -> str:
    """
    Uses a vision model to generate a description of an image.
    """
    try:
        # Use a dedicated vision model like Llava for description
        response = chat(
            model="gemma3:latest", # Or another powerful vision model
            messages=[
                {
                    "role": "user",
                    "content": "Describe this image in detail. What is it about? What are the key elements, objects, and any text present?",
                    "images": [image_path]
                }
            ]
        )
        return response["message"]["content"]
    except Exception as e:
        print(f"Error generating image description: {e}")
        return "Could not generate a description for this image."

def extract_and_describe(file_path):
    """
    Extracts text and generates descriptions for images from a PDF.
    """
    doc = fitz.open(file_path)
    documents = []

    # Extract text per page
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text()
        if text.strip():
            documents.append(
                Document(
                    page_content=text,
                    metadata={"type": "text", "page": page_num + 1, "source": str(file_path)}
                )
            )
        # Extract images, save temporarily, and generate description
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]

            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_img_file:
                temp_img_file.write(image_bytes)
                temp_img_path = temp_img_file.name

            # Generate description from the temp image file
            description = generate_image_description(temp_img_path)

            documents.append(
                Document(
                    page_content=description, # The description is the main content now
                    metadata={
                        "type": "image_description",
                        "page": page_num + 1,
                        "source": str(file_path)
                    }
                )
            )
            os.remove(temp_img_path) # Clean up the temp image

    doc.close()
    return documents

def vector_db(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = Path(temp_file.name)

    # Extract text and image descriptions
    all_docs = extract_and_describe(temp_file_path)
    os.remove(temp_file_path)

    # Split all documents (text and descriptions)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(all_docs)

    # Embed and store in ChromaDB
    embeddings = OllamaEmbeddings(model="granite-embedding:latest")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    return vectorstore