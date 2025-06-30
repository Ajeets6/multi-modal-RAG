# Retrieve the most relevant chunks from the vectorstore
from ollama import chat
def retrieve(query: str,vectorstore) -> str:
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