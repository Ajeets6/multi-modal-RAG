# Retrieve the most relevant chunks from the vectorstore
from ollama import chat
def retrieve(query: str,vectorstore) -> str:
    results = vectorstore.similarity_search(query, k=3)
    context="\n\n".join([doc.page_content for doc in results])
    return context

def generate_answer(query: str, context: str) -> str:
    response = chat(
        model="gemma3:latest",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. If applicable answer questions based on the provided context from the document otherwise reply conversationally"},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}
        ]
    )
    return response["message"]["content"]