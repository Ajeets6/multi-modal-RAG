from ollama import chat

def retrieve(query: str, vectorstore, k: int = 3):
    """
    Retrieves relevant text chunks or image descriptions.
    """
    results = vectorstore.similarity_search(query, k=k)
    # No need to check for images, just join the page_content
    context = "\n\n".join([doc.page_content for doc in results])
    return context

def generate_answer(query: str, context: str) -> str:
    """
    Generates an answer based on the retrieved context (text or image descriptions).
    """
    system_prompt = "You are a helpful assistant. Answer the user's question based on the provided context. The context might be a standard text or a description of an image from a document. If no context is provided, answer conversationally."

    if context.strip():
        # Case 1: Context is available (text or image description)
        user_content = f"Context: {context}\n\nQuestion: {query}"
    else:
        # Case 2: No context, just a conversational query
        user_content = query

    response = chat(
        model="gemma3:latest", # Or your preferred final model
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
    )

    return response["message"]["content"]