import numpy as np
import os
from google import genai
from google.genai import types
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --- Step 1: Read the external text file and split it into chunks ---
def load_and_chunk_text(file_path):
    """Reads and chunks text from a file."""
    # Open and read the entire content of the specified file.
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    
    # Split the text into chunks based on paragraph breaks (double newlines).
    # The strip() method removes leading/trailing whitespace from each chunk.
    # It also filters out any empty chunks that might result from multiple newlines.
    chunks = [chunk.strip() for chunk in text.split('\n\n') if chunk.strip()]
    return chunks

# --- Step 2: Initialize an embedding model ---
def initialize_embedding_model(model_name='all-MiniLM-L6-v2'):
    """Loads a pre-trained model for creating text embeddings."""
    # Instantiate the SentenceTransformer model from the library.
    model = SentenceTransformer(model_name)
    return model

# --- Step 3: Generate embeddings for each chunk ---
def generate_chunk_embeddings(chunks, model):
    """Creates numerical representations (embeddings) for each text chunk."""
    # Use the model's encode method to convert the list of text chunks into vectors.
    # show_progress_bar=True displays a progress bar during this process.
    chunk_embeddings = model.encode(chunks, show_progress_bar=True)
    return chunk_embeddings

# --- Step 4: Generate embedding of the query ---
def generate_query_embedding(query, model):
    """Creates a numerical representation (embedding) for the user's query."""
    # Use the model's encode method to convert the query string into a vector.
    query_embedding = model.encode(query)
    return query_embedding

# --- Step 5: Calculate similarity scores ---
def calculate_similarity(query_embedding, chunk_embeddings):
    """Measures the similarity between the query and each text chunk."""
    # Calculate the cosine similarity between the single query embedding and all chunk embeddings.
    # The query embedding is reshaped to a 2D array to be compatible with the function.
    # The [0] at the end extracts the similarity scores from the nested array structure.
    similarities = cosine_similarity(query_embedding.reshape(1, -1), chunk_embeddings)[0]
    return similarities

# --- Step 6: Extract top K chunks based on similarity score ---
def get_top_k_chunks(chunks, similarities, k=5):
    """Retrieves the most relevant chunks based on similarity."""
    # Get the indices of the chunks that would sort the similarity array in ascending order.
    top_k_indices = np.argsort(similarities)
    # Take the last 'k' indices for the highest scores, and reverse them for descending order.
    top_k_indices = top_k_indices[-k:][::-1]
    
    # Use a list comprehension to get the actual chunk texts corresponding to the top indices.
    top_chunks = [chunks[i] for i in top_k_indices]
    return top_chunks

# --- Step 7: Frame a prompt with the query and the top-k chunks ---
def frame_prompt(query, top_k_chunks):
    """Constructs the final prompt to be sent to the language model."""
    # Join the retrieved chunks into a single string, separated by a clear delimiter.
    context = "\n\n---\n\n".join(top_k_chunks)
    
    # Create a formatted string (f-string) that templates the context and query.
    prompt = f"""
Answer the following query based only on the provided context. If the answer is not in the context, say "I cannot answer based on the provided context."

[CONTEXT]
{context}
[/CONTEXT]

[QUERY]
{query}
[/QUERY]

Answer:
"""
    return prompt

# --- Step 8: Prompt the LLM (Google Gemini) ---
def get_llm_response(prompt):
    """Sends the prompt to the Google Gemini API and gets a response."""
    try:
        # Create the Gemini client - it will automatically pick up the GEMINI_API_KEY environment variable
        client = genai.Client()
        
        # Send the prompt to the model and get the response
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=0)  # Disables thinking
            )
        )
        
        # Return the text part of the model's response
        return response.text
    except Exception as e:
        # If any error occurs (e.g., API issues), print the error message.
        print(f"An error occurred while contacting the LLM: {e}")
        # Return an error message to the user.
        return "Error: Could not get a response from the language model."

# --- Main execution block ---
if __name__ == '__main__':
    # Define the path to your document and your query.
    FILE_PATH = "data/book.txt"  # <<< Make sure this file exists
    USER_QUERY = "what is greatest fear of Dursleys?" # <<< Change this query

    print("--- RAG Pipeline with Google Gemini ---")
    
    # Step 1
    chunks = load_and_chunk_text(FILE_PATH)
    
    # Step 2
    embedding_model = initialize_embedding_model()
    
    # Step 3
    chunk_embeddings = generate_chunk_embeddings(chunks, embedding_model)
    
    # Step 4
    query_embedding = generate_query_embedding(USER_QUERY, embedding_model)
    
    # Step 5
    similarity_scores = calculate_similarity(query_embedding, chunk_embeddings)
    
    # Step 6
    relevant_chunks = get_top_k_chunks(chunks, similarity_scores)
    
    # Step 7
    final_prompt = frame_prompt(USER_QUERY, relevant_chunks)
    print("\n--- Sending Prompt to Gemini ---")
    print(final_prompt) # Display the final prompt for inspection
    
    # Step 8
    print("\n--- Gemini's Response ---")
    llm_response = get_llm_response(final_prompt)
    print(llm_response)