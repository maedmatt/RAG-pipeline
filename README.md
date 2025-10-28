# RAG Pipeline with Google Gemini

This project implements a simple Retrieval-Augmented Generation (RAG) pipeline in Python.

It uses a text file as a knowledge base, generates embeddings using `sentence-transformers`, finds relevant chunks for a user query, and then uses Google's Gemini model to generate an answer based on the retrieved context.

## What is RAG?

-   **Retrieval**: When you ask a question (a "query"), the system first searches through your document to find the most relevant snippets of text. This is the "R" in RAG.
-   **Augmentation**: It then takes these retrieved snippets and "augments" your original question by bundling them together as context.
-   **Generation**: Finally, it sends this combined package (context + original question) to the LLM, which uses the provided information to "generate" a well-informed answer. This is the "G" in RAG.

## How it works

The pipeline follows these main steps:

1.  Read the external text file and split it into chunks.
2.  Initialize an embedding model.
3.  Generate embeddings for each chunk.
4.  Generate an embedding for the user's query.
5.  Calculate a similarity score between the query embedding and each chunk embedding.
6.  Extract the top 'k' chunks based on their similarity scores.
7.  Frame a prompt containing the user's query and the retrieved chunks.
8.  Send the framed prompt to the LLM to get the final answer.

## How to run

1.  **Set up the environment:**
    -   Make sure you have Python 3.12+ and `uv` installed.
    -   Install the dependencies:
        ```bash
        uv sync
        ```

2.  **Set your API Key:**
    -   Add your `GEMINI_API_KEY` as an environment variable.

3.  **Run the script:**
    ```bash
    uv run main.py
    ```
