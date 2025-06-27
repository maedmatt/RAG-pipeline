# RAG Pipeline with Google Gemini

This project implements a simple Retrieval-Augmented Generation (RAG) pipeline in Python.

It uses a text file as a knowledge base, generates embeddings using `sentence-transformers`, finds relevant chunks for a user query, and then uses Google's Gemini model to generate an answer based on the retrieved context.

## How it works

The pipeline follows these main steps:
1.  **Load Data**: Reads a text file and splits it into smaller chunks.
2.  **Generate Embeddings**: Uses a pre-trained model to create numerical representations (embeddings) of the text chunks.
3.  **Find Relevant Chunks**: Compares the user's query embedding with the chunk embeddings to find the most similar ones.
4.  **Generate Answer**: Feeds the relevant chunks and the original query to the Gemini LLM to get a final answer.

## How to run
1. Add your `GEMINI_API_KEY` as an environment variable.
2. Run `uv pip sync` to update your lock file with the exact versions of the packages.
3. Run `python src/main.py`
