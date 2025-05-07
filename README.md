# Understanding LLMs: Tokens and Embeddings

## Overview

This repository provides a practical guide to understanding fundamental concepts in Large Language Models (LLMs), specifically focusing on **tokens**, **tokenization**, and **embeddings**. The primary resource is the `Understanding_LLMs.ipynb` Jupyter Notebook, which walks through code examples to illustrate these concepts.

## Key Concepts Covered

### Tokens & Tokenization

*   **What are Tokens?** Tokens are the basic units of text that an LLM processes. They can be words, subwords, or even individual characters, depending on the tokenization strategy.
*   **What is Tokenization?** Tokenization is the process of breaking down a piece of text into these smaller units (tokens). This is a crucial preprocessing step for LLMs, as they operate on numerical representations of these tokens.
*   **Demonstration:** The notebook uses the `AutoTokenizer` from the Hugging Face `transformers` library to show how raw text is converted into a sequence of token IDs.

### Embeddings

*   **What are Embeddings?** Embeddings are dense vector representations of tokens or pieces of text. These vectors capture the semantic meaning and context of the words they represent. Words with similar meanings will have embeddings that are closer together in the vector space.
*   **Why are they Important?** Embeddings allow LLMs to understand and process language in a way that captures nuances, relationships, and context, which is essential for tasks like text generation, translation, and sentiment analysis.
*   **Demonstration:** The notebook utilizes the `google.genai` library to generate embeddings for sample text, illustrating how text can be converted into meaningful numerical vectors.

## Notebook: `Understanding_LLMs.ipynb`

### Purpose

The Jupyter Notebook `Understanding_LLMs.ipynb` serves as an interactive, step-by-step guide to:
1.  Install and set up necessary libraries.
2.  Understand and implement text tokenization.
3.  Generate and understand text embeddings.
4.  Perform basic text generation using a pre-trained causal language model.

### Key Operations Demonstrated

*   **Dependency Installation:**
    *   `transformers` (from Hugging Face for tokenizers and models)
    *   `google-generativeai` (for Google's Generative AI models, including embedding generation)
    *   `python-dotenv` (for managing API keys)
*   **API Key Management:**
    *   The notebook demonstrates setting API keys directly as environment variables (e.g., `os.environ["HF_TOKEN"]`).
    *   **Recommended Practice:** For security and better configuration management, it is highly recommended to store API keys (like `HF_TOKEN` for Hugging Face and `GOOGLE_API_KEY` for Google AI) in a `.env` file in the root of the project. The notebook can then be modified to load these keys using `python-dotenv`.
        ```python
        # Example of loading from .env (add this to the notebook)
        # At the beginning of your notebook:
        # !pip install python-dotenv
        # import os
        # from dotenv import load_dotenv
        # load_dotenv()
        # hf_token = os.getenv("HF_TOKEN")
        # google_api_key = os.getenv("GOOGLE_API_KEY")
        ```
*   **Text Tokenization:**
    *   Using `AutoTokenizer.from_pretrained("google/gemma-3-1b-it")` to load a tokenizer.
    *   Converting text strings into input IDs and attention masks.
*   **Embedding Generation:**
    *   Initializing the `genai.Client` with an API key.
    *   Using `client.models.embed_content()` to get embedding vectors for given text.
*   **Text Generation:**
    *   Loading a pre-trained Causal Language Model: `AutoModelForCausalLM.from_pretrained("google/gemma-3-1b-it")`.
    *   Generating text by providing tokenized input to the model using `model.generate()`.
    *   Decoding the generated token IDs back into human-readable text using `tokenizer.batch_decode()`.

## How to Run

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-name>
    ```
2.  **Create a `.env` file:**
    In the root directory of the project, create a file named `.env` and add your API keys:
    ```env
    HF_TOKEN="your_hugging_face_token"
    GOOGLE_API_KEY="your_google_ai_api_key"
    ```
3.  **Install Dependencies:**
    The notebook installs dependencies directly within its cells. Ensure you have pip installed.
    Alternatively, you can create a `requirements.txt` file with the following content and run `pip install -r requirements.txt`:
    ```txt
    transformers
    google-generativeai
    python-dotenv
    torch
    # Add any other specific versions if necessary
    ```
4.  **Open and Run the Notebook:**
    Launch Jupyter Notebook or JupyterLab and open `Understanding_LLMs.ipynb`. Execute the cells sequentially to see the concepts in action.

## Dependencies

The main Python libraries used in this project are:
*   `transformers` (Hugging Face)
*   `google-generativeai`
*   `python-dotenv`
*   `torch`

Please refer to the notebook for specific versions if encountering compatibility issues. 