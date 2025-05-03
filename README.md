# GYM_RAG: Fitness Assistant with RAG and Gemini

This project implements a command-line fitness assistant that provides personalized advice on exercise, nutrition, and physiotherapy. It uses Retrieval-Augmented Generation (RAG) by combining a local vector database (built with FAISS and Sentence Transformers) with the generative capabilities of Google's Gemini Flash 1.5 model.

## Features

*   **RAG Implementation:** Retrieves relevant information from local data files (nutrition, exercise, physiotherapy) before generating responses.
*   **Vector Search:** Uses FAISS for efficient similarity search based on user queries.
*   **LLM Integration:** Leverages Google Gemini Flash 1.5 for natural language understanding and response generation.
*   **Data Handling:** Loads and processes data from `.xlsx` and `.csv` files.
*   **Interactive CLI:** Provides a simple command-line interface for user interaction.
*   **Persistence:** Saves and loads the FAISS index and text data using Pickle for faster startup after the initial run.

## Built With

*   [Google Gemini](https://ai.google.dev/)
*   [FAISS](https://github.com/facebookresearch/faiss)
*   [Sentence Transformers](https://www.sbert.net/)
*   [Pandas](https://pandas.pydata.org/)
*   [NumPy](https://numpy.org/)

## License

Specify your project's license here (e.g., MIT, Apache 2.0). You should include a `LICENSE` file in your repository [6].
