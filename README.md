# PDF Query App

## Overview

The PDF Query App is a web application built using Streamlit and LangChain that allows users to upload PDF files, extract text from them, and perform similarity-based queries to find relevant information. The application leverages OpenAI's embeddings and language models to understand and respond to user queries based on the contents of the uploaded PDFs.

## Features

- **PDF Upload and Processing**: Users can upload PDF files which are then processed to extract text.
- **Text Embeddings**: Extracted text is converted into embeddings using OpenAI's embedding models.
- **Similarity Search**: Queries are processed to find relevant text chunks based on similarity scores.
- **Interactive UI**: A user-friendly interface for uploading PDFs, inputting queries, and viewing results.

## How It Works

### Text Extraction

1. **PDF Upload**: Users upload a PDF file using the file uploader provided in the sidebar.
2. **Text Extraction**: Once the PDF is uploaded and processed, the text is extracted from each page using `PyPDF2`. The text is then combined into a single string for further processing.

### Generating Embeddings

1. **Text Splitting**: The combined text is split into chunks using `CharacterTextSplitter` to ensure that it fits within the embedding model's limits.
2. **Embedding Generation**: Each text chunk is converted into a vector representation (embedding) using OpenAI's embeddings API.

### Similarity Search

1. **Vector Store Creation**: The text chunks and their embeddings are stored in a Cassandra vector store.
2. **Query Processing**: When a user submits a query, it is also converted into an embedding.
3. **Similarity Search**: The vector store performs a similarity search to find the most relevant text chunks based on the query embedding.

### User Interface

1. **Upload PDF**: Users upload a PDF file through the file uploader in the sidebar.
2. **Process PDF**: Click the "Process PDF" button to extract text and store embeddings.
3. **Enter Query**: Once the PDF is processed, users can enter their questions in the text input field.
4. **Get Answer**: The app queries the vector store and displays the most relevant answers along with the context from the document.

## Installation

1. **Clone the Repository**:

    ```bash
    git clone https://github.com/RajarshiB21/pdf-query-app.git
    cd pdf-query-app
    ```

2. **Install Dependencies**:

    Ensure you have Python 3.7 or higher installed. Then install the required packages:

    ```bash
    pip install streamlit langchain openai cassio PyPDF2
    ```

3. **Run the Application**:

    ```bash
    streamlit run app.py
    ```

## Configuration

You need to provide API keys and database tokens in the script. Replace the placeholder values with your actual keys and tokens:

- `ASTRA_DB_APPLICATION_TOKEN`: Your Astra DB application token.
- `ASTRA_DB_ID`: Your Astra DB database ID.
- `OPENAI_API_KEY`: Your OpenAI API key.

## Contributing

Feel free to submit issues or pull requests. For any major changes or feature requests, please open an issue to discuss.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

For any questions or further assistance, please contact [banerjeerajarshi24@gmail.com] or open an issue on this repository.
