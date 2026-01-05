# Multimodal-document-processing-system(MDP)

A production-ready system that ingests documents (PDF, Docx, Excel, Images) and answers questions using Google Gemini API.

## Prerequisites

Before starting, ensure you have the following installed on your system:

### 1. Tesseract OCR (Required for Images)
This project uses OCR to extract text from images.

*   **Ubuntu/Debian**:
    ```bash
    sudo apt-get install tesseract-ocr
    ```
*   **macOS**:
    ```bash
    brew install tesseract
    ```

### 2. Docker
Docker is required to run the Milvus vector database. Ensure the Docker daemon is running before proceeding.

## Database Setup

We use **Milvus** as our vector store. A standalone setup script is included for convenience.

> **Note**: `standalone_embed.sh` is the official standalone installer script from Milvus. You can verify or download it directly from the [Milvus Documentation](https://milvus.io/docs/install_standalone-docker.md).

To start the database:

```bash
bash standalone_embed.sh start
```

This will spin up the necessary Milvus containers.

## Installation

1.  **Clone the repository and enter the directory.**

2.  (Optional but recommended) **Create and activate a virtual environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use: venv\Scripts\activate
    ```

3.  **Install Python dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Configuration

You need a Google Gemini API key to generate answers.

Create a `.env` file in the root directory (or set it as an environment variable) containing:

```env
GEMINI_API_KEY=your_api_key_here
```

## Running the App

Start the FastAPI server using `uvicorn`:

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`.

## API Usage

### 1. Upload a Document
Uploads a file, embeds the content, and stores it in Milvus.

*   **Endpoint**: `POST /upload`
*   **Key**: `file`

```bash
curl -X POST -F "file=@/path/to/your/document.pdf" http://localhost:8000/upload
```
*Note the `collection_created` name returned in the JSON response.*

### 2. Ask a Question
Queries the stored document context to answer your question.

*   **Endpoint**: `POST /query`
*   **Body**: `collection_name` (from the upload step) and `question`.

```bash
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"collection_name": "your_file_collection_name", "question": "What is the summary of this document?"}'
```
