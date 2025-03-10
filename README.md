# Multilingual Embedding API

A FastAPI service that provides multilingual text embeddings using the `jinaai/jina-embeddings-v3` model.

## Overview

This API allows you to generate vector embeddings for texts in multiple languages. It's built on top of the `jinaai/jina-embeddings-v3` model and provides endpoints for:

- Generating embeddings for text batches
- Counting tokens in text batches

The service automatically selects the optimal device (CUDA GPU, Apple MPS, or CPU) based on availability.

## Features

- Text embedding generation for various tasks:
    - Retrieval (query and passage)
    - Separation
    - Classification
    - Text matching
- Efficient batched processing
- Automatic hardware acceleration (CUDA, MPS, CPU)
- Token counting
- Memory-efficient operation with garbage collection

## Requirements

- Python 3.x
- PyTorch
- Transformers
- FastAPI
- Pydantic
- Uvicorn

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd multilingual-embedding-api

# Install dependencies
pip install -r requirements.txt

# Download the model (optional, will be downloaded on first run otherwise)
python main.py download
```

## Environment Variables

| Variable    | Description                   | Default    |
|-------------|-------------------------------|------------|
| VERSION     | API version                   | "unknown"  |
| BUILD_ID    | Build identifier              | "unknown"  |
| COMMIT_SHA  | Git commit SHA                | "unknown"  |
| PORT        | Server port                   | 8000       |

## Usage

### Starting the server

```bash
python main.py serve
```

Or directly with uvicorn:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

### API Endpoints

#### GET /

Returns basic information about the service.

**Response Example:**

```json
{
  "model_name": "jinaai/jina-embeddings-v3",
  "device": "cuda",
  "version": "1.0.0",
  "build_id": "123",
  "commit_sha": "abc123"
}
```

#### POST /embed

Generates embeddings for a list of texts.

**Request Body:**

```json
{
  "texts": ["This is a sample text", "Another example"],
  "normalize": true,
  "task": "retrieval.query",
  "batch_size": 4
}
```

**Available tasks:**
- `retrieval.query`: Optimize embeddings for search queries
- `retrieval.passage`: Optimize embeddings for passages/documents
- `separation`: For text separation tasks
- `classification`: For classification tasks
- `text-matching`: For text matching tasks

**Response Example:**

```json
{
  "embeddings": [
    [0.1, 0.2, 0.3, ...],
    [0.4, 0.5, 0.6, ...]
  ],
  "computation_time": 0.125
}
```

#### POST /count-tokens

Counts the number of tokens in a list of texts.

**Request Body:**

```json
{
  "texts": ["This is a sample text", "Another example"]
}
```

**Response Example:**

```json
{
  "tokens_count": [
    {
      "text": "This is a sample text",
      "tokens_count": 5
    },
    {
      "text": "Another example",
      "tokens_count": 2
    }
  ],
  "computation_time": 0.01
}
```

## Performance Considerations

- The service dynamically manages memory using garbage collection
- For large batches, consider adjusting the `batch_size` parameter to balance between speed and memory usage
- The service will automatically use GPU acceleration if available

## License

MIT License. See the full license text in the code header.

## Author

Joan Fabrégat <j@fabreg.at>