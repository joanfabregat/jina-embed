# Multilingual Embedding API

[![Build and Push to GHCR and Docker Hub](https://github.com/joanfabregat/jina-embed/actions/workflows/build.yaml/badge.svg)](https://github.com/joanfabregat/jina-embed/actions/workflows/build.yaml)

A FastAPI service that provides text embedding capabilities using the [
`jinaai/jina-embeddings-v3`](https://huggingface.co/jinaai/jina-embeddings-v3) model.

## Overview

This API allows you to generate vector embeddings for text documents using the `jinaai/jina-embeddings-v3` model. The
service supports different embedding tasks (query and indexing) and is designed to process batches of texts efficiently.

## Features

- Generate embeddings for multiple texts in a single request
- Support for both query and index embedding tasks
- Efficient batch processing
- Token counting endpoint
- Service information endpoint

## Requirements

- Python 3.13+
- FastAPI
- fastembed
- Pydantic
- uvicorn

## Environment Variables

The service can be configured using the following environment variables:

- `PORT`: The port to run the server on (default: 8000)
- `VERSION`: The version of the service (default: "unknown")
- `BUILD_ID`: The build identifier (default: "unknown")
- `COMMIT_SHA`: The commit SHA (default: "unknown")

## Usage

### Starting the Server

The recommended way to run this service is using Docker.

```shell
docker run -p 8000:8000 joanfabregat/jina-embed:latest
```

Documentation for the API can be found at `/docs` or `/redoc` when running the server.

### API Endpoints

#### GET `/info`

Returns information about the service.

```json
{
  "model_name": "jinaai/jina-embeddings-v3",
  "version": "1.0.0",
  "build_id": "12345",
  "commit_sha": "abc123"
}
```

#### POST `/embed`

Generates embeddings for a list of texts.

Request body:

```json
{
  "texts": [
    "This is a sample text",
    "Another sample text"
  ],
  "task": "query",
  "batch_size": 4
}
```

Response:

```json
[
  [
    0.123,
    0.456,
    ...
  ],
  [
    0.789,
    0.012,
    ...
  ]
]
```

Parameters:

- `texts`: List of texts to embed (required)
- `task`: Embedding task, either "query" or "index" (default: "query")
- `batch_size`: Batch size for processing texts (default: 4)

## License

This project is licensed under the MIT License - see the license notice in the code for details.

## Credits

Developed by Joan Fabrégat, j@fabreg.at