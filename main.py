#  Copyright (c) 2025 Joan Fabrégat <j@fabreg.at>
#  Permission is hereby granted, free of charge, to any person
#  obtaining a copy of this software and associated documentation
#  files (the "Software"), to deal in the Software without
#  restriction, subject to the conditions in the full MIT License.
#  The Software is provided "as is", without warranty of any kind.

import enum
import logging
import os

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastembed import TextEmbedding
from fastembed.text.multitask_embedding import JinaEmbeddingV3
from pydantic import BaseModel, Field, conlist

##
# Initialize logging
##
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(message)s')

##
# Load the config
##
MODEL_NAME = "jinaai/jina-embeddings-v3"
VERSION = os.getenv("VERSION") or "unknown"
BUILD_ID = os.getenv("BUILD_ID") or "unknown"
COMMIT_SHA = os.getenv("COMMIT_SHA") or "unknown"
PORT = int(os.getenv("PORT", "8000"))


##
# Models
##
class EmbedRequest(BaseModel):
    class Task(str, enum.Enum):
        QUERY: str = "query"
        INDEX: str = "index"

    texts: conlist(str, min_length=1) = Field(..., description="List of texts to embed")
    task: Task = Field(Task.QUERY, description="Embedding task")
    batch_size: int = Field(4, description="Batch size for processing texts")


class EmbedResponse(BaseModel):
    embeddings: list[list[float]]
    computation_time: float = Field(..., description="Time taken to compute the reranking in seconds")


class CountTokensRequest(BaseModel):
    texts: conlist(str, min_length=1) = Field(..., description="List of texts to count tokens")


class CountTokensResponse(BaseModel):
    tokens_count: list[int] = Field(..., description="List of token counts for each text")
    computation_time: float = Field(..., description="Time taken to compute the reranking in seconds")


class InfoResponse(BaseModel):
    model_name: str = MODEL_NAME
    version: str = VERSION
    build_id: str = BUILD_ID
    commit_sha: str = COMMIT_SHA


##
# Create the FastAPI app
##
app = FastAPI(
    title="Multilingual embedding API",
    description=f"API for embedding documents based on query relevance using {MODEL_NAME}",
    version=VERSION,
)

##
# Load the model
##
try:
    logger.info(f"Loading model {MODEL_NAME}...")
    model = TextEmbedding(model_name=MODEL_NAME)
    logger.info(f"Model {MODEL_NAME} loaded successfully")
except Exception as e:
    raise RuntimeError(f"Failed to load model: {str(e)}")


##
# Routes
##
@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.get("/info", response_model=InfoResponse)
def info() -> InfoResponse:
    """Get the root endpoint."""
    return InfoResponse()


# noinspection PyTypeChecker
@app.post("/embed", response_model=list[list[float]])
def embed(request: EmbedRequest) -> list[list[float]]:
    """
    Get embeddings for a batch of texts with memory efficiency

    Args:
        request: Request object with texts to embed

    Returns:
        List of embeddings
    """
    logger.info(f"Embedding {len(request.texts)} texts using {MODEL_NAME} with batching")

    # Resolve task to Jina task ID
    match request.task:
        case EmbedRequest.Task.QUERY:
            task_id = JinaEmbeddingV3.QUERY_TASK
        case EmbedRequest.Task.INDEX:
            task_id = JinaEmbeddingV3.PASSAGE_TASK
        case _:
            raise ValueError(f"Unsupported task {request.task}")

    # Embed texts in batches
    embeddings = model.embed(request.texts, batch_size=request.batch_size, task_id=task_id)

    # Convert embeddings to list of lists
    embeddings = [
        embedding.tolist()
        for embedding in embeddings
    ]

    return embeddings


if __name__ == "__main__":
    import sys

    command = sys.argv[1] if len(sys.argv) > 1 else "serve"

    # Start the server
    if command == "serve":
        import uvicorn

        uvicorn.run("main:app", host="0.0.0.0", port=PORT)

    # Download the model
    elif command == "download":
        sys.exit(0)
