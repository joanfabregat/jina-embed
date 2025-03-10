#  Copyright (c) 2025 Joan Fabrégat <j@fabreg.at>
#  Permission is hereby granted, free of charge, to any person
#  obtaining a copy of this software and associated documentation
#  files (the "Software"), to deal in the Software without
#  restriction, subject to the conditions in the full MIT License.
#  The Software is provided "as is", without warranty of any kind.

import enum
import gc
import logging
import os
import time

import torch
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field, conlist
from transformers import AutoTokenizer, AutoModel

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
        RETRIEVAL_QUERY: str = "retrieval.query"
        RETRIEVAL_PASSAGE: str = "retrieval.passage"
        SEPARATION: str = "separation"
        CLASSIFICATION: str = "classification"
        TEXT_MATCHING: str = "text-matching"

    texts: conlist(str, min_length=1) = Field(..., description="List of texts to embed")
    normalize: bool = Field(True, description="Whether to normalize the embeddings")
    task: Task = Field(Task.RETRIEVAL_QUERY, description="Embedding task")
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
    device: str
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
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    )
    model = AutoModel.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if torch.mps.is_available()
        else torch.device("cpu")
    )
    model.eval()
    model.to(device)
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
    return InfoResponse(device=str(device))


# noinspection PyTypeChecker
@app.post("/embed", response_model=EmbedResponse)
def embed(request: EmbedRequest) -> EmbedResponse:
    """Get embeddings for a batch of texts with memory efficiency"""
    global model
    texts_num = len(request.texts)

    logger.info(f"Embedding {texts_num} texts using {MODEL_NAME} with batching")

    try:
        start = time.time()

        batches: list[str] = [
            request.texts[i:i + request.batch_size]
            for i in range(0, len(request.texts), request.batch_size)
        ]
        all_embeddings: list[float] = []
        for i, batch in enumerate(batches):
            try:
                logger.info(f"Processing batch {i + 1}/{(texts_num + request.batch_size - 1) // request.batch_size} "
                            f"with {len(batch)} texts")

                with torch.no_grad():
                    # Process a single batch
                    batch_output = model.encode(sentences=batch, task=request.task.value)

                # If output is a tensor, convert to list
                if isinstance(batch_output, torch.Tensor):
                    batch_output = batch_output.detach().cpu().numpy().tolist()

                # Handle different output types from the model's encode method
                if isinstance(batch_output, list):
                    # Check if we need to do conversion from tensors
                    if batch_output and isinstance(batch_output[0], torch.Tensor):
                        batch_output = [t.cpu().numpy().tolist() for t in batch_output]

                # Add batch results to the full results list
                all_embeddings.extend(batch_output)

                del batch_output
            finally:
                _force_gc()

        logger.info(f"Completed embedding {texts_num} texts")

        return EmbedResponse(
            embeddings=all_embeddings,
            computation_time=time.time() - start
        )
    except Exception as e:
        raise Exception(f"Failed to embed {texts_num} texts: {e}") from e
    finally:
        _force_gc()


@app.post("/count-tokens", response_model=CountTokensResponse)
def count_tokens(request: CountTokensRequest) -> CountTokensResponse:
    """Count the number of tokens in a batch of texts."""
    start = time.time()
    tokens_count = [
        len(tokenizer.tokenize(text)) for text in request.texts
    ]
    return CountTokensResponse(
        tokens_count=tokens_count,
        computation_time=time.time() - start
    )


def _force_gc():
    """Force garbage collection and clear CUDA/MPS cache."""
    logger.info("Forcing garbage collection")
    gc.collect()
    if device:
        match str(device):
            case "cuda":
                logger.debug("Clearing CUDA cache")
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            case "mps":
                logger.debug("Clearing MPS cache")
                torch.mps.empty_cache()
                torch.mps.synchronize()


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
