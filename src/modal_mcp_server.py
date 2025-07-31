"""
Modal app that exposes an MCP (Model Context Protocol) server.
A client sends a natural-language query and gets back the list of
transcript chunks the retriever deems most relevant.
"""
import sys
import os
from pathlib import Path

import modal
from fastmcp import FastMCP
from starlette.responses import JSONResponse

# Add the src directory to Python path for Modal environment
sys.path.append("/root/src")

from vector_emb import retrieve_relevant_chunks, get_chroma_client, get_or_create_collection

# Modal app setup
app = modal.App("bwai-mcp-server")

# Define the Docker image with all dependencies
image = modal.Image.debian_slim().pip_install(
    "fastmcp",
    "openai>=1.0.0",
    "chromadb",
    "tiktoken",
    "python-dotenv",
    "fastapi>=0.100.0",
    "uvicorn>=0.20.0"
).add_local_dir("src", "/root/src")

# Define secrets
secrets = [
    modal.Secret.from_name("openai-secret"),
]

# Define Modal volume for ChromaDB persistence
chroma_volume = modal.Volume.from_name("chroma-db-volume-2", create_if_missing=True)
volume_mounts = {
    "/root/chroma_db": chroma_volume
}

@app.function(
    image=image,
    secrets=secrets,
    volumes=volume_mounts,
)
@modal.asgi_app()
def fastapi_app():
    """
    This is the main entrypoint for the Modal app.
    The MCP server and its routes are defined inside this function
    to ensure they are created in an environment where secrets are available.
    """
    mcp = FastMCP("bwai-mcp-server")

    @mcp.tool
    def get_relevant_chunks(question: str, top_k: int = 10) -> dict:
        """Retrieve relevant transcript chunks for a given question"""
        try:
            chunks = retrieve_relevant_chunks(question, n_results=top_k)
            formatted_chunks = []
            for chunk in chunks:
                formatted_chunk = {
                    "id": chunk["id"],
                    "text": chunk["text"],
                    "workshop": chunk["metadata"].get("workshop_id", "Unknown"),
                    "timestamp": chunk["metadata"].get("timestamp", "Unknown"),
                    "speaker": chunk["metadata"].get("speaker", "Unknown"),
                    "position": chunk["metadata"].get("position", "Unknown"),
                    "relevance": chunk.get("relevance", 1.0)
                }
                formatted_chunks.append(formatted_chunk)
            
            return {
                "chunks": formatted_chunks,
                "total_chunks": len(formatted_chunks)
            }
        except Exception as e:
            return {
                "error": f"Failed to retrieve chunks: {str(e)}",
                "chunks": [],
                "total_chunks": 0
            }

    @mcp.custom_route("/health", methods=["GET"])
    async def health_check(request):
        """Health check endpoint"""
        api_key_found = os.getenv("OPENAI_API_KEY") is not None
        try:
            client = get_chroma_client()
            collection = get_or_create_collection(client)
            count = collection.count()
            return JSONResponse({
                "status": "healthy",
                "openai_api_key_found": api_key_found,
                "chroma_connected": True,
                "total_chunks": count
            })
        except Exception as e:
            return JSONResponse({
                "status": "unhealthy",
                "openai_api_key_found": api_key_found,
                "chroma_connected": False,
                "error": str(e)
            })

    # Return the HTTP app - use stateless mode for serverless deployment
    # This is important for Modal since each request may hit a different instance
    return mcp.http_app(stateless_http=True)

# The local_entrypoint is more complex now because it needs to replicate
# the behavior of the decorated Modal function for local testing.
@app.local_entrypoint()
def main():
    """Run the MCP server locally for testing."""
    from dotenv import load_dotenv
    # Load .env file to simulate secrets for local development
    load_dotenv()

    # We need to create a temporary app object to run locally,
    # as the real routes are defined inside the Modal-decorated function.
    local_mcp = FastMCP("bwai-mcp-server-local")

    @local_mcp.tool
    def get_relevant_chunks_local(question: str, top_k: int = 10) -> dict:
        """Local version of the get_relevant_chunks tool."""
        try:
            chunks = retrieve_relevant_chunks(question, n_results=top_k)
            formatted_chunks = []
            for chunk in chunks:
                formatted_chunk = {
                    "id": chunk["id"],
                    "text": chunk["text"],
                    "workshop": chunk["metadata"].get("workshop_id", "Unknown"),
                    "timestamp": chunk["metadata"].get("timestamp", "Unknown"),
                    "speaker": chunk["metadata"].get("speaker", "Unknown"),
                    "position": chunk["metadata"].get("position", "Unknown"),
                    "relevance": chunk.get("relevance", 1.0)
                }
                formatted_chunks.append(formatted_chunk)
            
            return {
                "chunks": formatted_chunks,
                "total_chunks": len(formatted_chunks)
            }
        except Exception as e:
            return {
                "error": f"Failed to retrieve chunks: {str(e)}",
                "chunks": [],
                "total_chunks": 0
            }

    @local_mcp.custom_route("/health", methods=["GET"])
    async def health_check_local(request):
        """Local version of the health check endpoint."""
        api_key_found = os.getenv("OPENAI_API_KEY") is not None
        try:
            client = get_chroma_client()
            collection = get_or_create_collection(client)
            count = collection.count()
            return JSONResponse({
                "status": "healthy",
                "openai_api_key_found": api_key_found,
                "chroma_connected": True,
                "total_chunks": count
            })
        except Exception as e:
            return JSONResponse({
                "status": "unhealthy",
                "openai_api_key_found": api_key_found,
                "chroma_connected": False,
                "error": str(e)
            })
    
    print("🚀 Starting local MCP Server...")
    print("🔗 MCP endpoint available at: http://localhost:8000/mcp")
    print("🌐 Health check available at: http://localhost:8000/health")
    local_mcp.run(transport="http", port=8000, host="0.0.0.0")
