"""
Local FastMCP server for workshop transcript chunk retrieval.
This can be run locally for testing before Modal deployment.
"""

import os
import sys
from pathlib import Path

# Add the src directory to Python path
sys.path.append(str(Path(__file__).parent))

from fastmcp import FastMCP
from starlette.responses import JSONResponse
from vector_emb import retrieve_relevant_chunks, get_chroma_client, get_or_create_collection

# Create FastMCP server instance
mcp = FastMCP("bwai-mcp-server")

@mcp.tool
def get_relevant_chunks(question: str, top_k: int = 10) -> dict:
    """Retrieve relevant transcript chunks for a given question"""
    try:
        # Get chunks from vector database
        chunks = retrieve_relevant_chunks(question, n_results=top_k)
        
        # Format response with metadata
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
    try:
        # Test ChromaDB connection
        client = get_chroma_client()
        collection = get_or_create_collection(client)
        count = collection.count()
        
        return JSONResponse({
            "status": "healthy", 
            "chroma_connected": True,
            "total_chunks": count
        })
    except Exception as e:
        return JSONResponse({
            "status": "unhealthy",
            "chroma_connected": False,
            "error": str(e)
        })

if __name__ == "__main__":
    print("🚀 Starting Discord MCP Server...")
    print("📚 Available tools:")
    print("  - get_relevant_chunks(question: str, top_k: int = 10)")
    print("🌐 Health check available at: http://localhost:8000/health")
    print("🔗 MCP endpoint available at: http://localhost:8000/mcp")
    print("\nPress Ctrl+C to stop the server")
    
    # Run the server locally
    mcp.run(transport="http", port=8000, host="0.0.0.0") 