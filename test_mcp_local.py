"""
Test script for the local MCP server.
"""
import asyncio
import json
import httpx
from fastmcp import Client

async def test_local_mcp():
    """Test the local MCP server."""
    local_url = "http://localhost:8000"
    mcp_endpoint = f"{local_url}/mcp/"
    
    print(f"🚀 Testing local MCP server at {mcp_endpoint}")
    
    try:
        # Test health check first
        async with httpx.AsyncClient(timeout=10.0) as client:
            health_response = await client.get(f"{local_url}/health")
            print(f"✅ Health check: {health_response.status_code}")
            print(json.dumps(health_response.json(), indent=2))
        
        # Test MCP tool
        async with Client(mcp_endpoint) as client:
            print("✅ Connected to local MCP server")
            
            test_question = "What is the definition of an agent?"
            print(f"   Testing question: {test_question}")
            
            result = await client.call_tool("get_relevant_chunks_local", {
                "question": test_question,
                "top_k": 3
            })
            
            print("\n✅ Local Server Response:")
            print(json.dumps(result.data, indent=2, ensure_ascii=False))

    except Exception as e:
        print(f"❌ Error: {type(e).__name__} - {e}")
        print("\nMake sure the local MCP server is running with:")
        print("python -m modal run src/modal_mcp_server.py")

if __name__ == "__main__":
    print("🧪 Testing Local MCP Server...")
    asyncio.run(test_local_mcp())
    print("\n✅ Local test completed!") 