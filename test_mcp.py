"""
Test script for the local MCP server.
Run this after starting the MCP server with: python src/modal_mcp.py
"""

import asyncio
import json
from fastmcp import Client

async def test_mcp_server():
    """Test the MCP server functionality"""
    print("🧪 Testing MCP Server...")
    
    try:
        # Connect to the local MCP server
        async with Client("http://localhost:8000/mcp") as client:
            print("✅ Connected to MCP server")
            
            # Test the get_relevant_chunks tool
            print("\n🔍 Testing chunk retrieval...")
            result = await client.call_tool("get_relevant_chunks", {
                "question": "What is RAG?", 
                "top_k": 3
            })
            
            print("📊 Response:")
            print(json.dumps(result.data, indent=2))
            
            # Test with a different question
            print("\n🔍 Testing with different question...")
            result2 = await client.call_tool("get_relevant_chunks", {
                "question": "How to debug RAG applications?", 
                "top_k": 2
            })
            
            print("📊 Response:")
            print(json.dumps(result2.data, indent=2))
            
    except Exception as e:
        print(f"❌ Error testing MCP server: {e}")
        print("Make sure the MCP server is running with: python src/modal_mcp.py")

async def test_health_check():
    """Test the health check endpoint"""
    import httpx
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:8000/health")
            print(f"\n🏥 Health check status: {response.status_code}")
            print(json.dumps(response.json(), indent=2))
    except Exception as e:
        print(f"❌ Health check failed: {e}")

if __name__ == "__main__":
    print("🚀 Starting MCP Server Tests...")
    
    # Run tests
    asyncio.run(test_health_check())
    asyncio.run(test_mcp_server())
    
    print("\n✅ Tests completed!") 