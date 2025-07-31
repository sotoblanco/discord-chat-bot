"""
Test script for the deployed MCP server.
"""
import os
import asyncio
import json
import httpx
from dotenv import load_dotenv
from fastmcp import Client

# Load environment variables from .env file
load_dotenv()

async def test_health_check(server_url: str):
    """Test the health check endpoint of the deployed server."""
    health_endpoint = f"{server_url.rstrip('/')}/health"
    print(f"\n🏥 Checking health at {health_endpoint}...")
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(health_endpoint)
            response.raise_for_status()
            print(f"✅ Health check successful: {response.status_code}")
            print(json.dumps(response.json(), indent=2))
            return True
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

async def test_mcp_tool(server_url: str):
    """Connects to the deployed MCP server and tests the get_relevant_chunks tool."""
    mcp_endpoint = f"{server_url.rstrip('/')}/mcp/"  # trailing slash is necessary
    print(f"\n🚀 Testing chunk retrieval MCP tool at {mcp_endpoint}...")
    
    try:
        async with Client(mcp_endpoint) as client:
            print("✅ Connected to MCP server")
            
            # Test question
            test_question = "What is the definition of an agent?"
            result = await client.call_tool("get_relevant_chunks", {
                "question": test_question,
                "top_k": 2
            })
            
            # Print the results
            print("\n✅ Server Response:")
            print(json.dumps(result.data, indent=2, ensure_ascii=False))

    except Exception as e:
        print(f"❌ MCP tool test failed: {e}")

async def main():
    """Main function to run the tests."""
    server_url = os.getenv("BWAI_MCP_SERVER_URL")
    if not server_url:
        print("Error: BWAI_MCP_SERVER_URL not found in .env file.")
        print("Please create a .env file and add the following line:")
        print("BWAI_MCP_SERVER_URL=<your-modal-app-url>")
        return

    if await test_health_check(server_url):
        await test_mcp_tool(server_url)

if __name__ == "__main__":
    print("🚀 Starting Deployed MCP Server Tests...")
    asyncio.run(main())
    print("\n✅ Tests completed!")
