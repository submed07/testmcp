"""
MCP test client — connects to the running FastAPI/MCP server via StreamableHTTP.

Usage:
    # 1. Start the server first:
    #    uv run uvicorn api:app --reload --port 8001

    # 2. Run this client:
    #    uv run python mcp_client.py
"""

import asyncio
import os
from dotenv import load_dotenv
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

load_dotenv()

PORT = int(os.getenv("PORT", 8001))
SERVER_URL = f"http://localhost:{PORT}/mcp"


async def main():
    print(f"Connecting to MCP server at {SERVER_URL} ...\n")

    async with streamable_http_client(SERVER_URL) as (read, write, _):
        async with ClientSession(read, write) as session:

            # ── 1. Handshake ─────────────────────────────────────────────────
            await session.initialize()
            print("Session initialized.\n")

            # ── 2. Discover tools ────────────────────────────────────────────
            tools_response = await session.list_tools()
            print("Available tools:")
            for tool in tools_response.tools:
                print(f"  • {tool.name}: {tool.description}")
            print()

            # ── 3. Call analyse_adf_pipeline ─────────────────────────────────
            run_id = os.getenv("ADF_RUN_ID", "")
            if run_id:
                print(f"\nCalling analyse_adf_pipeline with run_id: '{run_id}'")

                adf_result = await session.call_tool(
                    "analyse_adf_pipeline",
                    {"pipeline_run_id": run_id},
                )

                print("ADF Analysis:")
                for content in adf_result.content:
                    print(f"  {content.text}")
            else:
                print("\nSkipping analyse_adf_pipeline — set ADF_RUN_ID in .env to test.")


if __name__ == "__main__":
    asyncio.run(main())
