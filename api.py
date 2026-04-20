"""
FastAPI + MCP Server — exposes the ADF LangGraph agent over REST and MCP.

Layout
------
REST endpoints:
    GET  /            → API info & available endpoints
    GET  /health      → health check

MCP endpoint (StreamableHTTP transport):
    POST /mcp         → MCP message handler (bidirectional over a single endpoint)

MCP tools exposed:
    analyse_adf_pipeline(pipeline_run_id, query?) → runs the ADF agent and returns a summary

Run:
    uv run uvicorn api:app --reload --port 8000

Design notes
------------
* StreamableHTTP is the modern MCP transport (replaces legacy SSE split-endpoint).
* FastMCP's streamable_http_app() returns a Starlette sub-app whose default route
  is at "/mcp".  We mount the sub-app at "/" so that route stays at /mcp.
  (Mounting at "/mcp" would shift it to /mcp/mcp — wrong.)
* The StreamableHTTPSessionManager needs its run() context started before the
  first request arrives.  We wire that into FastAPI's lifespan explicitly,
  which is the pattern FastMCP documents for multi-server FastAPI apps.
"""

import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv
load_dotenv()

import uvicorn
from fastapi import FastAPI

from mcp.server.fastmcp import FastMCP

from adf_agents import ainvoke_agent


# ---------------------------------------------------------------------------
# MCP Server — FastMCP high-level API
# ---------------------------------------------------------------------------

mcp = FastMCP("langgraph-mcp-server")


@mcp.tool()
async def analyse_adf_pipeline(pipeline_run_id: str, query: str = "") -> str:
    """
    Analyse an Azure Data Factory pipeline run using the LangGraph ADF agent.

    Fetches the full pipeline hierarchy (parent + all child pipelines, every
    activity, status, errors, durations) and returns an LLM-generated summary.

    Args:
        pipeline_run_id: ADF pipeline run ID — parent or child run ID both work.
        query: Optional natural-language question. Defaults to a standard
               status + failure analysis if omitted.
    """
    if not query:
        query = (
            f"Analyse ADF pipeline run {pipeline_run_id}. "
            "Show the overall status, start time, end time, and duration for each pipeline and activity. "
            "Highlight any failures with their exact error messages."
        )
    return await ainvoke_agent(query)


# Build the MCP ASGI sub-app (this also lazily creates the session_manager).
# Must be done BEFORE the lifespan accesses mcp.session_manager.
_mcp_asgi_app = mcp.streamable_http_app()


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(_app: FastAPI):
    """
    Start the StreamableHTTP session manager before the first request.
    Without this the manager's internal task group is None and every
    request raises RuntimeError / returns a 404 that the client converts
    to 'Session terminated'.
    """
    async with mcp.session_manager.run():
        yield


app = FastAPI(
    title="ADF LangGraph MCP Server",
    description="Azure Data Factory LangGraph agent exposed via REST and MCP (StreamableHTTP)",
    version="0.1.0",
    lifespan=lifespan,
)

# Mount the MCP sub-app at "/" — its internal route is at "/mcp",
# so the full URL becomes http://host:port/mcp.
# Starlette checks explicit routes (/, /health) before the mount,
# so those endpoints are not shadowed.
app.mount("/", _mcp_asgi_app)


# ── REST endpoints ───────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "service": "ADF LangGraph MCP Server",
        "rest": {
            "GET  /":       "this info",
            "GET  /health": "health check",
        },
        "mcp": {
            "POST /mcp": "MCP StreamableHTTP endpoint — connect your MCP client here",
        },
    }


@app.get("/health")
def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    print(f"Starting server on http://0.0.0.0:{port}")
    print(f"  MCP  → http://localhost:{port}/mcp")
    print(f"  Docs → http://localhost:{port}/docs")
    uvicorn.run(app, host="0.0.0.0", port=port)
