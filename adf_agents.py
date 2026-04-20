"""Azure Data Factory Analysis Agent — LangGraph implementation.

Graph topology:

    START → agent ──── tool_calls? ──→ tools → agent
                  └─── no ──────────→ END

The agent node calls the LLM with tools bound.  If the model issues a
tool call, ToolNode executes get_pipeline_hierarchy and injects the result
back as a ToolMessage.  The model then sees that output and either calls
another tool or produces a final answer.

Usage:
    # CLI (requires ADF_RUN_ID in .env)
    uv run python adf_agents.py

    # Programmatic — sync
    from adf_agents import build_adf_agent, invoke_agent
    agent = build_adf_agent()
    print(invoke_agent("Why did run <run-id> fail?", agent=agent))

    # Programmatic — async
    from adf_agents import build_adf_agent, ainvoke_agent
    agent = build_adf_agent()
    print(await ainvoke_agent("Summarise run <run-id>", agent=agent))
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Annotated

from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict

from adf_tools import get_async_adf_client

logger = logging.getLogger(__name__)


# =============================================================================
# LangChain tool — thin wrapper so the LLM can call the ADF client
# =============================================================================


@tool
async def get_pipeline_hierarchy(pipeline_run_id: str) -> str:
    """Get the complete hierarchy of an Azure Data Factory pipeline run.

    Retrieves the full execution tree including:
    - Parent / child pipeline relationships (root is auto-discovered)
    - Every activity run with status, duration, and error detail
    - A _summary block with counts by status (succeeded / failed / in_progress)
    - A _metadata block identifying the root run and whether the given ID is root

    Args:
        pipeline_run_id: ADF pipeline run ID — parent or child, the tool finds the root.

    Returns:
        JSON string of the full hierarchy.
    """
    # async def lets LangGraph's ToolNode await this properly whether the graph
    # is invoked via agent.invoke() (sync) or agent.ainvoke() (async).
    # Avoids "This event loop is already running" when called from FastAPI/MCP.
    result = await get_async_adf_client().get_pipeline_hierarchy(pipeline_run_id)
    return json.dumps(result, default=str)


ADF_TOOLS = [get_pipeline_hierarchy]


# =============================================================================
# Agent state
# =============================================================================


class ADFAgentState(TypedDict):
    """Minimal state: just a list of messages with a reducer that appends."""

    messages: Annotated[list, add_messages]


# =============================================================================
# System prompt
# =============================================================================

_SYSTEM_PROMPT = """\
You are an expert Azure Data Factory (ADF) analyst.
Your job is to diagnose pipeline run issues, explain failures, and summarise execution status.

When given a pipeline run ID:
1. Call get_pipeline_hierarchy to retrieve the full execution tree.
2. Analyse the returned JSON:
   - _summary gives a quick count of succeeded / failed / in_progress pipelines.
   - Each pipeline node has status, message, durationInMs, and an activities list.
   - Failed activities carry an error field — that is the root cause.
   - childPipelines lists any Execute Pipeline activities that spawned child runs.
3. Respond with a structured summary:
   - Overall outcome (all succeeded / some failed / still running)
   - Which pipelines and activities failed, and the exact error message
   - Any pipelines still in progress
   - Performance notes (slow activities) when relevant

Be concise. Use bullet points. Quote error messages verbatim."""


# =============================================================================
# Graph construction
# =============================================================================


def _default_llm():
    """Auto-detect Azure OpenAI vs standard OpenAI from env vars and return the LLM.

    Azure OpenAI  → set AZURE_OPENAI_ENDPOINT  (required) +
                       AZURE_OPENAI_API_KEY or OPENAI_API_KEY +
                       OPENAI_MODEL (deployment name, e.g. "gpt-4o") +
                       OPENAI_API_VERSION (optional, default 2024-08-01-preview)

    Standard OpenAI → set OPENAI_API_KEY (starts with sk-)  +
                          OPENAI_MODEL (e.g. "gpt-4o-mini")

    Raises ImportError if langchain-openai is not installed.
    """
    try:
        from langchain_openai import AzureChatOpenAI, ChatOpenAI
    except ImportError as exc:
        raise ImportError(
            "langchain-openai is required. Install with: uv add langchain-openai"
        ) from exc

    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    if azure_endpoint:
        # Azure OpenAI — OPENAI_API_KEY is accepted as a fallback for AZURE_OPENAI_API_KEY
        return AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
            or os.getenv("OPENAI_MODEL", "gpt-4o"),
            azure_endpoint=azure_endpoint,
            api_key=os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY"),
            api_version=os.getenv("OPENAI_API_VERSION", "2024-08-01-preview"),
            temperature=0,
        )

    # Standard OpenAI
    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0,
    )


def build_adf_agent(llm=None):
    """Build and compile the ADF LangGraph agent.

    Args:
        llm: Any LangChain chat model that supports tool/function calling.
             Defaults to ChatOpenAI(gpt-4o-mini) — requires OPENAI_API_KEY in env.

    Returns:
        A compiled LangGraph (CompiledStateGraph) ready for invoke / ainvoke.

    Graph:
        START → agent → [tools → agent]* → END
    """
    if llm is None:
        llm = _default_llm()

    # Bind the ADF tools so the model knows it can call them
    llm_with_tools = llm.bind_tools(ADF_TOOLS)

    # ── Node: agent ───────────────────────────────────────────────────────────

    def agent_node(state: ADFAgentState) -> dict:
        """Call the LLM. Prepend the system prompt on the very first turn."""
        messages = state["messages"]

        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=_SYSTEM_PROMPT), *messages]

        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    # ── Node: tools ───────────────────────────────────────────────────────────

    # ToolNode inspects the last AIMessage for tool_calls, runs each one, and
    # returns a ToolMessage per result back into the messages list.
    tool_node = ToolNode(ADF_TOOLS)

    # ── Assemble graph ────────────────────────────────────────────────────────

    graph = StateGraph(ADFAgentState)

    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "agent")

    # tools_condition: routes to "tools" if the last message has tool_calls,
    # otherwise routes to END
    graph.add_conditional_edges("agent", tools_condition)

    # After executing tools, always return to the agent for interpretation
    graph.add_edge("tools", "agent")

    return graph.compile()


# =============================================================================
# Convenience helpers
# =============================================================================


def invoke_agent(query: str, agent=None) -> str:
    """Run the agent with a plain-text query; return the final answer as a string.

    Args:
        query: Natural-language question, e.g. "Why did run abc-123 fail?"
        agent: Pre-built compiled agent. A new one is created if not provided.

    Returns:
        The agent's final text response.
    """
    if agent is None:
        agent = build_adf_agent()

    result = agent.invoke({"messages": [HumanMessage(content=query)]})
    return result["messages"][-1].content


async def ainvoke_agent(query: str, agent=None) -> str:
    """Async version of invoke_agent — use this inside async contexts (FastAPI, etc.).

    Args:
        query: Natural-language question.
        agent: Pre-built compiled agent. A new one is created if not provided.

    Returns:
        The agent's final text response.
    """
    if agent is None:
        agent = build_adf_agent()

    result = await agent.ainvoke({"messages": [HumanMessage(content=query)]})
    return result["messages"][-1].content


# =============================================================================
# CLI entry point
# =============================================================================


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s  %(name)s: %(message)s",
    )

    run_id = os.getenv("ADF_RUN_ID", "")
    if run_id:
        query = (
            f"Analyse ADF pipeline run {run_id}."
            "Show the status, start time, end time, and duration for each pipeline and activity."
        )
    else:
        print(
            "Tip: set ADF_RUN_ID=<pipeline-run-id> in .env to analyse a real run.\n"
            "Running a demo query with a placeholder ID.\n"
        )
        run_id = "00000000-0000-0000-0000-000000000000"
        query = (
            f"Analyse ADF pipeline run {run_id}. "
            "What is the overall status and were there any failures?"
        )

    print(f"Query : {query}\n")
    print("─" * 60)

    agent = build_adf_agent()
    answer = await ainvoke_agent(query, agent=agent)

    print("Answer:")
    print(answer)


if __name__ == "__main__":
    asyncio.run(main())
