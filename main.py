# %%
"""
Dummy LangGraph agent — used to test LangSmith tracing.
"""

import os
from dotenv import load_dotenv

# Must load before LangGraph/LangSmith imports so tracing env vars are set.
load_dotenv()

from typing import TypedDict
from langgraph.graph import StateGraph, END


class State(TypedDict):
    message: str


def hello_agent(state: State) -> dict:
    response = f"Hello from LangGraph! You said: '{state['message']}'"
    print(response)
    return {"message": response}


def build_graph():
    workflow = StateGraph(State)
    workflow.add_node("hello_agent", hello_agent)
    workflow.set_entry_point("hello_agent")
    workflow.add_edge("hello_agent", END)
    return workflow.compile()


def main():
    tracing = os.getenv("LANGCHAIN_TRACING_V2", "false")
    project  = os.getenv("LANGCHAIN_PROJECT", "default")
    api_key  = os.getenv("LANGCHAIN_API_KEY", "")

    print(f"LangSmith tracing : {'ENABLED' if tracing == 'true' else 'DISABLED'}")
    print(f"Project           : {project}")
    print(f"API key           : {'set' if api_key else 'NOT SET'}\n")

    graph = build_graph()
    result = graph.invoke({"message": "Hello World"})
    print(f"\nResult: {result['message']}")

    if tracing == "true":
        print(f"\nView trace at: https://smith.langchain.com/projects/{project}")


if __name__ == "__main__":
    main()
