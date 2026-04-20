"""
LangGraph agent — no LangSmith tracing.
"""

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


if __name__ == "__main__":
    graph = build_graph()
    result = graph.invoke({"message": "Hello World"})
    print(f"\nResult: {result['message']}")
