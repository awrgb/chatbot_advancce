import os
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import AIMessage, ToolMessage
from pydantic import BaseModel
import streamlit as st

tavily_api_key = "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"
memory = MemorySaver()

class State(TypedDict):
    messages: Annotated[list, add_messages]
    ask_human: bool

class RequestAssistance(BaseModel):
    request: str

tool = TavilySearchResults(max_results=2, api_wrapper={"tavily_api_key": tavily_api_key})
tools = [tool]
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", api_key="AIzaSyD6H9Mms-VXuT1WNS7kg9jlSVF-1H3qbfs")
llm_with_tools = llm.bind_tools(tools + [RequestAssistance])

def chatbot(state: State):
    response = llm_with_tools.invoke(state["messages"])
    ask_human = response.tool_calls and response.tool_calls[0]["name"] == RequestAssistance.__name__
    return {"messages": [response], "ask_human": ask_human}

graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", ToolNode(tools=[tool]))

def create_response(response: str, ai_message: AIMessage):
    return ToolMessage(content=response, tool_call_id=ai_message.tool_calls[0]["id"])

def human_node(state: State):
    new_messages = []
    if not isinstance(state["messages"][-1], ToolMessage):
        new_messages.append(create_response("No response from human.", state["messages"][-1]))
    return {"messages": new_messages, "ask_human": False}

graph_builder.add_node("human", human_node)

def select_next_node(state: State):
    return "human" if state["ask_human"] else tools_condition(state)

graph_builder.add_conditional_edges("chatbot", select_next_node, {"human": "human", "tools": "tools", END: END})
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge("human", "chatbot")
graph_builder.add_edge(START, "chatbot")
graph = graph_builder.compile(checkpointer=memory, interrupt_before=["human"])

st.title("LangGraph Chatbot")
user_input = st.text_input("User: ", "")

if user_input:
    config = {"configurable": {"thread_id": "1"}}
    events = graph.stream({"messages": [("user", user_input)]}, config, stream_mode="values")
    for event in events:
        if "messages" in event:
            st.write("Assistant:", event["messages"][-1].content)
