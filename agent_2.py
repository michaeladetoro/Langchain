import sys, time, os
from dotenv import load_dotenv
from typing import Annotated
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
# os.environ["LANGCHAIN_PROJECT"] = "Customer Support Bot Tutorial"
# os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

tool = TavilySearchResults(max_results=10)
tools = [tool]
# llm = ChatAnthropic(model = "claude-3-5-sonnet-20240620")
llm = ChatOpenAI(model = "gpt-4o", temperature = 0)
llm_with_tools = llm.bind_tools(tools)
memory = SqliteSaver.from_conn_string(":memory:")

def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
# Any time a tool is called, we return to the chatbot to decide the next step
graph_builder.add_edge("tools", "chatbot")
graph_builder.set_entry_point("chatbot")
graph = graph_builder.compile(checkpointer = memory)

config = {"configurable": {"thread_id": "1a"}}

while True:
    user = input("Enter your question: ")
    if user in ['exit', 'quit']:
        print("Thanks. Goodbye!")
        sys.exit()
    else:
        for event in graph.stream({"messages": user}, config = config, stream_mode = "values"):
            for value in event.values():
                print(value["messages"][-1].content)
        print("\n")