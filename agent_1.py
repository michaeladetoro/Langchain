import os, sys, time
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import create_react_agent

load_dotenv()

# enviroment variables
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

# Create the agent
memory = SqliteSaver.from_conn_string(":memory:")
model = ChatOpenAI(model = "gpt-4o")
search = TavilySearchResults(max_results=2)
tools = [search]
agent_executor = create_react_agent(model, tools, checkpointer = memory)

# Use the agent
config = {"configurable": {"thread_id": "agent_1"}}
while True:
    question = input("Enter your question: ")
    if question in ['exit', 'quit']:
        print("Thanks. Goodbye!")
        sys.exit()
    else:
        for chunk in agent_executor.stream({"messages": question}, config = config):
            print(chunk, end="", flush=True)
            time.sleep(0.01)
        print("\n")