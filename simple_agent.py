import os, bs4, sys, time
from dotenv import load_dotenv
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.sqlite import SqliteSaver

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

model = ChatOpenAI(model = "gpt-4o", 
                   temperature = 0,
                   max_tokens = 256)

loader = WebBaseLoader(
    web_paths = ("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs = dict(parse_only = bs4.SoupStrainer(class_ = ("post-content", "post-title", "post-header"))))

docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()


### Build retriever tool ###
tool = create_retriever_tool(
    retriever,
    "blog_post_retriever",
    "Searches and returns excerpts from the Autonomous Agents blog post.",
)
tools = [tool]

memory = SqliteSaver.from_conn_string(":memory:")

agent_executor = create_react_agent(model, tools, checkpointer = memory)

config = {"configurable": {"thread_id": "abc123"}}

while True:
    question = input("Enter your question: ")
    if question in ['exit', 'quit']:
        print("Thanks. Goodbye!")
        sys.exit()
    else:
        for s in agent_executor.stream({"messages": question}, config = config).content:
            print(s)
        print("\n")