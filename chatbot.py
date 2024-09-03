import os, sys, time
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a Health Assistant:"),
        ("user", "Question: {question}")
    ]
)
model = ChatOpenAI(model="gpt-4o")
parser = StrOutputParser()

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

chat_history = RunnableWithMessageHistory(model, get_session_history)
config = {"configurable": {"session_id": "1"}}

while True:
    question = input("Enter your question: ")
    if question in ['exit', 'quit']:
        print("Thanks for using the Health Assistant. Goodbye!")
        sys.exit()
    else:
        for chunk in chat_history.stream({"question": question}, config = config):
            print(chunk.content, end="", flush=True)
            time.sleep(0.05)
        print("\n")
