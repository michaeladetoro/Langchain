import os, bs4, sys, time
from dotenv import load_dotenv
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

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

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", ("""Given a chat history and the latest user question which might reference context in the chat history, 
                      formulate a standalone question which can be understood without the chat history. 
                      Do NOT answer the question, just reformulate it if needed and otherwise return it as is.
                   """)),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

history_aware_retriever = create_history_aware_retriever(model, retriever, contextualize_q_prompt)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", ("""You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. 
                      If you don't know the answer, say that you don't know. Use three sentences maximum and keep the answer concise.
                      \n\n{context}
                   """)),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(model, prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

config = {"configurable": {"session_id": "2"}}

while True:
    question = input("Enter your question: ")
    if question in ['exit', 'quit']:
        print("Thanks. Goodbye!")
        sys.exit()
    else:
        for chunk in conversational_rag_chain.invoke({"input": question}, config = config)["answer"]:
            print(chunk, end = "", flush = True)
            time.sleep(0.01)
        print("\n")