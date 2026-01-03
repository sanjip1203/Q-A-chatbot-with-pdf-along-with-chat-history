import os
import tempfile
import streamlit as st

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain


#  UI 
st.title("Conversational RAG with PDF + Chat History")
st.write("Upload a PDF and ask questions. The assistant remembers your conversation per session.")

groq_api_key = st.text_input("Enter your Groq API key:", type="password")
hf_token = st.text_input("Enter your Hugging Face token:", type="password")
session_id = st.text_input("Session ID", value="default_session")

uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"], accept_multiple_files=False)

with st.expander("Where do I get these keys?"):
    st.markdown(
        "- Groq API key: https://console.groq.com/keys\n"
        "- Hugging Face token: https://huggingface.co/settings/tokens"
    )

#  GATEKEEP 
if not groq_api_key or not hf_token:
    st.warning("Please enter BOTH Groq API key and Hugging Face token.")
    st.stop()

if uploaded_file is None:
    st.info("Upload a PDF to start.")
    st.stop()

#  STATE 
if "store" not in st.session_state:
    st.session_state.store = {}  # session_id -> ChatMessageHistory

def get_session_history(sid: str) -> BaseChatMessageHistory:
    if sid not in st.session_state.store:
        st.session_state.store[sid] = ChatMessageHistory()
    return st.session_state.store[sid]


#  MODELS 
os.environ["HF_TOKEN"] = hf_token  # must be set BEFORE creating embeddings

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.1-8b-instant",
    temperature=0
)


#  BUILD RAG 
def build_conversational_rag(llm, retriever):
    contextualize_prompt = ChatPromptTemplate.from_messages(
        [
            ("system",
             "Given the chat history and the latest user question (which may refer to earlier context), "
             "rewrite it as a standalone question that can be understood without the chat history. "
             "Do NOT answer the question. If rewriting isn't needed, return it as-is."
            ),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm=llm,
        retriever=retriever,
        prompt=contextualize_prompt
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system",
             "You are an assistant for question-answering tasks. "
             "Use the retrieved context to answer the question. "
             "If you don't know, say you don't know. "
             "Use at most three sentences and keep it concise.\n\n"
             "{context}"
            ),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    conversational = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    return conversational


#  VECTORSTORE  
file_fingerprint = f"{uploaded_file.name}-{uploaded_file.size}"

if st.session_state.get("vs_fingerprint") != file_fingerprint:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
    splits = splitter.split_documents(documents)

    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    st.session_state.vectorstore = vectorstore
    st.session_state.vs_fingerprint = file_fingerprint

retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 4})
conversational_rag_chain = build_conversational_rag(llm, retriever)


#  CHAT 
user_input = st.text_input("Your question:")

if user_input:
    response = conversational_rag_chain.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": session_id}},
    )
    st.success(response["answer"])

#  SHOW HISTORY 
history = get_session_history(session_id)

with st.sidebar:
    st.subheader("🕘 Chat History")

    if not history.messages:
        st.caption("No messages yet.")
    else:
        for msg in history.messages:
            if msg.type == "human":
                st.markdown(f"**🧑 You:** {msg.content}")
            else:
                st.markdown(f"**🤖 Assistant:** {msg.content}")

