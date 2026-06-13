import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# Page config
st.set_page_config(
    page_title="RBI Report Q&A",
    page_icon="🏦",
    layout="centered"
)

st.title("🏦 RBI Annual Report Q&A System")
st.caption("Ask any question from RBI Annual Report 2024")

# Load model only once
@st.cache_resource
def load_rag():
    loader = PyPDFLoader("test.pdf")
    documents = loader.load()

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100
    ).split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": False},
        cache_folder="C:\\Users\\Nikhil Dinesh Ghanek\\.cache\\huggingface\\hub"
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )

    llm = ChatGroq(
        api_key= os.environ.get("GROQ_API_KEY"),
        model_name="llama-3.3-70b-versatile",
        temperature=0
    )

    prompt = ChatPromptTemplate.from_template("""
    Answer only from the context below.
    If the answer is not in the context, say "I don't know".
    Context: {context}
    Question: {question}
    """)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
    )

    return chain, retriever

# Load RAG
with st.spinner("Loading RAG system..."):
    chain, retriever = load_rag()

st.success("System ready! Ask your question below.")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Input
question = st.chat_input("Ask a question about RBI Report...")

if question:
    # Show user question
    with st.chat_message("user"):
        st.write(question)
    st.session_state.messages.append({"role": "user", "content": question})

    # Get answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = chain.invoke(question)
            docs = retriever.invoke(question)
            source_page = docs[0].metadata.get("page", 0) + 1

        st.write(answer.content)
        st.caption(f"Source: Page {source_page}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer.content
    })
