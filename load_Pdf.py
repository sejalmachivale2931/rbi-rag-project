from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# Step 1 - Load
loader = PyPDFLoader("test.pdf")
documents = loader.load()
print(f"Total Pages: {len(documents)}")

# Step 2 - Chunk
chunks = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=50
).split_documents(documents)
print(f"Total Chunks: {len(chunks)}")

# Step 3 - Embeddings + ChromaDB
print("\nEmbeddings ban rahi hain...")
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

# Step 4 - Groq LLM
llm = ChatGroq(
    api_key="Enter your key here",
    model_name="llama-3.3-70b-versatile",
    temperature=0
)

# Step 5 - RAG Chain (modern method)
prompt = ChatPromptTemplate.from_template("""
Sirf neeche diye gaye context se answer do.
Agar answer context mein nahi hai toh kaho "Mujhe nahi pata".

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

# Chat loop
print("\nRAG ready! Questions poochho ('quit' likhke band karo)\n")
while True:
    question = input("Tumhara Question: ")
    if question.lower() == "quit":
        break
    answer = chain.invoke(question)
    print(f"\nAnswer: {answer.content}\n")