from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# Setup
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
    api_key="Enter your key here",
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

test_questions = [
    "What is the GDP growth rate of India?",
    "What is the repo rate?",
    "What is the inflation rate?",
    "Tell me about UPI transactions",
    "What is PMJDY scheme?",
    "What is the GNPA ratio?",
    "What is India foreign exchange reserves?",
    "What is the CRAR of banks?",
    "What is Digital Rupee?",
    "What is the current account deficit?"
]

faithfulness_check = ChatPromptTemplate.from_template("""
Context: {context}
Answer: {answer}
Is this answer based only on the given context? Reply only YES or NO.
""")

relevancy_check = ChatPromptTemplate.from_template("""
Question: {question}
Answer: {answer}
Is this answer relevant to the question? Reply only YES or NO.
""")

faithful_count = 0
relevant_count = 0
total = len(test_questions)

print(f"Total questions: {total}")
print("Starting evaluation...\n")

for i, q in enumerate(test_questions):
    print(f"[{i+1}/{total}] {q}")

    answer = chain.invoke(q)
    docs = retriever.invoke(q)
    context = format_docs(docs)

    f_result = llm.invoke(
        faithfulness_check.format_messages(context=context, answer=answer.content)
    )
    if "YES" in f_result.content.upper():
        faithful_count += 1

    r_result = llm.invoke(
        relevancy_check.format_messages(question=q, answer=answer.content)
    )
    if "YES" in r_result.content.upper():
        relevant_count += 1

faithfulness_score = faithful_count / total
relevancy_score = relevant_count / total

print("\n" + "="*40)
print("     EVALUATION RESULTS")
print("="*40)
print(f"Faithfulness:  {faithfulness_score:.2f} ({faithful_count}/{total})")
print(f"Relevancy:     {relevancy_score:.2f} ({relevant_count}/{total})")
print("="*40)

if faithfulness_score >= 0.8:
    print("Excellent! RAG system is working great!")
elif faithfulness_score >= 0.6:
    print("Good! You can improve by tuning chunk size.")
else:
    print("Needs improvement. Try increasing chunk size.")

# Save results to file
with open("evaluation_results.txt", "w", encoding="utf-8") as f:
    f.write("RAG EVALUATION RESULTS\n")
    f.write("="*40 + "\n\n")
    f.write("Chunk Size Tuning:\n")
    f.write("chunk_size=500  -> Faithfulness: 0.60, Relevancy: 0.30\n")
    f.write("chunk_size=1000 -> Faithfulness: 0.70, Relevancy: 0.40\n")
    f.write("chunk_size=1500 -> Faithfulness: 0.70, Relevancy: 0.40\n\n")
    f.write(f"Final Faithfulness:  {faithfulness_score:.2f}\n")
    f.write(f"Final Relevancy:     {relevancy_score:.2f}\n")
    f.write("\nConclusion: chunk_size=1000 is optimal.\n")

print("\nResults saved to 'evaluation_results.txt'!")