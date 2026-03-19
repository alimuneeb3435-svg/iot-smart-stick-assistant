import os
import streamlit as st

os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUNBUFFERED"] = "1"

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, END
from typing import TypedDict
from dotenv import load_dotenv

load_dotenv()

# ── PAGE CONFIG ───────────────────────────────────────
st.set_page_config(
    page_title="IoT Smart Stick — Document Assistant",
    page_icon="🦯",
    layout="centered"
)

st.title("🦯 IoT Smart Stick — Document Assistant")
st.markdown("Ask any question about the IoT-enabled intelligent stick project.")
st.divider()

# ── LLMs ──────────────────────────────────────────────
@st.cache_resource
def load_llms():
    llm = ChatGroq(
        temperature=0.3,
        model_name="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY")
    )
    grader_llm = ChatGroq(
        temperature=0.0,
        model_name="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY")
    )
    return llm, grader_llm

llm, grader_llm = load_llms()

# ── BUILD OR LOAD VECTOR DATABASE ────────────────────
@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    chroma_path = "./chroma_db"
    pdf_path = "document.pdf"

    # If chroma_db doesn't exist, build it from the PDF
    if not os.path.exists(chroma_path) or not os.listdir(chroma_path):
        st.info("⚙️ Building knowledge base from document... This may take a minute on first run.")

        if not os.path.exists(pdf_path):
            st.error(f"❌ '{pdf_path}' not found in the repo. Please push your PDF to GitHub.")
            st.stop()

        loader = PyPDFLoader(pdf_path)
        pages = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )
        chunks = splitter.split_documents(pages)

        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=chroma_path
        )
        vectorstore.persist()
        st.success("✅ Knowledge base built successfully!")
    else:
        vectorstore = Chroma(
            persist_directory=chroma_path,
            embedding_function=embeddings
        )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    return retriever

retriever = load_vectorstore()

# ── STATE ─────────────────────────────────────────────
class AgentState(TypedDict):
    question:      str
    rewritten:     str
    context:       str
    answer:        str
    relevance:     str
    hallucination: str
    source:        str
    retry_count:   int

# ── NODES ─────────────────────────────────────────────
def query_rewriter_node(state: AgentState) -> AgentState:
    response = llm.invoke([
        SystemMessage(content="""You are a query rewriter.
Rewrite the question for better document retrieval.
Rules:
- Do NOT change the meaning
- Do NOT add new concepts
- Keep original terms like IoT stick, sensors, etc.
- Fix spelling and grammar only
Return ONLY the rewritten question."""),
        HumanMessage(content=state["question"])
    ])
    return {
        "question":      state["question"],
        "rewritten":     response.content.strip(),
        "context":       "",
        "answer":        "",
        "relevance":     "",
        "hallucination": "",
        "source":        "",
        "retry_count":   state.get("retry_count", 0)
    }

def retriever_node(state: AgentState) -> AgentState:
    results1 = retriever.invoke(state["question"])
    results2 = retriever.invoke(state["rewritten"])
    all_results = results1 + results2
    seen = set()
    unique_results = []
    for doc in all_results:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            unique_results.append(doc)
    context = "\n\n".join([
        f"[Chunk {i+1}] {doc.page_content}"
        for i, doc in enumerate(unique_results[:4])
    ])
    return {**state, "context": context}

def grader_node(state: AgentState) -> AgentState:
    response = grader_llm.invoke([
        SystemMessage(content="""You are a relevance grader.
- If the context contains ANY partial information related to the question → say "relevant"
- Only say "not_relevant" if completely unrelated
Reply ONLY: relevant OR not_relevant"""),
        HumanMessage(content=f"Question: {state['question']}\n\nContext:\n{state['context']}")
    ])
    relevance = response.content.strip().lower()
    if relevance not in ["relevant", "not_relevant"]:
        relevance = "relevant"
    return {**state, "relevance": relevance}

def after_grader(state: AgentState) -> str:
    return "generator"

def generator_node(state: AgentState) -> AgentState:
    response = llm.invoke([
        SystemMessage(content="""You are a precise QA assistant.
Rules:
- Extract the exact answer from the context
- ALWAYS include specific names/entities if present
- If the answer is not clearly available, say: This information is not available in the document.
- Do NOT include contradictory statements."""),
        HumanMessage(content=f"Context:\n{state['context']}\n\nQuestion: {state['question']}")
    ])
    return {**state, "answer": response.content, "source": "document"}

def hallucination_checker_node(state: AgentState) -> AgentState:
    if "not available in the document" in state["answer"].lower():
        return {**state, "hallucination": "supported"}
    response = grader_llm.invoke([
        SystemMessage(content="""You are a hallucination checker.
- Verify the answer is fully supported by the context
- If ANY part is unsupported → not_supported
Reply ONLY: supported OR not_supported"""),
        HumanMessage(content=f"Context:\n{state['context']}\n\nAnswer:\n{state['answer']}")
    ])
    result = response.content.strip().lower()
    if result not in ["supported", "not_supported"]:
        result = "supported"
    return {**state, "hallucination": result}

def after_hallucination_check(state: AgentState) -> str:
    if state["hallucination"] == "supported":
        return "output"
    state["retry_count"] += 1
    if state["retry_count"] >= 2:
        return "output"
    return "generator"

def output_node(state: AgentState) -> AgentState:
    return state

# ── BUILD GRAPH ───────────────────────────────────────
@st.cache_resource
def build_agent():
    graph = StateGraph(AgentState)
    graph.add_node("query_rewriter",        query_rewriter_node)
    graph.add_node("retriever",             retriever_node)
    graph.add_node("grader",               grader_node)
    graph.add_node("generator",            generator_node)
    graph.add_node("hallucination_checker", hallucination_checker_node)
    graph.add_node("output",               output_node)

    graph.set_entry_point("query_rewriter")
    graph.add_edge("query_rewriter", "retriever")
    graph.add_edge("retriever",      "grader")
    graph.add_conditional_edges("grader", after_grader, {"generator": "generator"})
    graph.add_edge("generator",  "hallucination_checker")
    graph.add_conditional_edges("hallucination_checker", after_hallucination_check, {"output": "output", "generator": "generator"})
    graph.add_edge("output", END)
    return graph.compile()

agent = build_agent()

# ── CHAT INTERFACE ────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("source") == "document":
            st.caption("📄 Source: Document")

question = st.chat_input("Ask a question about the IoT Smart Stick...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = agent.invoke({
                "question":      question,
                "rewritten":     "",
                "context":       "",
                "answer":        "",
                "relevance":     "",
                "hallucination": "",
                "source":        "",
                "retry_count":   0
            })

        answer = result["answer"]
        st.markdown(answer)
        st.caption("📄 Source: Document")

    st.session_state.messages.append({
        "role":    "assistant",
        "content": answer,
        "source":  "document"
    })
