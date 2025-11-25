import os
from typing import List, Dict, Any, Optional

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


# --- Pydantic Data Models ---

class GraphState(BaseModel):
    """
    Represents the state of our graph.

    Attributes:
        question: The user's question to be answered
        generation: LLM generated answer
        documents: List of retrieved document texts
        loop_step: Number of query transformation iterations
    """
    question: str = Field(description="The user's question to be answered")
    generation: Optional[str] = Field(default=None, description="LLM generated answer")
    documents: List[str] = Field(default_factory=list, description="List of retrieved document texts")
    loop_step: int = Field(default=0, description="Number of query transformation iterations")
    
    class Config:
        arbitrary_types_allowed = True


class RetrievalConfig(BaseModel):
    """Configuration for document retrieval."""
    
    db_path: str = Field(default="chroma_db", description="Path to ChromaDB database")
    collection_name: str = Field(default="my_collection", description="Name of the collection")
    similarity_top_k: int = Field(default=3, ge=1, le=10, description="Number of top similar documents to retrieve")
    embed_model_name: str = Field(default="BAAI/bge-small-en-v1.5", description="Name of the embedding model")
    
    class Config:
        arbitrary_types_allowed = True


class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'",
        pattern="^(yes|no)$"
    )


class QueryTransformation(BaseModel):
    """Model for transformed query."""
    
    original_query: str = Field(description="The original user question")
    transformed_query: str = Field(description="The improved/transformed question")
    transformation_reason: Optional[str] = Field(
        default=None, 
        description="Reason for the transformation"
    )


class GenerationResult(BaseModel):
    """Model for the final generated answer."""
    
    answer: str = Field(description="The generated answer to the question")
    source_documents_count: int = Field(ge=0, description="Number of source documents used")
    confidence: Optional[str] = Field(
        default=None, 
        description="Confidence level of the answer"
    )


# --- Nodes ---

def retrieve(state: GraphState) -> Dict[str, Any]:
    """
    Retrieve documents from the vector store.

    Args:
        state (GraphState): The current graph state

    Returns:
        dict: Dictionary with documents and question to update state
    """

    print("\n" + "="*50)
    print("🔍 Retrieving Documents...")
    question = state.question

    # Use default config
    config = RetrievalConfig()
    
    # Initialize ChromaDB client and collection
    client = chromadb.PersistentClient(path=config.db_path)
    collection = client.get_or_create_collection(config.collection_name)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    
    # Load index
    embed_model = HuggingFaceEmbedding(model_name=config.embed_model_name)
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        embed_model=embed_model,
    )
    
    retriever = index.as_retriever(similarity_top_k=config.similarity_top_k)
    documents = retriever.retrieve(question)
    
    # Extract text from nodes
    docs_text = [d.get_content() for d in documents]
    
    return {"documents": docs_text, "question": question}


def grade_documents(state: GraphState) -> Dict[str, Any]:
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (GraphState): The current graph state

    Returns:
        dict: Updates documents key with only filtered relevant documents
    """
    print("\n" + "="*50)
    print("⚖️  Grading Documents...")
    question = state.question
    documents = state.documents
    
    # LLM with function call
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    structured_llm_grader = llm.with_structured_output(GradeDocuments)
    
    # Prompt
    system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
        ]
    )
    
    grader = grade_prompt | structured_llm_grader
    
    filtered_docs = []
    for d in documents:
        score = grader.invoke({"question": question, "document": d})
        grade = score.binary_score
        # print(f"  - Document Grade: {grade}") # Optional: Uncomment for detailed logs
        
        if grade == "yes":
            filtered_docs.append(d)
        else:
            continue
            
    print(f"  ✅ Kept {len(filtered_docs)} relevant documents")
    return {"documents": filtered_docs, "question": question}


def generate(state: GraphState) -> Dict[str, Any]:
    """
    Generate answer using LLM.

    Args:
        state (GraphState): The current graph state

    Returns:
        dict: New key added to state, generation, that contains LLM generation
    """

    print("\n" + "="*50)
    print("💡 Generating Answer...")
    question = state.question
    documents = state.documents
    
    # Prompt
    prompt = ChatPromptTemplate.from_template(
        """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
        Question: {question} 
        Context: {context} 
        Answer:"""
    )
    
    # LLM
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    
    # Chain
    rag_chain = prompt | llm | StrOutputParser()
    
    # Run
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}


def transform_query(state: GraphState) -> Dict[str, Any]:
    """
    Transform the query to produce a better question.

    Args:
        state (GraphState): The current graph state

    Returns:
        dict: Updates question key with a re-phrased question
    """

    print("\n" + "="*50)
    print("🔄 Transforming Query...")
    question = state.question
    documents = state.documents
    
    # Create a prompt
    system = """You a question re-writer that converts an input question to a better version that is optimized \n 
     for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning."""
    re_write_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Here is the initial question: \n\n {question} \n Formulate an improved question."),
        ]
    )
    
    # LLM
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    
    # Chain
    question_rewriter = re_write_prompt | llm | StrOutputParser()
    
    better_question = question_rewriter.invoke({"question": question})
    print(f"  ✨ New Question: {better_question}")
    
    loop_step = state.loop_step + 1
    
    return {"documents": documents, "question": better_question, "loop_step": loop_step}


def decide_to_generate(state: GraphState) -> str:
    """
    Determines whether to generate an answer, or re-generate a question.

    Args:
        state (GraphState): The current graph state

    Returns:
        str: Binary decision for next node to call
    """
    
    # print("---ASSESS GRADED DOCUMENTS---") # Reduced verbosity
    filtered_documents = state.documents
    loop_step = state.loop_step
    
    if loop_step >= 3:
        print("  ⚠️  Max retries reached. Proceeding to generation.")
        return "generate"
    
    if not filtered_documents:
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print("  ❌ No relevant documents found. Re-generating query.")
        return "transform_query"
    else:
        # We have relevant documents, so generate answer
        print("  ✅ Relevant documents found. Proceeding to generation.")
        return "generate"


# --- Graph Construction ---
workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)
workflow.add_node("transform_query", transform_query)

# Build graph
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "generate": "generate",
    },
)
workflow.add_edge("transform_query", "retrieve")
workflow.add_edge("generate", END)

# Compile
app = workflow.compile()
