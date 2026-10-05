"""Expose the existing HW2 retrieval-augmented question chain as a tool."""

import os
from functools import lru_cache
from pathlib import Path

from langchain.tools import tool
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_vertexai import VertexAIEmbeddings


# Resolve this from the source file, so launching the app from another working
# directory still opens the HW2 database in its existing location.
HW2_CHROMA_DIRECTORY = (
    Path(__file__).resolve().parents[1]
    / "02_LangChain"
    / "hw2"
    / "rag_data"
    / ".chromadb"
)


def format_docs(documents: list) -> str:
    """Combine retrieved document text into the context used by the prompt."""
    return "\n\n".join(document.page_content for document in documents)


@lru_cache(maxsize=1)
def get_rag_chain():
    """Build and cache the HW2-style RAG chain over the existing Chroma store."""
    if not HW2_CHROMA_DIRECTORY.is_dir():
        raise FileNotFoundError(
            f"HW2 Chroma database was not found: {HW2_CHROMA_DIRECTORY}"
        )

    model_name = os.getenv("GOOGLE_MODEL")
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    if not model_name:
        raise ValueError("Set the GOOGLE_MODEL environment variable first.")
    if not project_id:
        raise ValueError("Set the GOOGLE_CLOUD_PROJECT environment variable first.")

    # Reopen the already populated HW2 store with the same embedding model.
    vectorstore = Chroma(
        persist_directory=str(HW2_CHROMA_DIRECTORY),
        embedding_function=VertexAIEmbeddings(
            model_name="gemini-embedding-001",
            project=project_id,
            location="us-west1",
        ),
    )
    retriever = vectorstore.as_retriever()
    llm = ChatGoogleGenerativeAI(model=model_name)
    prompt = ChatPromptTemplate.from_template(
        """You answer questions using the provided course-document context.
If the context does not contain the answer, say you do not know. Keep the
answer concise and use at most three sentences.

Question: {question}

Context: {context}

Answer:"""
    )

    # Retrieve matching documents, format their text, and ask the model to answer.
    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )


@tool
def hw2_rag(question: str) -> str:
    """Answer a question using information in the indexed HW2 course documents."""
    return get_rag_chain().invoke(question)
