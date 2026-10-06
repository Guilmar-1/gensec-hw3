"""Run a LangChain agent with Python and HW2 course-document tools."""

import os

from langchain.agents import create_agent
from langchain_community.tools.arxiv.tool import ArxivAPIWrapper, ArxivQueryRun
from langchain_community.tools.wikipedia.tool import (
    WikipediaAPIWrapper,
    WikipediaQueryRun,
)
from langchain_experimental.tools import PythonREPLTool
from langchain_google_genai import ChatGoogleGenerativeAI
from rag_tool import hw2_rag


def create_llm() -> ChatGoogleGenerativeAI:
    """Create the Google Gemini chat model using environment configuration.

    Set ``GOOGLE_API_KEY`` and ``GOOGLE_MODEL`` in the environment before
    running the application. The API key is read by the provider integration.
    """
    model_name = os.getenv("GOOGLE_MODEL")
    if not model_name:
        raise ValueError("Set the GOOGLE_MODEL environment variable first.")

    return ChatGoogleGenerativeAI(model=model_name)


def run_agent() -> None:
    """Start an interactive agent with Python, RAG, Wikipedia, and ArXiv tools."""
    llm = create_llm()
    python_repl = PythonREPLTool()
    arxiv_search = ArxivQueryRun(
        api_wrapper=ArxivAPIWrapper(top_k_results=3),
        description=(
            "Search ArXiv for academic papers by topic or query. Returns paper "
            "titles, authors, abstracts, and links. Use for requests to find "
            "or summarize research papers."
        ),
    )
    wikipedia_search = WikipediaQueryRun(
        api_wrapper=WikipediaAPIWrapper(top_k_results=2)
    )
    tools = [python_repl, hw2_rag, wikipedia_search, arxiv_search]

    # The model chooses among tools based on what kind of information is needed.
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=(
            "You are a helpful assistant. Use PythonREPL for calculations, "
            "numerical work, or short Python programs. Use the HW2 RAG tool "
            "for questions that require information from the indexed HW2/course "
            "documents. Use Wikipedia for general encyclopedia lookups. Use "
            "ArXiv search when the user asks to find academic papers or research "
            "on a topic, and summarize the most relevant results. If no tool is "
            "needed, answer directly. Explain results clearly."
        ),
    )

    print("HW3 agent ready. Enter a question, or press Enter to quit.")
    for tool in tools:
        print(f"Available tool: {tool.name} — {tool.description}")

    while True:
        try:
            user_input = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_input:
            break

        try:
            # The agent handles any tool calls and returns the conversation state.
            answer = ""
            # Stream agent updates so tool selections and returned data are visible.
            for update in agent.stream(
                {"messages": [{"role": "user", "content": user_input}]},
                stream_mode="updates",
            ):
                if "model" in update:
                    for message in update["model"].get("messages", []):
                        for tool_call in getattr(message, "tool_calls", []):
                            print(
                                f"🔧 Tool call: {tool_call['name']} "
                                f"Args: {tool_call['args']}"
                            )
                        if not getattr(message, "tool_calls", []):
                            answer = _content_to_text(message.content)

                if "tools" in update:
                    for message in update["tools"].get("messages", []):
                        tool_result = _content_to_text(message.content)
                        # Keep terminal tracing readable when a tool returns
                        # several long abstracts or encyclopedia excerpts.
                        preview = tool_result[:1200]
                        if len(tool_result) > len(preview):
                            preview += " ... [truncated]"
                        print(f"✅ Tool result ({message.name}): {preview}")
            print(f"assistant> {answer}")
        except Exception as error:
            print(f"Sorry, the request failed: {error}")


def _content_to_text(content: object) -> str:
    """Convert plain text or provider text blocks into printable text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    return str(content)


if __name__ == "__main__":
    run_agent()
