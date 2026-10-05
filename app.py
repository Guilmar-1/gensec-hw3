"""Run a LangChain agent with Python and HW2 course-document tools."""

import os

from langchain.agents import create_agent
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
    """Start an interactive agent with Python and HW2 RAG tools."""
    llm = create_llm()
    python_repl = PythonREPLTool()
    tools = [python_repl, hw2_rag]

    # The model selects a tool based on whether the request needs computation
    # or information from the indexed course documents.
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=(
            "You are a helpful assistant. Use PythonREPL for calculations, "
            "numerical work, or short Python programs. Use the HW2 RAG tool "
            "for questions that require information from the indexed HW2/course "
            "documents. If neither tool is needed, answer directly. Explain "
            "results clearly."
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
            result = agent.invoke(
                {"messages": [{"role": "user", "content": user_input}]}
            )
            answer = result["messages"][-1].content
            # Provider responses may be plain text or a list of text blocks.
            if isinstance(answer, list):
                answer = "".join(
                    block.get("text", "") if isinstance(block, dict) else str(block)
                    for block in answer
                )
            print(f"assistant> {answer}")
        except Exception as error:
            print(f"Sorry, the request failed: {error}")


if __name__ == "__main__":
    run_agent()
