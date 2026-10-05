"""Run a simple LangChain agent with access to a Python REPL."""

import os

from langchain.agents import create_agent
from langchain_experimental.tools import PythonREPLTool
from langchain_google_genai import ChatGoogleGenerativeAI


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
    """Start an interactive agent that can use Python to answer questions."""
    llm = create_llm()
    python_repl = PythonREPLTool()

    # Supplying the REPL as a tool lets the model choose when code will help.
    agent = create_agent(
        model=llm,
        tools=[python_repl],
        system_prompt=(
            "You are a helpful assistant with access to a Python REPL. "
            "Use the Python tool when calculations or short programs will help "
            "answer the user's request. Explain the result clearly."
        ),
    )

    print("Python REPL agent ready. Enter a question, or press Enter to quit.")
    print(f"Available tool: {python_repl.name} — {python_repl.description}")

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
            print(f"assistant> {result['messages'][-1].content}")
        except Exception as error:
            print(f"Sorry, the request failed: {error}")


if __name__ == "__main__":
    run_agent()
