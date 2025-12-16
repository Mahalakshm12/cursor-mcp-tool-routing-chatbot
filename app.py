"""
Simple chat example using MCPAgent with built-in conversation memory.
"""

import asyncio
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from mcp_use import MCPAgent, MCPClient


def should_use_tools(user_input: str) -> bool:
    """
    Decide when MCP tools (browser/search) are required.
    """
    keywords = [
        "open",
        "browser",
        "website",
        "search",
        "find",
        "navigate",
        "click",
        "go to"
    ]
    return any(keyword in user_input.lower() for keyword in keywords)


async def run_memory_chat():
    load_dotenv()
    os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

    config_file = "browser_mcp.json"

    print("Initializing chat...")
# MCP client
    client = MCPClient.from_config_file(config_file)

# Groq LLM These lines create the LLM agent brain that reasons and decides what to do.
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0
    )

    # MCP Agent with memory
    agent = MCPAgent(
        llm=llm,
        client=client,
        max_steps=15,
        memory_enabled=True,
    )

    print("\n===== Interactive MCP Chat =====")
    print("Type 'exit' or 'quit' to end the conversation")
    print("Type 'clear' to clear conversation history")
    print("================================\n")

    try:
        while True:
            user_input = input("\nYou: ").strip()

            if user_input.lower() in ["exit", "quit"]:
                print("Ending conversation...")
                break

            if user_input.lower() == "clear":
                agent.clear_conversation_history()
                print("Conversation history cleared.")
                continue

            print("\nAssistant:", end=" ", flush=True)

            try:
                # 🔑 TOOL ROUTING LOGIC
                if should_use_tools(user_input):
                    response = await agent.run(user_input)
                else:
                    # Direct LLM answer 
                    response = llm.invoke(user_input).content
                    print(response)

            except Exception as e:
                print(f"\nError: {e}")

    finally:
        if client and client.sessions:
            await client.close_all_sessions()


if __name__ == "__main__":
    asyncio.run(run_memory_chat())