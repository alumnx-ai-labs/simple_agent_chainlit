import os
import sys

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv(override=True)

# --- DEBUG: confirm which key got loaded ---
key = os.getenv("GOOGLE_API_KEY")
if key:
    print(f"Loaded key: starts {key[:8]}... ends ...{key[-4:]}  (length {len(key)})")
else:
    print("No GOOGLE_API_KEY found in environment!")
# -------------------------------------------

# --- LangSmith tracing ---
# Enable by setting LANGSMITH_TRACING=true and LANGSMITH_API_KEY / LANGSMITH_PROJECT
# in your .env file. LangChain picks these env vars up automatically, so no extra
# code is needed here beyond loading them via dotenv above.


def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


def create_daily_thought() -> str:
    """Generate an inspirational daily thought using the LLM."""
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")
    response = llm.invoke("Generate a short, unique inspirational daily thought in one or two sentences.")
    return response.content


agent = create_agent(
    model="google_genai:gemini-3.5-flash-lite",
    tools=[get_weather, create_daily_thought],
    system_prompt="You are a helpful assistant. Make sure that you only respond with whatever is coming as input to the agent, and do not add any extra commentary or explanation.",
)


def extract_text(content) -> str:
    """Gemini returns message content as a list of blocks instead of a plain string."""
    if isinstance(content, list):
        return "".join(block.get("text", "") for block in content if isinstance(block, dict) and block.get("type") == "text")
    return content


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python agent.py "<your message>"')
        sys.exit(1)

    user_input = " ".join(sys.argv[1:])

    result = agent.invoke({"messages": [{"role": "user", "content": user_input}]})
    print(extract_text(result["messages"][-1].content))
