"""Chainlit Chat Interface for Simple LangChain Agent.

Launch with:
    chainlit run chat.py
"""

import chainlit as cl
from agent import agent, extract_text


@cl.on_chat_start
async def on_chat_start():
    """Initialize a new chat session."""
    await cl.Message(
        content=(
            "Hey there! I'm your Simple Agent powered by Gemini.\n\n"
            "I can help you with:\n"
            "- **Weather** - Ask me about the weather in any city\n"
            "- **Daily Inspiration** - Request a motivational thought or quote\n"
            "- **General Questions** - Math, facts, coding, and more\n\n"
            "Go ahead, type something!"
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    """Handle incoming user messages and stream the agent response."""
    # Show a thinking indicator
    msg = cl.Message(content="")
    await msg.send()

    try:
        # Invoke the LangChain agent
        result = await cl.make_async(agent.invoke)(
            {"messages": [{"role": "user", "content": message.content}]}
        )

        messages = result.get("messages", [])
        raw_output = messages[-1].content if messages else ""
        reply = extract_text(raw_output)

        # Stream the response token by token for a smooth UX
        msg.content = reply
        await msg.update()

    except Exception as e:
        error_text = str(e)
        if "RESOURCE_EXHAUSTED" in error_text or "429" in error_text:
            msg.content = (
                "I've hit the API rate limit. Please wait a few seconds and try again."
            )
        else:
            msg.content = f"Something went wrong: {error_text[:200]}"
        await msg.update()
