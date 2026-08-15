# Simple Agent

A minimal LangChain agent powered by Google's free Gemini API, with just two tools:

- 🌤️ **Weather Tool** – Retrieve weather information for any city
- 💡 **Daily Thought Tool** – Generate a unique, inspirational thought using Gemini

Trimmed down from [Ai_Agent_new](https://github.com/alumnx-ai-labs/Ai_Agent_new) to the two core tools, with LangSmith tracing wired in.

## Prerequisites

- Python 3.9+
- Gemini API Key (free, no credit card required — see setup below)
- LangSmith API Key (optional, for tracing)

## Setup

### 1. Clone the Repository
```bash
git clone https://github.com/alumnx-ai-labs/simple_agent.git
cd simple_agent
```

### 2. Create and Activate a Virtual Environment
```bash
python -m venv myenv

# Windows
myenv\Scripts\activate

# macOS/Linux
source myenv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Get a Free Gemini API Key

1. Go to [aistudio.google.com/apikey](https://aistudio.google.com/apikey)
2. Sign in with a Google account (no credit card required)
3. Click **Create API key**

### 5. Set Up Environment Variables

Copy the example `.env` file and add your API key:

```bash
cp .env.example .env
```

Edit `.env` and add your Gemini key:
```env
GOOGLE_API_KEY=your_gemini_api_key_here
```

The `LANGSMITH_*` variables are optional and disabled by default (`LANGSMITH_TRACING=false`). The agent works fully without them — see [LangSmith Tracing](#langsmith-tracing) below to enable it.

## Usage

Run the agent with a message from the command line:

```bash
# Query the weather tool
python agent.py "What's the weather in Hyderabad?"

# Request an inspirational thought
python agent.py "Give me an inspirational thought for today"

# Multi-part requests
python agent.py "Tell me the weather in Mumbai and share a daily thought"
```

### Optional: Web UI

`app.py` is a small Flask wrapper around the same agent, so you can chat with it in a browser instead of the terminal.

```bash
python app.py
```

Then open [http://127.0.0.1:5000](http://127.0.0.1:5000) and chat with the agent from the page.

## Project Structure

```
simple_agent/
├── agent.py               # Core agent + tools (run this for CLI queries)
├── app.py                 # Optional Flask web UI for the agent
├── templates/
│   └── index.html         # UI page served by app.py
├── requirements.txt       # Python dependencies
├── README.md              # This file
├── .env.example           # Example environment variables
└── .gitignore             # Git ignore rules
```

## How It Works

The agent is created using LangChain's `create_agent()`:

```python
agent = create_agent(
    model="google_genai:gemini-3.5-flash-lite",
    tools=[get_weather, create_daily_thought],
    system_prompt="You are a helpful assistant. Make sure that you only respond with whatever is coming as input to the agent, and do not add any extra commentary or explanation.",
)
```

- **Input:** User message passed to `agent.invoke()`
- **Reasoning:** Gemini analyzes the input and decides which tool(s) to call
- **Acting:** The selected tool(s) run (`get_weather`, `create_daily_thought`)
- **Output:** The final response is returned with no extra commentary

### Available Tools

#### `get_weather(city: str) -> str`
Returns weather information for a given city.
- **Example:** `get_weather("New York")` → `"It's always sunny in New York!"`

#### `create_daily_thought() -> str`
Generates a unique inspirational daily thought using Gemini.
- **Example:** Returns thoughts like "Every day is a new opportunity to grow and learn."

## LangSmith Tracing

[LangSmith](https://smith.langchain.com) lets you inspect every step of the agent's run (which tool was called, with what input, and what it returned) in a web dashboard. LangChain reads the `LANGSMITH_*` environment variables automatically — no extra code is required.

To enable tracing:

1. Create a free LangSmith account at [smith.langchain.com](https://smith.langchain.com)
2. Generate an API key from **Settings → API Keys**
3. Add these to your `.env` file:
   ```env
   LANGSMITH_API_KEY=your_langsmith_api_key_here
   LANGSMITH_TRACING=true
   LANGSMITH_PROJECT=simple-agent
   ```
4. Run the agent as usual — traces will show up under your project at [smith.langchain.com](https://smith.langchain.com)

Tracing is entirely optional; the agent runs fine with `LANGSMITH_TRACING=false`.

## Deployment

To deploy the web UI (`app.py`) so it's reachable outside your machine:

1. **Use a production WSGI server** instead of Flask's dev server:
   ```bash
   pip install gunicorn   # not needed on Windows; use waitress instead
   gunicorn -w 2 -b 0.0.0.0:8000 app:app
   ```
   On Windows, use [waitress](https://pypi.org/project/waitress/) instead of gunicorn:
   ```bash
   pip install waitress
   waitress-serve --host=0.0.0.0 --port=8000 app:app
   ```

2. **Set environment variables on the host** (do not commit `.env`):
   - `GOOGLE_API_KEY`
   - `LANGSMITH_API_KEY`, `LANGSMITH_TRACING=true`, `LANGSMITH_PROJECT` (optional, for tracing production traffic)

3. **Pick a host.** Any platform that runs a Python web process works, for example:
   - **Render / Railway**: connect the GitHub repo, set the build command to `pip install -r requirements.txt`, and the start command to `waitress-serve --host=0.0.0.0 --port=$PORT app:app`. Add the env vars above in the dashboard.
   - **Fly.io / Heroku**: same idea via a `Procfile` (`web: waitress-serve --host=0.0.0.0 --port=$PORT app:app`).
   - **Docker**: build an image with `requirements.txt` installed and `CMD ["waitress-serve", "--host=0.0.0.0", "--port=8000", "app:app"]`, then run it on any container host.

4. **Verify** by opening the deployed URL and sending a test message; check the LangSmith dashboard (if enabled) to confirm traces are arriving from production.

## Configuration

### Model Selection

- **Agent & Daily Thoughts:** `gemini-3.5-flash-lite` – free tier, well suited for learning and experimentation. You can change the model name in `agent.py` if Google retires it — see [ai.google.dev/gemini-api/docs/models](https://ai.google.dev/gemini-api/docs/models) for current options.

## Troubleshooting

### "No GOOGLE_API_KEY found in environment!"
- Ensure your `.env` file exists and contains `GOOGLE_API_KEY`
- Verify the `.env` file is in the same directory as `agent.py`
- Check that your API key is valid at [aistudio.google.com/apikey](https://aistudio.google.com/apikey)

### "404 NOT_FOUND ... model ... is no longer available to new users"
- Google periodically retires older free-tier model names. Check [ai.google.dev/gemini-api/docs/models](https://ai.google.dev/gemini-api/docs/models) for the current Flash model name and update it in `agent.py`.

### "429 RESOURCE_EXHAUSTED ... quota exceeded"
- The free tier has a daily/per-minute request limit per model. Wait for the quota to reset, or check current limits at [ai.google.dev/gemini-api/docs/rate-limits](https://ai.google.dev/gemini-api/docs/rate-limits).

### LangSmith tracing not working
- Verify `LANGSMITH_API_KEY` is set in your `.env` file
- Ensure `LANGSMITH_TRACING=true`
- Check that your LangSmith API key is valid
- Note: the agent still functions without LangSmith keys

## License

[Add your license here]
