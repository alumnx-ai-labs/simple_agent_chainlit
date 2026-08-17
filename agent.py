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
    """Get real-time weather for a given city using the Open-Meteo API (free, no key required)."""
    import requests

    # Step 1: Geocode city name to latitude/longitude
    geo_url = "https://geocoding-api.open-meteo.com/v1/search"
    geo_resp = requests.get(geo_url, params={"name": city, "count": 1}, timeout=10)
    geo_data = geo_resp.json()

    if not geo_data.get("results"):
        return f"Sorry, I couldn't find a location called '{city}'."

    location = geo_data["results"][0]
    lat = location["latitude"]
    lon = location["longitude"]
    resolved_name = location.get("name", city)
    country = location.get("country", "")

    # Step 2: Fetch current weather
    weather_url = "https://api.open-meteo.com/v1/forecast"
    weather_resp = requests.get(weather_url, params={
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code",
        "temperature_unit": "celsius",
        "wind_speed_unit": "kmh",
    }, timeout=10)
    weather_data = weather_resp.json()

    current = weather_data.get("current", {})
    temp = current.get("temperature_2m", "N/A")
    humidity = current.get("relative_humidity_2m", "N/A")
    wind = current.get("wind_speed_10m", "N/A")
    code = current.get("weather_code", -1)

    # WMO weather interpretation codes -> human-readable condition
    wmo_codes = {
        0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
        45: "Foggy", 48: "Depositing rime fog",
        51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
        61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
        71: "Slight snowfall", 73: "Moderate snowfall", 75: "Heavy snowfall",
        80: "Slight rain showers", 81: "Moderate rain showers", 82: "Violent rain showers",
        95: "Thunderstorm", 96: "Thunderstorm with slight hail", 99: "Thunderstorm with heavy hail",
    }
    condition = wmo_codes.get(code, "Unknown")

    return (
        f"Weather in {resolved_name}, {country}: {condition}. "
        f"Temperature: {temp}°C, Humidity: {humidity}%, Wind: {wind} km/h."
    )


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
