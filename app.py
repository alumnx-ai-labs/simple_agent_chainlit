from flask import Flask, render_template, request, jsonify

from agent import agent, extract_text

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "")
    result = agent.invoke({"messages": [{"role": "user", "content": user_input}]})
    reply = extract_text(result["messages"][-1].content)
    return jsonify({"reply": reply})


if __name__ == "__main__":
    app.run(debug=True)
