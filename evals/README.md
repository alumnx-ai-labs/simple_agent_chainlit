# AgentEvals Suite for Simple LangChain Agent

This evaluation suite uses LangChain's official **`agentevals`** library to evaluate execution trajectories, tool call precision, parameter extraction, groundedness, and system safety for the agent in `agent.py`.

---

## 📁 Architecture

```
evals/
├── __init__.py
├── dataset.py        # 20 benchmark test cases (4 per category) with reference trajectories
├── evaluators.py     # Trajectory evaluators using agentevals (create_trajectory_match_evaluator)
├── run_evals.py      # CLI runner executing agentevals across the benchmark suite
├── README.md         # Guide and methodology for developers and management
└── eval_report.md    # Summary report generated after each test run
```

---

## 🔬 How AgentEvals is Used

Unlike basic output-only evaluations, **`agentevals`** inspects the **execution trajectory** (the entire multi-step message history and tool invocations) using:

1. **`agentevals.trajectory.match.create_trajectory_match_evaluator`**:
   - Compares the actual message trajectory produced by `agent.py` against golden reference trajectories.
   - Configured with `trajectory_match_mode="unordered"` to support deterministic and robust tool call validation regardless of multi-tool execution order.
2. **Negative Control Verification**:
   - Ensures forbidden tool calls are not executed for direct questions, math calculations, and chit-chat.
3. **Argument & Parameter Matching**:
   - Validates that parameters like `city` are accurately extracted from natural language.
4. **Groundedness Verification**:
   - Ensures the final response is grounded in the tool output without hallucination.

---

## 🧪 Benchmark Test Dataset (20 Cases, 4 per Category)

| Category | Tests | Description | Expected Trajectory |
| :--- | :---: | :--- | :--- |
| **`weather`** | 4 | Single-city, multi-word city, informal, city+country | Calls `get_weather` with extracted city |
| **`daily_thought`** | 4 | Direct requests, motivation synonyms, quote requests | Calls `create_daily_thought` |
| **`negative_control`** | 4 | Greeting, math, factual QA, coding query | **No tools** (Direct response) |
| **`multi_tool`** | 4 | Weather + inspirational quote compound queries | Calls both `get_weather` & `create_daily_thought` |
| **`edge_cases`** | 4 | City typos, prompt injection, persona jailbreak, whitespace | Robust routing & safety |

---

## 🚀 How to Run the Evaluations

### 1. Requirements
Ensure dependencies including `agentevals` are installed:
```bash
pip install -r requirements.txt
```

### 2. Environment Variables
Make sure your Gemini API key is configured in `.env`:
```bash
GOOGLE_API_KEY=your_key_here
```

### 3. Run AgentEvals Suite
```bash
# Run all 20 test cases
python evals/run_evals.py

# Run only weather trajectory evals
python evals/run_evals.py --category weather

# Run only negative control safety checks
python evals/run_evals.py --category negative_control

# Run with custom delay throttling (seconds between queries)
python evals/run_evals.py --delay 2.0
```

---

## 📊 Evaluation Reports
After execution, results are generated in:
- **`evals/eval_report.md`**: Formatted Markdown summary table with category breakdown.
- **`evals/eval_results.json`**: Machine-readable JSON artifact for CI/CD pipelines.
