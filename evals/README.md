# Multi-Tier Agent Evaluation Framework

A production-grade evaluation suite for the LangChain agent in `agent.py`, featuring:
1. **Tier 1: Deterministic Trajectory Matching** (`agentevals`, Argument Precision, Groundedness)
2. **Tier 2: LLM-as-a-Judge Evaluation** (`agentevals` Trajectory Judge, Faithfulness Grader)
3. **Tier 3: LangSmith Cloud Integration** (`langsmith.evaluate()`, Dataset Sync, Tracing & Dashboards)

---

## 📁 Architecture

```
evals/
├── __init__.py
├── dataset.py        # 20 benchmark test cases (4 per category) with reference trajectories
├── evaluators.py     # Tier 1 & Tier 2 evaluators (agentevals match, LLM judges, faithfulness)
├── run_evals.py      # CLI runner for Tier 1 & Tier 2 evals (CLI table, Markdown, JSON)
├── langsmith_eval.py # Tier 3: LangSmith cloud dataset synchronization and evaluate() runner
├── README.md         # Guide & methodology for developers and management
├── eval_report.md    # Formatted Markdown report generated after each test run
└── eval_results.json # Machine-readable JSON output for CI/CD archiving
```

---

## 🔬 Evaluation Tiers & Graders

### Tier 1: Deterministic Trajectory & Groundedness Graders (Fast, 0 extra LLM cost)
- **`agentevals.trajectory.match`**: Verifies tool execution trajectories against golden references using `create_trajectory_match_evaluator`.
- **`argument_extraction`**: Checks parameter accuracy (e.g. `city: "Tokyo"`).
- **`output_groundedness`**: Confirms presence of key factual elements from tools.
- **`negative_controls`**: Enforces that forbidden tools are never called for direct QA, math, or greetings.

### Tier 2: LLM-as-a-Judge Evaluators (Qualitative & Semantic Reasoning)
- **`agentevals.trajectory.llm` (`TRAJECTORY_ACCURACY_PROMPT`)**: An LLM judge evaluates whether the agent's multi-step trajectory was logical, necessary, and efficient.
- **`llm_faithfulness`**: Verifies that the agent's final answer strictly relies on tool output without inventing facts or hallucinating.

### Tier 3: LangSmith Cloud Evaluation (`langsmith.evaluate`)
- **Dataset Sync**: Automatically syncs test cases to a LangSmith cloud dataset (`simple-agent-benchmark`).
- **Interactive Tracing**: Captures multi-step execution traces, latency, token consumption, and evaluator scores in a live cloud dashboard.

---

## 🧪 Benchmark Test Dataset (20 Cases, 4 per Category)

| Category | Tests | Description | Expected Trajectory |
| :--- | :---: | :--- | :--- |
| **`weather`** | 4 | Single-city, multi-word city, informal phrasing, city+country | Calls `get_weather` with real Open-Meteo API |
| **`daily_thought`** | 4 | Direct requests, motivation synonyms, quote requests | Calls `create_daily_thought` |
| **`negative_control`** | 4 | Greeting, math, factual QA, coding query | **No tools** (Direct response) |
| **`multi_tool`** | 4 | Weather + inspirational quote compound queries | Calls both `get_weather` & `create_daily_thought` |
| **`edge_cases`** | 4 | City typos, prompt injection, persona jailbreak, whitespace | Robust routing & safety |

---

## 🚀 How to Run the Evaluations

### 1. Tier 1: Deterministic Evaluation (Fast)
```bash
# Run all 20 test cases
python evals/run_evals.py

# Run specific categories
python evals/run_evals.py --category weather negative_control
```

### 2. Tier 2: Deterministic + LLM-as-a-Judge Evaluation
```bash
# Run with LLM-as-a-Judge enabled
python evals/run_evals.py --llm-judge

# Run specific category with LLM-as-a-Judge
python evals/run_evals.py --category weather --llm-judge
```

### 3. Tier 3: LangSmith Cloud Evaluation
1. Set `LANGSMITH_API_KEY`, `LANGSMITH_TRACING=true`, and `LANGSMITH_PROJECT` in `.env`.
2. Run:
```bash
python evals/langsmith_eval.py
```
3. Open [smith.langchain.com](https://smith.langchain.com) to view the interactive test suite and traces.

---

## 📊 Evaluation Reports
After execution, results are generated in:
- **`evals/eval_report.md`**: Markdown summary table with per-test details and LLM judge reasoning.
- **`evals/eval_results.json`**: Full JSON data for automated CI/CD pipelines.
