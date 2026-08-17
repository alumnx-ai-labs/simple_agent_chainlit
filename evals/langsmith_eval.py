"""LangSmith Integration & Dataset Evaluation Runner (Tier 3).

Usage:
    python evals/langsmith_eval.py
    python evals/langsmith_eval.py --dataset-name simple-agent-benchmark
    python evals/langsmith_eval.py --category weather

Requirements:
    1. Set your LANGSMITH_API_KEY in .env or environment variable
    2. Set LANGSMITH_TRACING=true and LANGSMITH_PROJECT (optional)
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

# Windows terminal UTF-8 encoding safeguard
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure parent directory is on sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from dotenv import load_dotenv
load_dotenv(os.path.join(PARENT_DIR, ".env"), override=True)

from agent import agent, extract_text
from evals.dataset import EVAL_DATASET, TestCase, get_dataset
from evals.evaluators import (
    extract_tool_calls,
    format_messages_for_agentevals,
    evaluate_argument_extraction,
    evaluate_groundedness,
    evaluate_trajectory_with_agentevals,
    evaluate_faithfulness_llm,
    evaluate_trajectory_llm_judge,
)

try:
    from langsmith import Client, evaluate
    from langsmith.evaluation import EvaluationResult
    HAS_LANGSMITH = True
except ImportError:
    HAS_LANGSMITH = False


def check_langsmith_configured() -> bool:
    """Verify if valid LangSmith credentials are present."""
    api_key = os.getenv("LANGSMITH_API_KEY", "").strip()
    if not api_key or api_key.startswith("<<") or api_key == "your_langsmith_api_key_here":
        return False
    return True


def sync_dataset_to_langsmith(client: Any, dataset_name: str, test_cases: List[TestCase]):
    """Ensure the evaluation dataset exists and is up to date in LangSmith."""
    print(f">> Checking LangSmith dataset: '{dataset_name}'...")
    
    # Check if dataset already exists
    if client.has_dataset(dataset_name=dataset_name):
        dataset = client.read_dataset(dataset_name=dataset_name)
        print(f"   Found existing dataset '{dataset_name}' (ID: {dataset.id})")
        return dataset

    print(f"   Creating new dataset '{dataset_name}' with {len(test_cases)} test cases...")
    dataset = client.create_dataset(
        dataset_name=dataset_name,
        description="Comprehensive evaluation dataset for Simple LangChain Agent across 5 benchmark categories."
    )

    for tc in test_cases:
        client.create_example(
            inputs={
                "input": tc.input,
                "category": tc.category,
                "test_id": tc.id,
            },
            outputs={
                "expected_tools": tc.expected_tools,
                "expected_args": tc.expected_args,
                "expected_output_contains": tc.expected_output_contains,
                "forbidden_tools": tc.forbidden_tools,
                "description": tc.description,
            },
            dataset_id=dataset.id,
        )

    print(f"   Successfully uploaded {len(test_cases)} examples to LangSmith.")
    return dataset


# =====================================================================
# LANGSMITH CUSTOM EVALUATORS
# =====================================================================

def langsmith_trajectory_evaluator(run: Any, example: Any) -> Dict[str, Any]:
    """LangSmith evaluator for tool trajectory match."""
    outputs = run.outputs or {}
    messages = outputs.get("messages", [])
    expected_outputs = example.outputs or {}

    tc = TestCase(
        id=example.inputs.get("test_id", "test"),
        category=example.inputs.get("category", "general"),
        input=example.inputs.get("input", ""),
        expected_tools=expected_outputs.get("expected_tools", []),
        expected_args=expected_outputs.get("expected_args", []),
        expected_output_contains=expected_outputs.get("expected_output_contains", []),
        forbidden_tools=expected_outputs.get("forbidden_tools", []),
    )

    score = evaluate_trajectory_with_agentevals(messages=messages, test_case=tc)
    return {
        "key": "agentevals_trajectory",
        "score": score.score,
        "comment": score.details,
    }


def langsmith_argument_evaluator(run: Any, example: Any) -> Dict[str, Any]:
    """LangSmith evaluator for tool argument extraction accuracy."""
    outputs = run.outputs or {}
    messages = outputs.get("messages", [])
    expected_args = (example.outputs or {}).get("expected_args", [])

    actual_tool_calls = extract_tool_calls(messages)
    score = evaluate_argument_extraction(actual_tool_calls=actual_tool_calls, expected_args=expected_args)
    return {
        "key": "argument_accuracy",
        "score": score.score,
        "comment": score.details,
    }


def langsmith_groundedness_evaluator(run: Any, example: Any) -> Dict[str, Any]:
    """LangSmith evaluator for final output groundedness."""
    outputs = run.outputs or {}
    output_text = outputs.get("output", "")
    expected_contains = (example.outputs or {}).get("expected_output_contains", [])

    score = evaluate_groundedness(output_text=output_text, expected_contains=expected_contains)
    return {
        "key": "output_groundedness",
        "score": score.score,
        "comment": score.details,
    }


def langsmith_faithfulness_evaluator(run: Any, example: Any) -> Dict[str, Any]:
    """LangSmith LLM-as-a-Judge evaluator for response faithfulness."""
    outputs = run.outputs or {}
    output_text = outputs.get("output", "")
    messages = outputs.get("messages", [])

    score = evaluate_faithfulness_llm(output_text=output_text, messages=messages)
    return {
        "key": "llm_faithfulness",
        "score": score.score,
        "comment": score.details,
    }


def run_langsmith_evaluation(dataset_name: str = "simple-agent-benchmark", category: Optional[List[str]] = None):
    """Main execution function for LangSmith evaluation."""
    if not HAS_LANGSMITH:
        print("Error: 'langsmith' package is not installed. Please run: pip install langsmith")
        return

    if not check_langsmith_configured():
        print("=" * 70)
        print(">> LANGSMITH CREDENTIALS REQUIRED (Tier 3)")
        print("=" * 70)
        print("To run LangSmith cloud evaluations and view live dashboards:")
        print("1. Sign up for free at: https://smith.langchain.com")
        print("2. Generate an API Key under Settings -> API Keys")
        print("3. Add the following to your .env file:")
        print("      LANGSMITH_TRACING=true")
        print("      LANGSMITH_API_KEY=lsv2_pt_...")
        print("      LANGSMITH_PROJECT=simple-agent-evals")
        print("4. Re-run: python evals/langsmith_eval.py")
        print("=" * 70)
        return

    client = Client()
    test_cases = get_dataset(category)

    # 1. Sync dataset
    sync_dataset_to_langsmith(client, dataset_name, test_cases)

    # 2. Target Agent Runner
    def target(inputs: Dict[str, Any]) -> Dict[str, Any]:
        user_input = inputs["input"]
        result = agent.invoke({"messages": [{"role": "user", "content": user_input}]})
        messages = result.get("messages", [])
        output_text = extract_text(messages[-1].content) if messages else ""
        return {
            "output": output_text,
            "messages": messages,
            "trajectory": format_messages_for_agentevals(messages),
        }

    # 3. Evaluators list
    evaluators = [
        langsmith_trajectory_evaluator,
        langsmith_argument_evaluator,
        langsmith_groundedness_evaluator,
        langsmith_faithfulness_evaluator,
    ]

    print("\n" + "=" * 70)
    print(f">> Running LangSmith Evaluation on dataset: '{dataset_name}'")
    print("=" * 70)

    experiment_prefix = f"simple-agent-eval-{int(time.time())}"
    results = evaluate(
        target,
        data=dataset_name,
        evaluators=evaluators,
        experiment_prefix=experiment_prefix,
        max_concurrency=2,  # Respect API rate limits
    )

    print("\n" + "=" * 70)
    print(">> LangSmith Evaluation Complete!")
    print("=" * 70)
    print("You can view the full interactive trace and evaluation results at:")
    print("👉 https://smith.langchain.com")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Run LangSmith Cloud Evaluations for Simple LangChain Agent")
    parser.add_argument("--dataset-name", default="simple-agent-benchmark", help="LangSmith dataset name")
    parser.add_argument("--category", "-c", nargs="+", help="Filter categories to evaluate")
    args = parser.parse_args()

    run_langsmith_evaluation(dataset_name=args.dataset_name, category=args.category)


if __name__ == "__main__":
    main()
