"""CLI Evaluation Runner for Simple LangChain Agent with AgentEvals.

Usage:
    python evals/run_evals.py
    python evals/run_evals.py --category weather
    python evals/run_evals.py --category negative_control
    python evals/run_evals.py --delay 2.0

Features:
- Executes benchmark evaluation against live LangChain agent
- Evaluates trajectories using LangChain's official AgentEvals library
- Handles 429 rate limits gracefully with auto-retry and backoff
- Windows cp1252 / UTF-8 safe console output
- Generates terminal summary, JSON results, and Markdown report
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List

# Windows terminal UTF-8 encoding safeguard
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure parent directory is on sys.path so we can import agent and evals
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from agent import agent, extract_text
from evals.dataset import EVAL_DATASET, TestCase, get_dataset
from evals.evaluators import TestCaseResult, run_test_evaluation


def run_single_eval(test_case: TestCase, max_retries: int = 3, verbose: bool = False) -> TestCaseResult:
    """Execute the agent for a single test case with automatic 429 retry backoff."""
    start_time = time.time()
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            result = agent.invoke({"messages": [{"role": "user", "content": test_case.input}]})
            latency = round(time.time() - start_time, 3)

            messages = result.get("messages", [])
            raw_output = messages[-1].content if messages else ""
            output_text = extract_text(raw_output)

            return run_test_evaluation(
                test_case=test_case,
                messages=messages,
                output_text=output_text,
                latency=latency,
            )

        except Exception as e:
            last_error = str(e)
            # Handle rate limit (429 RESOURCE_EXHAUSTED) with backoff
            if "RESOURCE_EXHAUSTED" in last_error or "429" in last_error:
                wait_time = 8 * attempt
                print(f" [Rate limit hit, pausing {wait_time}s for retry {attempt}/{max_retries}]...", end="", flush=True)
                time.sleep(wait_time)
                continue
            else:
                break

    latency = round(time.time() - start_time, 3)
    return TestCaseResult(
        test_id=test_case.id,
        category=test_case.category,
        input_text=test_case.input,
        passed=False,
        latency_seconds=latency,
        output_text="",
        error=last_error,
    )


def generate_markdown_report(results: List[TestCaseResult], output_path: str):
    """Generate a clean Markdown evaluation report."""
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    pass_rate = (passed / total * 100) if total > 0 else 0
    avg_latency = sum(r.latency_seconds for r in results) / total if total > 0 else 0

    categories = sorted(list(set(r.category for r in results)))

    lines = [
        "# Agent Evaluation Report (AgentEvals)",
        f"**Run Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Overall Summary",
        f"- **Total Tests:** {total}",
        f"- **Passed:** {passed}",
        f"- **Failed:** {total - passed}",
        f"- **Pass Rate:** **{pass_rate:.1f}%**",
        f"- **Average Latency:** **{avg_latency:.2f}s**",
        "",
        "## Category Breakdown",
        "| Category | Tests | Passed | Pass Rate | Avg Latency |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ]

    for cat in categories:
        cat_results = [r for r in results if r.category == cat]
        cat_total = len(cat_results)
        cat_passed = sum(1 for r in cat_results if r.passed)
        cat_rate = (cat_passed / cat_total * 100) if cat_total > 0 else 0
        cat_latency = sum(r.latency_seconds for r in cat_results) / cat_total if cat_total > 0 else 0
        lines.append(f"| {cat} | {cat_total} | {cat_passed} | {cat_rate:.1f}% | {cat_latency:.2f}s |")

    lines.extend([
        "",
        "## Detailed Test Results",
        "| Status | Test ID | Category | Input | Latency | Details |",
        "| :---: | :--- | :--- | :--- | :---: | :--- |",
    ])

    for r in results:
        status = "PASS" if r.passed else "FAIL"
        input_preview = (r.input_text[:35] + "...") if len(r.input_text) > 35 else r.input_text
        failed_details = "; ".join([m.details for m in r.metric_scores if not m.passed]) if not r.passed else "Passed"
        if r.error:
            failed_details = f"Error: {r.error[:80]}..."
        lines.append(f"| {status} | `{r.test_id}` | `{r.category}` | {input_preview} | {r.latency_seconds}s | {failed_details} |")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def generate_json_results(results: List[TestCaseResult], output_path: str):
    """Serialize results to JSON for CI/CD archiving."""
    data = {
        "timestamp": datetime.now().isoformat(),
        "framework": "agentevals",
        "summary": {
            "total": len(results),
            "passed": sum(1 for r in results if r.passed),
            "failed": sum(1 for r in results if not r.passed),
            "pass_rate": round((sum(1 for r in results if r.passed) / len(results) * 100), 2) if results else 0,
            "avg_latency": round(sum(r.latency_seconds for r in results) / len(results), 3) if results else 0,
        },
        "results": [
            {
                "test_id": r.test_id,
                "category": r.category,
                "input": r.input_text,
                "passed": r.passed,
                "latency_seconds": r.latency_seconds,
                "actual_tools": r.actual_tools,
                "actual_args": r.actual_args,
                "output_text": r.output_text,
                "error": r.error,
                "metrics": [{"name": m.name, "passed": m.passed, "score": m.score, "details": m.details} for m in r.metric_scores]
            }
            for r in results
        ]
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Run Evaluation Suite for Simple LangChain Agent with AgentEvals")
    parser.add_argument("--category", "-c", nargs="+", help="Filter test categories to run (e.g. weather, negative_control)")
    parser.add_argument("--delay", "-d", type=float, default=1.0, help="Delay in seconds between test cases")
    parser.add_argument("--verbose", "-v", action="store_true", default=True, help="Print detailed per-test output")
    parser.add_argument("--report", "-r", action="store_true", default=True, help="Generate eval_report.md and eval_results.json")
    args = parser.parse_args()

    dataset = get_dataset(args.category)
    print("=" * 70)
    print(f">> Running Simple Agent Evaluation Suite with AgentEvals ({len(dataset)} test cases)")
    if args.category:
        print(f"   Filtering Categories: {', '.join(args.category)}")
    print("=" * 70)

    results: List[TestCaseResult] = []
    for i, tc in enumerate(dataset, 1):
        print(f"Running [{i:02d}/{len(dataset):02d}]: {tc.id} ...", end="", flush=True)
        res = run_single_eval(tc, verbose=False)
        status = "PASSED" if res.passed else "FAILED"
        print(f" [{status}] ({res.latency_seconds}s)")
        if args.verbose and not res.passed:
            print(f"   -> Input: {tc.input}")
            print(f"   -> Tools: {res.actual_tools} (Expected: {tc.expected_tools})")
            for m in res.metric_scores:
                if not m.passed:
                    print(f"   -> Failed: [{m.name}] {m.details}")
            if res.error:
                print(f"   -> Error: {res.error[:100]}...")
        results.append(res)

        # Gentle throttle between calls to respect Gemini Free Tier RPM limits
        if i < len(dataset) and args.delay > 0:
            time.sleep(args.delay)

    # Summary Calculations
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    pass_rate = (passed / total * 100) if total > 0 else 0
    avg_latency = sum(r.latency_seconds for r in results) / total if total > 0 else 0

    print("\n" + "=" * 70)
    print("EVALUATION RESULTS SUMMARY (AgentEvals)")
    print("=" * 70)
    print(f"Total Test Cases:    {total}")
    print(f"Passed:              {passed}")
    print(f"Failed:              {total - passed}")
    print(f"Pass Rate:           {pass_rate:.1f}%")
    print(f"Average Latency:     {avg_latency:.2f}s")
    print("-" * 70)
    print(f"{'Category':<20} | {'Tests':<6} | {'Passed':<6} | {'Pass Rate':<10} | {'Avg Latency'}")
    print("-" * 70)

    categories = sorted(list(set(r.category for r in results)))
    for cat in categories:
        cat_res = [r for r in results if r.category == cat]
        c_tot = len(cat_res)
        c_pas = sum(1 for r in cat_res if r.passed)
        c_rate = (c_pas / c_tot * 100) if c_tot > 0 else 0
        c_lat = sum(r.latency_seconds for r in cat_res) / c_tot if c_tot > 0 else 0
        print(f"{cat:<20} | {c_tot:<6} | {c_pas:<6} | {c_rate:>8.1f}% | {c_lat:.2f}s")
    print("=" * 70)

    if args.report:
        md_path = os.path.join(CURRENT_DIR, "eval_report.md")
        json_path = os.path.join(CURRENT_DIR, "eval_results.json")
        generate_markdown_report(results, md_path)
        generate_json_results(results, json_path)
        print("Reports generated:")
        print(f"   - Markdown: {md_path}")
        print(f"   - JSON:     {json_path}")


if __name__ == "__main__":
    main()
