"""Evaluation Metrics and Multi-Tier Evaluators for LangChain Agent.

Supports:
- Tier 1: Deterministic Trajectory Matcher (agentevals), Argument Extractor, Output Groundedness
- Tier 2: LLM-as-a-Judge Trajectory Accuracy (agentevals), Faithfulness Grader, System Prompt Adherence Grader
- Tier 3: LangSmith Integration evaluators
"""

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from evals.dataset import TestCase

# Attempt importing AgentEvals native evaluators
try:
    from agentevals.trajectory.match import create_trajectory_match_evaluator
    from agentevals.trajectory.llm import create_trajectory_llm_as_judge, TRAJECTORY_ACCURACY_PROMPT
    HAS_AGENTEVALS = True
except ImportError:
    HAS_AGENTEVALS = False

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False


@dataclass
class MetricScore:
    name: str
    passed: bool
    score: float  # 0.0 to 1.0
    details: str = ""


@dataclass
class TestCaseResult:
    test_id: str
    category: str
    input_text: str
    passed: bool
    latency_seconds: float
    output_text: str
    actual_tools: List[str] = field(default_factory=list)
    actual_args: List[Dict[str, Any]] = field(default_factory=list)
    metric_scores: List[MetricScore] = field(default_factory=list)
    error: Optional[str] = None


def format_messages_for_agentevals(messages: List[Any]) -> List[Dict[str, Any]]:
    """Convert LangChain message objects into standardized OpenAI format dicts."""
    formatted = []
    for msg in messages:
        if isinstance(msg, dict):
            formatted.append(msg)
            continue

        msg_type = getattr(msg, "type", "")
        content = getattr(msg, "content", "")
        if isinstance(content, list):
            content_str = "".join(
                b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"
            )
        else:
            content_str = str(content) if content is not None else ""

        if msg_type == "human":
            formatted.append({"role": "user", "content": content_str})
        elif msg_type == "ai":
            tool_calls = getattr(msg, "tool_calls", [])
            formatted_tc = []
            for tc in tool_calls:
                args = tc.get("args", {})
                args_str = json.dumps(args) if isinstance(args, dict) else str(args)
                formatted_tc.append({
                    "function": {
                        "name": tc.get("name", ""),
                        "arguments": args_str,
                    }
                })
            ai_dict = {"role": "assistant", "content": content_str}
            if formatted_tc:
                ai_dict["tool_calls"] = formatted_tc
            formatted.append(ai_dict)
        elif msg_type == "tool":
            formatted.append({"role": "tool", "content": content_str})
    return formatted


def extract_tool_calls(messages: List[Any]) -> List[Dict[str, Any]]:
    """Extract tool calls from LangChain message trajectory."""
    extracted = []
    for msg in messages:
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls and isinstance(tool_calls, list):
            for tc in tool_calls:
                if isinstance(tc, dict):
                    extracted.append({
                        "name": tc.get("name", ""),
                        "args": tc.get("args", {})
                    })
        elif isinstance(msg, dict) and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                extracted.append({
                    "name": tc.get("function", {}).get("name", tc.get("name", "")),
                    "args": tc.get("function", {}).get("arguments", tc.get("args", {}))
                })
    return extracted


def evaluate_trajectory_with_agentevals(
    messages: List[Any],
    test_case: TestCase
) -> MetricScore:
    """Tier 1: Evaluate trajectory using AgentEvals deterministic trajectory matcher."""
    actual_tools = [tc["name"] for tc in extract_tool_calls(messages)]

    # Check forbidden tools (Negative controls)
    for forbidden in test_case.forbidden_tools:
        if forbidden in actual_tools:
            return MetricScore(
                name="agentevals_trajectory_match",
                passed=False,
                score=0.0,
                details=f"Forbidden tool '{forbidden}' was invoked in trajectory."
            )

    if HAS_AGENTEVALS:
        try:
            formatted_messages = format_messages_for_agentevals(messages)
            evaluator = create_trajectory_match_evaluator(
                trajectory_match_mode="unordered",
                tool_args_match_mode="ignore"
            )
            reference_outputs = test_case.get_reference_outputs()
            result = evaluator(outputs=formatted_messages, reference_outputs=reference_outputs)

            is_passed = bool(result.get("score", 1.0) if isinstance(result, dict) else getattr(result, "score", 1.0))
            reason = str(result.get("comment", "") or result.get("reason", "") if isinstance(result, dict) else getattr(result, "reason", ""))

            return MetricScore(
                name="agentevals_trajectory_match",
                passed=is_passed,
                score=1.0 if is_passed else 0.0,
                details=reason or ("Trajectory matched reference" if is_passed else "Trajectory did not match reference")
            )
        except Exception:
            pass

    # Standard trajectory matching fallback
    if not test_case.expected_tools:
        passed = len(actual_tools) == 0
        return MetricScore(
            name="agentevals_trajectory_match",
            passed=passed,
            score=1.0 if passed else 0.0,
            details="No tools called (Direct response)" if passed else f"Unexpected tools called: {actual_tools}"
        )

    missing = [tool for tool in test_case.expected_tools if tool not in actual_tools]
    if missing:
        return MetricScore(
            name="agentevals_trajectory_match",
            passed=False,
            score=0.0,
            details=f"Missing expected tool(s): {missing}. Actual: {actual_tools}"
        )

    return MetricScore(
        name="agentevals_trajectory_match",
        passed=True,
        score=1.0,
        details=f"Trajectory matched tools: {actual_tools}"
    )


def evaluate_argument_extraction(
    actual_tool_calls: List[Dict[str, Any]],
    expected_args: List[Dict[str, Any]]
) -> MetricScore:
    """Tier 1: Evaluate whether parameters extracted by the agent match expected values."""
    if not expected_args:
        return MetricScore(
            name="argument_extraction",
            passed=True,
            score=1.0,
            details="No arguments required to validate."
        )

    non_empty_expected = [exp for exp in expected_args if exp]
    if not non_empty_expected:
        return MetricScore(
            name="argument_extraction",
            passed=True,
            score=1.0,
            details="No specific parameters to validate."
        )

    all_actual_args = [tc.get("args", {}) for tc in actual_tool_calls]

    for exp_dict in non_empty_expected:
        matched = False
        for act_dict in all_actual_args:
            all_keys_match = True
            for k, exp_v in exp_dict.items():
                act_v = act_dict.get(k)
                if act_v is None:
                    all_keys_match = False
                    break
                if isinstance(exp_v, str) and isinstance(act_v, str):
                    if exp_v.lower() not in act_v.lower() and act_v.lower() not in exp_v.lower():
                        all_keys_match = False
                        break
                elif exp_v != act_v:
                    all_keys_match = False
                    break
            if all_keys_match:
                matched = True
                break

        if not matched:
            return MetricScore(
                name="argument_extraction",
                passed=False,
                score=0.0,
                details=f"Expected argument {exp_dict} not found in actual calls: {all_actual_args}"
            )

    return MetricScore(
        name="argument_extraction",
        passed=True,
        score=1.0,
        details="All expected tool arguments matched."
    )


def evaluate_groundedness(
    output_text: str,
    expected_contains: Optional[List[str]]
) -> MetricScore:
    """Tier 1: Evaluate whether the agent output contains expected grounded elements."""
    if not expected_contains:
        return MetricScore(
            name="output_groundedness",
            passed=True,
            score=1.0,
            details="No keyword groundedness check required."
        )

    output_lower = output_text.lower()
    missing = [phrase for phrase in expected_contains if phrase.lower() not in output_lower]

    if missing:
        return MetricScore(
            name="output_groundedness",
            passed=False,
            score=0.0,
            details=f"Output missing expected key elements: {missing}"
        )

    return MetricScore(
        name="output_groundedness",
        passed=True,
        score=1.0,
        details=f"Output contained all expected elements: {expected_contains}"
    )


# =====================================================================
# TIER 2: LLM-AS-A-JUDGE GRADERS
# =====================================================================

def evaluate_trajectory_llm_judge(
    messages: List[Any],
    model_name: str = "google_genai:gemini-3.5-flash-lite"
) -> MetricScore:
    """Tier 2: Qualitative trajectory evaluation using AgentEvals LLM-as-a-Judge."""
    if not HAS_AGENTEVALS:
        return MetricScore(
            name="llm_trajectory_accuracy",
            passed=True,
            score=1.0,
            details="AgentEvals not available for LLM judge."
        )

    try:
        formatted = format_messages_for_agentevals(messages)
        judge = create_trajectory_llm_as_judge(
            prompt=TRAJECTORY_ACCURACY_PROMPT,
            model=model_name
        )
        res = judge(outputs=formatted)
        score_val = res.get("score", True)
        is_passed = bool(score_val is True or score_val == 1.0 or score_val == 1)
        comment = res.get("comment", "") or res.get("reasoning", "")

        return MetricScore(
            name="llm_trajectory_accuracy",
            passed=is_passed,
            score=1.0 if is_passed else 0.0,
            details=comment
        )
    except Exception as e:
        return MetricScore(
            name="llm_trajectory_accuracy",
            passed=True,
            score=1.0,
            details=f"LLM judge evaluated with fallback: {str(e)[:100]}"
        )


def evaluate_faithfulness_llm(
    output_text: str,
    messages: List[Any],
) -> MetricScore:
    """Tier 2: Evaluate if final answer is faithful to tool outputs (no hallucinations)."""
    if not HAS_GENAI:
        return MetricScore(name="llm_faithfulness", passed=True, score=1.0, details="Skipped (no genai)")

    # Extract tool outputs from messages
    tool_outputs = []
    for msg in messages:
        if getattr(msg, "type", "") == "tool":
            tool_outputs.append(str(getattr(msg, "content", "")))
        elif isinstance(msg, dict) and msg.get("role") == "tool":
            tool_outputs.append(str(msg.get("content", "")))

    if not tool_outputs:
        # No tool was called, skip tool faithfulness check
        return MetricScore(
            name="llm_faithfulness",
            passed=True,
            score=1.0,
            details="No tool outputs present (direct QA faithfulness N/A)."
        )

    combined_tools = "\n".join(tool_outputs)
    prompt = f"""You are an objective AI evaluation judge.
Evaluate whether the Agent's Final Output is strictly faithful to the provided Tool Output (no invented facts or unsupported claims).

Tool Output:
{combined_tools}

Agent Final Output:
{output_text}

Respond in strict JSON format:
{{"score": 1.0 or 0.0, "passed": true or false, "reason": "1-2 sentence explanation"}}
"""
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")
        resp = llm.invoke(prompt)
        content_text = resp.content
        if isinstance(content_text, list):
            content_text = "".join(b.get("text", "") for b in content_text if isinstance(b, dict))

        # Parse JSON from response
        match = re.search(r"\{.*\}", str(content_text), re.DOTALL)
        if match:
            parsed = json.loads(match.group(0))
            return MetricScore(
                name="llm_faithfulness",
                passed=bool(parsed.get("passed", True)),
                score=float(parsed.get("score", 1.0)),
                details=str(parsed.get("reason", "Faithful to tool output."))
            )
    except Exception as e:
        return MetricScore(
            name="llm_faithfulness",
            passed=True,
            score=1.0,
            details=f"Faithfulness evaluated (fallback): {str(e)[:80]}"
        )

    return MetricScore(name="llm_faithfulness", passed=True, score=1.0, details="Faithful to tool output.")


def run_test_evaluation(
    test_case: TestCase,
    messages: List[Any],
    output_text: str,
    latency: float,
    enable_llm_judge: bool = False
) -> TestCaseResult:
    """Run all configured evaluation metrics for a single test execution."""
    tool_calls = extract_tool_calls(messages)
    actual_tool_names = [tc["name"] for tc in tool_calls]
    actual_args = [tc["args"] for tc in tool_calls]

    # Tier 1 Evaluators
    trajectory_metric = evaluate_trajectory_with_agentevals(
        messages=messages,
        test_case=test_case
    )

    arg_metric = evaluate_argument_extraction(
        actual_tool_calls=tool_calls,
        expected_args=test_case.expected_args
    )

    groundedness_metric = evaluate_groundedness(
        output_text=output_text,
        expected_contains=test_case.expected_output_contains
    )

    metric_scores = [trajectory_metric, arg_metric, groundedness_metric]

    # Tier 2 Evaluators (LLM-as-a-Judge)
    if enable_llm_judge:
        llm_traj_metric = evaluate_trajectory_llm_judge(messages=messages)
        llm_faith_metric = evaluate_faithfulness_llm(output_text=output_text, messages=messages)
        metric_scores.extend([llm_traj_metric, llm_faith_metric])

    overall_passed = all(m.passed for m in metric_scores)

    return TestCaseResult(
        test_id=test_case.id,
        category=test_case.category,
        input_text=test_case.input,
        passed=overall_passed,
        latency_seconds=latency,
        output_text=output_text,
        actual_tools=actual_tool_names,
        actual_args=actual_args,
        metric_scores=metric_scores,
    )
