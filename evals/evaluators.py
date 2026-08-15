"""Evaluation Metrics and Trajectory Evaluation using AgentEvals.

Integrates:
- agentevals.trajectory.match (create_trajectory_match_evaluator) for deterministic trajectory matching
- Output groundedness and faithfulness verification
- Latency & performance tracking
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from evals.dataset import TestCase

# Attempt importing AgentEvals native evaluators
try:
    from agentevals.trajectory.match import create_trajectory_match_evaluator
    HAS_AGENTEVALS = True
except ImportError:
    HAS_AGENTEVALS = False

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


def extract_tool_calls(messages: List[Any]) -> List[Dict[str, Any]]:
    """Extract tool calls from LangChain/LangGraph message trajectory."""
    extracted = []
    for msg in messages:
        # Check AIMessage tool_calls attribute
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls and isinstance(tool_calls, list):
            for tc in tool_calls:
                if isinstance(tc, dict):
                    extracted.append({
                        "name": tc.get("name", ""),
                        "args": tc.get("args", {})
                    })
        # Check dict message structure if applicable
        elif isinstance(msg, dict) and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                extracted.append({
                    "name": tc.get("name", ""),
                    "args": tc.get("args", {})
                })
    return extracted


def evaluate_trajectory_with_agentevals(
    messages: List[Any],
    test_case: TestCase
) -> MetricScore:
    """Evaluate trajectory using AgentEvals trajectory matcher."""
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
            # Create AgentEvals trajectory matcher (unordered matching for multi-tool resilience)
            evaluator = create_trajectory_match_evaluator(
                trajectory_match_mode="unordered",
                tool_args_match_mode="ignore"
            )
            reference_outputs = test_case.get_reference_outputs()
            result = evaluator(outputs=messages, reference_outputs=reference_outputs)
            
            # AgentEvals result contains score / passed
            is_passed = bool(result.get("score", 1.0) if isinstance(result, dict) else getattr(result, "score", 1.0))
            reason = str(result.get("reason", "") if isinstance(result, dict) else getattr(result, "reason", ""))
            
            return MetricScore(
                name="agentevals_trajectory_match",
                passed=is_passed,
                score=1.0 if is_passed else 0.0,
                details=reason or ("Trajectory matched reference" if is_passed else "Trajectory did not match reference")
            )
        except Exception as e:
            # Fallback to structural trajectory match if message type formatting differs
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
    """Evaluate whether parameters extracted by the agent match expected values."""
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
                details=f"Expected argument {exp_dict} was not found in actual tool calls: {all_actual_args}"
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
    """Evaluate whether the agent output contains expected grounded elements."""
    if not expected_contains:
        return MetricScore(
            name="output_groundedness",
            passed=True,
            score=1.0,
            details="No specific keyword groundedness check required."
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


def run_test_evaluation(
    test_case: TestCase,
    messages: List[Any],
    output_text: str,
    latency: float
) -> TestCaseResult:
    """Run all evaluation metrics for a single test execution using AgentEvals."""
    tool_calls = extract_tool_calls(messages)
    actual_tool_names = [tc["name"] for tc in tool_calls]
    actual_args = [tc["args"] for tc in tool_calls]

    # AgentEvals Trajectory Matcher
    trajectory_metric = evaluate_trajectory_with_agentevals(
        messages=messages,
        test_case=test_case
    )

    # Argument Extraction Validator
    arg_metric = evaluate_argument_extraction(
        actual_tool_calls=tool_calls,
        expected_args=test_case.expected_args
    )

    # Groundedness Validator
    groundedness_metric = evaluate_groundedness(
        output_text=output_text,
        expected_contains=test_case.expected_output_contains
    )

    metric_scores = [trajectory_metric, arg_metric, groundedness_metric]
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
