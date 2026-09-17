"""Bounded syntax-only inspection. Never execute or import generated code."""
from dataclasses import dataclass
import re
import sys


@dataclass(frozen=True)
class SyntaxCheck:
    issues: tuple[str, ...]
    summary: str


CODE_CHECKS = {
    "requirements": (
        "Does the Python code in `draft` clearly fail to satisfy `user_message` given "
        "`recent_conversation` and coding requirements in `master_prompt`? "
        "Evaluate supplied text only; do not assume unseen files or requirements.",
        "Correct code that fails the requested behavior or supplied coding requirements.",
    ),
    "runtime": (
        "Does the Python code in `draft` contain a likely runtime or logic error, such as an "
        "undefined variable, incorrect API use, faulty calculation, or wrong control flow? "
        "Use the supplied context and `syntax_check`; code has not been executed. "
        "Do not treat missing external environment information as proof of an error.",
        "Inspect and fix likely runtime and logic errors; do not claim execution or tests passed.",
    ),
    "edge_cases": (
        "Does the Python code in `draft` fail to handle edge cases relevant to the actual "
        "request in `user_message`, such as empty inputs or invalid values? "
        "Avoid demanding unrelated production features for a small example.",
        "Handle edge cases relevant to the user's requested behavior without unrelated complexity.",
    ),
}


def check_python(text: str) -> SyntaxCheck:
    issues: list[str] = []
    blocks: list[str] = []
    current: list[str] | None = None
    language = ""
    for line in text.splitlines():
        fence = re.fullmatch(r"\s*```([^`]*)", line)
        if fence:
            if current is None:
                language = fence.group(1).strip().lower()
                current = []
            elif not fence.group(1).strip():
                if language in ("python", "py", ""):
                    blocks.append("\n".join(current))
                current = None
            else:
                current.append(line)
        elif current is not None:
            current.append(line)
    if current is not None:
        issues.append("Unclosed code fence; the response may be truncated.")
    if not blocks:
        issues.append("No complete Python code block found. Use a fenced python block.")
    for index, code in enumerate(blocks, 1):
        if not code.strip():
            issues.append(f"Python block {index} is empty.")
            continue
        if len(code) > 100_000:
            issues.append(f"Python block {index} exceeds the 100,000 character syntax-check limit.")
            continue
        try:
            # Compilation checks syntax/scope only; the returned code object is discarded.
            compile(code, f"<reply-block-{index}>", "exec", dont_inherit=True)
        except SyntaxError as exc:
            issues.append(f"Python block {index}, line {exc.lineno}: {exc.msg}.")
        except (ValueError, RecursionError, MemoryError, OverflowError):
            issues.append(f"Python block {index} could not be checked; simplify or shorten it.")
    outcome = "FAIL — " + " ".join(issues) if issues else f"PASS ({len(blocks)} block(s))"
    summary = (f"Python syntax: {outcome} [Python {sys.version_info.major}.{sys.version_info.minor}]. "
               "Code not executed; runtime behavior, dependencies and tests are unverified.")
    return SyntaxCheck(tuple(issues), summary)
