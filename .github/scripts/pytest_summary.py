"""Turn a pytest JUnit-XML report into a GitHub Actions job summary.

    python .github/scripts/pytest_summary.py report.xml "python 3.12, latest torch"

Writes markdown to the file named by ``$GITHUB_STEP_SUMMARY`` (and to stdout when that
variable is unset, so the script can be tried locally).  The summary is what the run page
shows without anyone having to open the log: the counts, every failure with its message,
why anything was skipped, and the slowest tests.

Never fails the job: the exit status of ``pytest`` decides that, not the readability of
its report.  If the report is missing -- pytest crashed before writing it -- that is said
plainly instead of raising.
"""
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

MAX_FAILURES_SHOWN = 20
MAX_MESSAGE_CHARS = 800
SLOWEST_SHOWN = 10


def emit(markdown: str) -> None:
    """Append to the job summary, or print it when running outside Actions."""
    target = os.environ.get("GITHUB_STEP_SUMMARY")
    if not target:
        print(markdown)
        return
    with open(target, "a", encoding="utf-8") as fh:
        fh.write(markdown + "\n")


def collect(report: Path):
    """``(cases, totals)`` from a JUnit XML file; ``cases`` is one dict per test."""
    root = ET.parse(report).getroot()
    # pytest writes <testsuites><testsuite>...; older versions write <testsuite> alone.
    suites = root.iter("testsuite") if root.tag == "testsuites" else [root]
    cases, totals = [], {"passed": 0, "failed": 0, "error": 0, "skipped": 0}
    for suite in suites:
        for case in suite.iter("testcase"):
            outcome, message = "passed", ""
            for tag, name in (("failure", "failed"), ("error", "error"), ("skipped", "skipped")):
                node = case.find(tag)
                if node is not None:
                    outcome = name
                    message = (node.get("message") or node.text or "").strip()
                    break
            totals[outcome] += 1
            cases.append({
                "name": f"{case.get('classname', '')}::{case.get('name', '')}".lstrip(":"),
                "time": float(case.get("time") or 0.0),
                "outcome": outcome,
                "message": message,
            })
    return cases, totals


def main() -> int:
    report = Path(sys.argv[1] if len(sys.argv) > 1 else "report.xml")
    label = sys.argv[2] if len(sys.argv) > 2 else "test run"

    if not report.exists():
        emit(f"## {label}\n\n"
             f"No `{report}` was written -- pytest did not get as far as reporting "
             f"(a collection error, or the job was cancelled or timed out). "
             f"The step log above has the reason.")
        return 0

    try:
        cases, totals = collect(report)
    except ET.ParseError as err:
        emit(f"## {label}\n\n`{report}` could not be parsed ({err}).")
        return 0

    total = sum(totals.values())
    broken = totals["failed"] + totals["error"]
    mark = "x" if broken else ("warning" if not total else "white_check_mark")
    lines = [
        f"## :{mark}: {label}",
        "",
        "| passed | failed | errors | skipped | total | time |",
        "|-------:|-------:|-------:|--------:|------:|-----:|",
        f"| {totals['passed']} | {totals['failed']} | {totals['error']} | "
        f"{totals['skipped']} | {total} | {sum(c['time'] for c in cases):.1f}s |",
        "",
    ]

    broken_cases = [c for c in cases if c["outcome"] in ("failed", "error")]
    if broken_cases:
        lines.append(f"### Failures ({len(broken_cases)})")
        lines.append("")
        for case in broken_cases[:MAX_FAILURES_SHOWN]:
            # A backtick fence inside the message would end the block early and spill raw
            # markdown into the summary.
            message = case["message"][:MAX_MESSAGE_CHARS].replace("```", "'''") or "(no message)"
            lines += [f"<details><summary><code>{case['name']}</code></summary>", "",
                      "```", message, "```", "</details>", ""]
        if len(broken_cases) > MAX_FAILURES_SHOWN:
            lines.append(f"...and {len(broken_cases) - MAX_FAILURES_SHOWN} more; see the log.")
            lines.append("")

    skipped = [c for c in cases if c["outcome"] == "skipped"]
    if skipped:
        # Group by reason: 15 tests skipped for one missing dataset is one line, not 15.
        reasons = {}
        for case in skipped:
            reasons.setdefault(case["message"] or "(no reason given)", []).append(case["name"])
        lines += [f"### Skipped ({len(skipped)})", ""]
        for reason, names in sorted(reasons.items(), key=lambda kv: -len(kv[1])):
            lines.append(f"* **{len(names)}** -- {reason}")
        lines.append("")

    slowest = sorted(cases, key=lambda c: -c["time"])[:SLOWEST_SHOWN]
    if slowest and slowest[0]["time"] > 0:
        lines += [f"<details><summary>{SLOWEST_SHOWN} slowest tests</summary>", "",
                  "| test | seconds |", "|------|--------:|"]
        lines += [f"| `{c['name']}` | {c['time']:.2f} |" for c in slowest]
        lines += ["", "</details>", ""]

    emit("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
