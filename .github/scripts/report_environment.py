"""Print what is actually installed, to the log and to the GitHub Actions job summary.

    python .github/scripts/report_environment.py "python 3.12, latest torch"

The job name says what CI *intended* to install; this says what pip resolved, which is the
first thing worth knowing when a run behaves differently from a local one.
"""
import os
import platform
import sys

import torch


def main() -> int:
    label = sys.argv[1] if len(sys.argv) > 1 else "environment"

    try:
        import numpy
        numpy_version = numpy.__version__
    except ImportError:
        numpy_version = "not installed"

    rows = [
        ("python", platform.python_version()),
        ("torch", torch.__version__),
        ("numpy", numpy_version),
        ("platform", platform.platform()),
        ("CPU threads torch will use", str(torch.get_num_threads())),
    ]
    width = max(len(name) for name, _ in rows)
    print(f"--- {label} ---")
    for name, version in rows:
        print(f"{name.rjust(width)} : {version}")

    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as fh:
            fh.write(f"## {label}\n\n| | |\n|---|---|\n")
            fh.writelines(f"| {name} | `{version}` |\n" for name, version in rows)
            fh.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
