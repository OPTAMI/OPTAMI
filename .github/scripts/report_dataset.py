"""Say whether a LibSVM dataset arrived, in the log and in the job summary.

    python .github/scripts/report_dataset.py data/LibSVM/a9a.txt

The download step is deliberately allowed to fail -- an unreachable university server should
not cost the whole run -- so the tests that need the file skip themselves instead.  That is
only acceptable if the skip is impossible to miss, which is what this prints.

Exits 0 either way: whether a missing dataset is a failure is the suite's call, not this
script's.
"""
import os
import sys
from pathlib import Path


def emit(line: str) -> None:
    print(line)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as fh:
            fh.write(line + "\n\n")


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "data/LibSVM/a9a.txt")

    if not path.exists() or path.stat().st_size == 0:
        emit(f":warning: **`{path}` is missing or empty** -- the download did not succeed. "
             f"The tests that need it will be reported as skipped below, and the rest of the "
             f"suite still runs.")
        return 0

    size_mb = path.stat().st_size / 1024 ** 2
    with path.open("rb") as fh:
        rows = sum(1 for _ in fh)
    emit(f":heavy_check_mark: `{path}` -- {size_mb:.1f} MB, {rows:,} rows.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
