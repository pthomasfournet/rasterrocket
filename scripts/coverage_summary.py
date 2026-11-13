#!/usr/bin/env python3
"""Per-package line coverage as a Markdown table.

Reads the JSON summary `cargo llvm-cov report --json --summary-only` writes
and attributes each source file to the workspace package whose manifest
directory is the longest prefix of its path; files outside every package
are dropped.  The table goes to stdout.

    python3 -I scripts/coverage_summary.py target/coverage/summary.json
"""

import json
import subprocess
import sys
from pathlib import Path

USAGE_ARGC = 2


def package_dirs() -> list[tuple[Path, str]]:
    """List every workspace member's manifest directory and package name.

    Deepest directory first, so the first match is the longest prefix.
    """
    metadata = json.loads(
        subprocess.run(
            ["cargo", "metadata", "--no-deps", "--format-version", "1"],  # noqa: S607 - cargo from PATH, as in CI
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    dirs = [
        (Path(pkg["manifest_path"]).parent.resolve(), pkg["name"]) for pkg in metadata["packages"]
    ]
    return sorted(dirs, key=lambda entry: len(entry[0].parts), reverse=True)


def owner(path: Path, dirs: list[tuple[Path, str]]) -> str | None:
    """Return the package owning `path`, or None outside every package."""
    for directory, name in dirs:
        if path.is_relative_to(directory):
            return name
    return None


def main() -> int:
    """Print the per-package table for the summary JSON named on the command line."""
    if len(sys.argv) != USAGE_ARGC:
        print(f"usage: {sys.argv[0]} SUMMARY_JSON", file=sys.stderr)
        return 2
    report = json.loads(Path(sys.argv[1]).read_text())
    dirs = package_dirs()

    lines: dict[str, list[int]] = {}
    for export in report["data"]:
        for entry in export["files"]:
            name = owner(Path(entry["filename"]).resolve(), dirs)
            if name is None:
                continue
            counts = entry["summary"]["lines"]
            total = lines.setdefault(name, [0, 0])
            total[0] += counts["covered"]
            total[1] += counts["count"]

    def percent(covered: int, count: int) -> str:
        return f"{100 * covered / count:.1f} %" if count else "n/a"

    print("| Package | Lines | Covered | Coverage |")
    print("|---|---:|---:|---:|")
    for name in sorted(lines):
        covered, count = lines[name]
        print(f"| `{name}` | {count} | {covered} | {percent(covered, count)} |")
    covered = sum(c for c, _ in lines.values())
    count = sum(n for _, n in lines.values())
    print(f"| **Total** | {count} | {covered} | **{percent(covered, count)}** |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
