"""External-tool invocation shared by the test, benchmark and fuzz scripts.

Every tool is resolved to an absolute path before it runs, every run
captures its output as text, and a failed run is summarised by the last
non-blank line of its stderr, so the scripts report a missing tool, a
timeout or a failure in one form.
"""

import argparse
import os
import pathlib
import shutil
import signal
import subprocess
import sys
from collections.abc import Sequence


def find(name: str) -> str | None:
    """Return the absolute path of `name` on PATH, or `None` when absent."""
    path = shutil.which(name)
    return None if path is None else str(pathlib.Path(path).absolute())


def require(name: str) -> str:
    """Return the absolute path of `name` on PATH; exit when it is absent."""
    path = find(name)
    if path is None:
        sys.exit(f"{name} not found on PATH")
    return path


def last_line(text: str, default: str = "") -> str:
    """Return the last non-blank line of `text`, or `default` when there is none."""
    return next((line for line in reversed(text.splitlines()) if line.strip()), default)


def describe_exit(returncode: int) -> str:
    """Return `exit N`, or `killed by SIGNAME` for a negative (signal) return code."""
    if returncode >= 0:
        return f"exit {returncode}"
    try:
        return f"killed by {signal.Signals(-returncode).name}"
    except ValueError:
        return f"killed by signal {-returncode}"


def _capture(cmd: Sequence[str], timeout: float | None) -> subprocess.CompletedProcess[str]:
    """Run `cmd` with captured text output; `subprocess.TimeoutExpired` on timeout."""
    return subprocess.run(  # noqa: S603 -- argument list from the caller's resolved tool
        cmd,
        check=False,
        capture_output=True,
        text=True,
        errors="replace",
        timeout=timeout,
    )


def run(cmd: Sequence[str]) -> subprocess.CompletedProcess[str]:
    """Run `cmd` (an argument list, never a shell line) and capture its output.

    Undecodable output bytes are replaced, so a binary diagnostic never
    raises.  A non-zero exit is returned, not raised.
    """
    return _capture(cmd, None)


def run_limited(cmd: Sequence[str], timeout: float) -> subprocess.CompletedProcess[str] | None:
    """Run `cmd` as `run` does; `None` when it is still running after `timeout` s.

    The child is killed on timeout.
    """
    try:
        return _capture(cmd, timeout)
    except subprocess.TimeoutExpired:
        return None


def run_checked(cmd: Sequence[str]) -> str:
    """Run `cmd` and return its stdout; exit with its last stderr line when it fails."""
    r = run(cmd)
    if r.returncode != 0:
        sys.exit(f"{cmd[0]}: {describe_exit(r.returncode)}: {last_line(r.stderr, 'no stderr')}")
    return r.stdout


def executable(text: str) -> pathlib.Path:
    """Parse the path of an executable file, as an argparse `type`."""
    path = pathlib.Path(text)
    if not (path.is_file() and os.access(path, os.X_OK)):
        msg = f"not an executable file: {text}"
        raise argparse.ArgumentTypeError(msg)
    return path


def positive_int(text: str) -> int:
    """Parse a strictly positive integer, as an argparse `type`."""
    try:
        value = int(text)
    except ValueError:
        value = 0
    if value < 1:
        msg = f"not a positive integer: {text}"
        raise argparse.ArgumentTypeError(msg)
    return value
