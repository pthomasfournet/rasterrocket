#!/usr/bin/env python3
"""Report what changed upstream for every vendored crate since its base.

Each vendored crate carries an `UPSTREAM.toml` (crate, version, repository,
path, commit, date, and optionally reviewed, reviewed_date,
reviewed_version).  The reference point is the last review when `reviewed`
is set, else the base commit.  For each crate this reports:

- upstream commits touching the crate's directory after the reference
  commit;
- crates.io releases newer than the reference version;
- RustSec advisories filed against the crate (cargo-deny and cargo-audit do
  not see path dependencies).

It only reads public APIs.  Exit status: 0 nothing new, 1 something new, 2 a
lookup failed.  Set GITHUB_TOKEN to raise the GitHub API rate limit.
"""

import json
import os
import pathlib
import sys
import tomllib
import urllib.error
import urllib.request
from http import HTTPStatus
from typing import Any, NamedTuple

ROOT = pathlib.Path(__file__).resolve().parents[1]
PER_PAGE = 100
MAX_PAGES = 10


class LookupFailedError(Exception):
    """A public API answered with something other than the expected data."""


def get_json(url: str) -> Any:  # noqa: ANN401 -- arbitrary JSON document
    """GET `url` as JSON; `None` on 404."""
    headers = {"User-Agent": "rasterrocket-vendored-check", "Accept": "application/json"}
    token = os.environ.get("GITHUB_TOKEN")
    if token and url.startswith("https://api.github.com/"):
        headers["Authorization"] = f"Bearer {token}"
    if not url.startswith("https://"):
        msg = f"refusing a non-HTTPS URL: {url}"
        raise LookupFailedError(msg)
    request = urllib.request.Request(url, headers=headers)  # noqa: S310 -- HTTPS checked above
    try:
        with urllib.request.urlopen(request, timeout=30) as r:  # noqa: S310 -- HTTPS checked above
            return json.load(r)
    except urllib.error.HTTPError as e:
        if e.code == HTTPStatus.NOT_FOUND:
            return None
        raise


def get_list(url: str) -> list[Any]:
    """GET `url` as a JSON list; `[]` on 404, `LookupFailedError` for any other shape."""
    value = get_json(url)
    if value is None:
        return []
    if not isinstance(value, list):
        msg = f"expected a list from {url}, got {type(value).__name__}"
        raise LookupFailedError(msg)
    return value


def get_paged(url: str) -> list[Any]:
    """GET every page of a GitHub list endpoint (`url` already has a query).

    Raises `LookupFailedError` rather than truncate silently past `MAX_PAGES`.
    """
    items: list[Any] = []
    for page in range(1, MAX_PAGES + 1):
        batch = get_list(f"{url}&per_page={PER_PAGE}&page={page}")
        items += batch
        if len(batch) < PER_PAGE:
            return items
    msg = f"more than {MAX_PAGES * PER_PAGE} items at {url}"
    raise LookupFailedError(msg)


def version_key(v: str) -> tuple[int, ...]:
    """Return a sortable key of the release part of semver `v`.

    Pre-release and build suffixes are ignored, so `1.2.0-rc.1` and
    `1.2.0+meta` both key as `1.2.0`.
    """
    release = v.split("+", 1)[0].split("-", 1)[0]
    return tuple(int(p) if p.isdigit() else 0 for p in release.split("."))


class Reference(NamedTuple):
    """The upstream state a vendored crate was last compared against."""

    commit: str
    since: str  # ISO 8601 timestamp; upstream commits after it are new
    version: str
    what: str  # "reviewed" or "vendored base", for the report


def reference(base: dict[str, Any], repo: str) -> Reference:
    """Return the last review when `base` records one, else the vendored base."""
    if "reviewed" in base:
        return Reference(
            base["reviewed"], base["reviewed_date"], base["reviewed_version"], "reviewed"
        )
    # The base commit's own timestamp: `date` may be a calendar day, which
    # would also list the base's same-day ancestors.
    commit = get_json(f"https://api.github.com/repos/{repo}/commits/{base['commit']}")
    since = commit["commit"]["committer"]["date"] if commit else base["date"]
    return Reference(base["commit"], since, base["version"], "vendored base")


def subject(message: str) -> str:
    """Return the first line of a commit message."""
    return message.partition("\n")[0]


def check(manifest: pathlib.Path) -> tuple[str, list[str]]:
    """Return the crate name of `manifest` and what changed upstream since its reference."""
    base = tomllib.loads(manifest.read_text(encoding="utf-8"))
    name, repo, path = base["crate"], base["repository"], base["path"]
    ref = reference(base, repo)
    findings = []

    commits = get_paged(
        f"https://api.github.com/repos/{repo}/commits?path={path}&since={ref.since}"
    )
    new = [c for c in commits if c["sha"] not in (ref.commit, base["commit"])]
    if new:
        findings.append(
            f"{len(new)} upstream commit(s) touch {path}/ since {ref.what} {ref.commit[:9]}:"
        )
        findings += [f"    {c['sha'][:9]} {subject(c['commit']['message'])}" for c in new]

    crate = get_json(f"https://crates.io/api/v1/crates/{name}")
    if crate:
        latest = crate["crate"]["max_stable_version"] or crate["crate"]["max_version"]
        if version_key(latest) > version_key(ref.version):
            findings.append(f"crates.io has {name} {latest} ({ref.what} {ref.version})")

    advisories = get_list(
        f"https://api.github.com/repos/rustsec/advisory-db/contents/crates/{name}"
    )
    findings += [f"RustSec advisory: {a['name']} ({a['html_url']})" for a in advisories]
    return name, findings


def main() -> int:
    """Check every vendored crate; return the exit status described above."""
    manifests = sorted(ROOT.glob("crates/*/UPSTREAM.toml"))
    if not manifests:
        print("no vendored crates found")
        return 0
    status = 0
    for manifest in manifests:
        try:
            name, findings = check(manifest)
        except (
            urllib.error.URLError,
            TimeoutError,
            KeyError,
            TypeError,
            ValueError,
            LookupFailedError,
        ) as e:
            print(f"{manifest.parent.name}: lookup failed: {type(e).__name__}: {e}")
            status = max(status, 2)
            continue
        if findings:
            print(f"{name}:")
            for f in findings:
                print(f"  {f}")
            status = max(status, 1)
        else:
            print(f"{name}: nothing new upstream")
    return status


if __name__ == "__main__":
    sys.exit(main())
