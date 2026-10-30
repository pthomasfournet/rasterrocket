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

ROOT = pathlib.Path(__file__).resolve().parent.parent


def get_json(url):
    """GET `url` as JSON; `None` on 404."""
    headers = {"User-Agent": "rasterrocket-vendored-check", "Accept": "application/json"}
    token = os.environ.get("GITHUB_TOKEN")
    if token and url.startswith("https://api.github.com/"):
        headers["Authorization"] = f"Bearer {token}"
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=30) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise


def get_paged(url, pages=10):
    """GET every page of a GitHub list endpoint, up to `pages` pages."""
    items = []
    for page in range(1, pages + 1):
        batch = get_json(f"{url}&page={page}") or []
        items += batch
        if len(batch) < 100:
            break
    return items


def version_key(v):
    return tuple(int(p) if p.isdigit() else 0 for p in v.split("-")[0].split("."))


def check(manifest):
    base = tomllib.loads(manifest.read_text())
    name, repo, path = base["crate"], base["repository"], base["path"]
    findings = []

    if "reviewed" in base:
        ref, since, version, what = (
            base["reviewed"],
            base["reviewed_date"],
            base["reviewed_version"],
            "reviewed",
        )
    else:
        ref, version, what = base["commit"], base["version"], "vendored base"
        # The base commit's own timestamp: `date` may be a calendar day,
        # which would also list the base's same-day ancestors.
        commit = get_json(f"https://api.github.com/repos/{repo}/commits/{ref}")
        since = commit["commit"]["committer"]["date"] if commit else base["date"]

    commits = get_paged(
        f"https://api.github.com/repos/{repo}/commits?path={path}&since={since}&per_page=100"
    )
    new = [c for c in commits if c["sha"] not in (ref, base["commit"])]
    if new:
        findings.append(f"{len(new)} upstream commit(s) touch {path}/ since {what} {ref[:9]}:")
        findings += [f"    {c['sha'][:9]} {c['commit']['message'].splitlines()[0]}" for c in new]

    crate = get_json(f"https://crates.io/api/v1/crates/{name}")
    if crate:
        latest = crate["crate"]["max_stable_version"] or crate["crate"]["max_version"]
        if version_key(latest) > version_key(version):
            findings.append(f"crates.io has {name} {latest} ({what} {version})")

    advisories = get_json(
        f"https://api.github.com/repos/rustsec/advisory-db/contents/crates/{name}"
    ) or []
    for a in advisories:
        findings.append(f"RustSec advisory: {a['name']} ({a['html_url']})")

    return name, findings


def main():
    manifests = sorted(ROOT.glob("crates/*/UPSTREAM.toml"))
    if not manifests:
        print("no vendored crates found")
        return 0
    status = 0
    for manifest in manifests:
        try:
            name, findings = check(manifest)
        except (urllib.error.URLError, KeyError, ValueError) as e:
            print(f"{manifest.parent.name}: lookup failed: {e}")
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
