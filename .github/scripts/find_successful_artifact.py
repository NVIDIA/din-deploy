#!/usr/bin/env python3
"""Find the newest unexpired artifact from a successful workflow run."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen


def github_api(path: str) -> object:
    token = os.environ["GH_TOKEN"]
    request = Request(
        f"https://api.github.com/{path.lstrip('/')}",
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urlopen(request, timeout=30) as response:
        return json.load(response)


def write_outputs(*, found: bool, artifact_name: str, run_id: str = "", as_json: bool = False) -> None:
    if as_json:
        print(json.dumps({"found": found, "name": artifact_name if found else "", "run_id": run_id}))
        return
    output_path = Path(os.environ["GITHUB_OUTPUT"])
    with output_path.open("a", encoding="utf-8") as output:
        output.write(f"found={'true' if found else 'false'}\n")
        output.write(f"name={artifact_name if found else ''}\n")
        output.write(f"run_id={run_id}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", required=True)
    parser.add_argument("--artifact-name", required=True)
    parser.add_argument("--default-branch", required=True)
    parser.add_argument("--event-name", required=True)
    parser.add_argument("--source-branch", required=True)
    parser.add_argument("--pull-request", default="")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    artifact_name = args.artifact_name
    allowed_branches = {args.default_branch}
    if args.event_name != "pull_request" or args.pull_request:
        allowed_branches.add(args.source_branch)
    response = github_api(f"repos/{args.repository}/actions/artifacts?name={quote(artifact_name)}&per_page=100")
    artifacts = sorted(
        (
            artifact
            for artifact in response["artifacts"]
            if artifact["name"] == artifact_name
            and not artifact["expired"]
            and artifact["workflow_run"]["head_branch"] in allowed_branches
        ),
        key=lambda artifact: artifact["created_at"],
        reverse=True,
    )

    conclusions: dict[int, bool] = {}
    for artifact in artifacts:
        run_id = int(artifact["workflow_run"]["id"])
        if run_id not in conclusions:
            run = github_api(f"repos/{args.repository}/actions/runs/{run_id}")
            same_pr = args.pull_request and any(
                str(pr["number"]) == args.pull_request for pr in run.get("pull_requests", [])
            )
            trusted_branch = run["event"] != "pull_request" and run["head_branch"] in allowed_branches
            conclusions[run_id] = (
                (trusted_branch or same_pr) and run["status"] == "completed" and run["conclusion"] == "success"
            )
        if conclusions[run_id]:
            write_outputs(
                found=True,
                artifact_name=artifact_name,
                run_id=str(run_id),
                as_json=args.json,
            )
            return

    write_outputs(found=False, artifact_name=artifact_name, as_json=args.json)


if __name__ == "__main__":
    main()
