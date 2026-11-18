"""One version for the whole repository: every package manifest and the
latest released section of CHANGELOG.md must agree, so a release can never
ship with packages out of step."""

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = [
    "worldgen/Cargo.toml",
    "orchestrator/pyproject.toml",
    "eval/pyproject.toml",
    "train/pyproject.toml",
]


def manifest_version(path: str) -> str:
    data = tomllib.loads((ROOT / path).read_text(encoding="utf-8"))
    return (data.get("package") or data["project"])["version"]


def changelog_version() -> str:
    text = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    released = re.findall(r"^## \[(\d+\.\d+\.\d+)\]", text, re.M)
    assert released, "CHANGELOG.md has no released version"
    return released[0]


def test_every_package_carries_the_released_version():
    assert {path: manifest_version(path) for path in MANIFESTS} == dict.fromkeys(
        MANIFESTS, changelog_version()
    )
