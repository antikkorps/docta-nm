"""One open licence for the whole repository: every package declares the
AGPL, and LICENSE holds its full text, so the metadata never contradicts it."""

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Same list as test_versions.py: a second use, not yet worth a shared module.
MANIFESTS = [
    "worldgen/Cargo.toml",
    "orchestrator/pyproject.toml",
    "eval/pyproject.toml",
    "train/pyproject.toml",
]

SPDX = "AGPL-3.0-only"


def manifest_licence(path: str) -> str:
    data = tomllib.loads((ROOT / path).read_text(encoding="utf-8"))
    return (data.get("package") or data["project"])["license"]


def test_every_package_declares_the_open_licence():
    assert {path: manifest_licence(path) for path in MANIFESTS} == dict.fromkeys(MANIFESTS, SPDX)


def test_licence_file_is_the_agpl():
    text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    # Its opening, its network clause and its end: a header alone, or a cut
    # copy, would not do.
    for part in (
        "GNU AFFERO GENERAL PUBLIC LICENSE",
        "Version 3",
        "Remote Network Interaction",
        "END OF TERMS AND CONDITIONS",
    ):
        assert part in text
