"""Create an isolated, hash-checked Genesis v1.4.5 with walk output VJPs."""

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = "1b6dfa927f3836d49cf80df7a024116499e68d15"
ENGINE = ROOT / ".deps/genesis"
PATCH = ROOT / "patches/genesis-v1.4.5-walk-output-grad.patch"


def prepare() -> None:
    if not ENGINE.exists():
        ENGINE.parent.mkdir(exist_ok=True)
        subprocess.run(
            [
                "git",
                "clone",
                "--depth=1",
                "--branch",
                "v1.4.5",
                "https://github.com/Genesis-Embodied-AI/Genesis.git",
                str(ENGINE),
            ],
            check=True,
        )
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ENGINE, text=True).strip()
    if commit != UPSTREAM:
        raise RuntimeError(f"Expected {UPSTREAM}, found {commit}; refusing to patch another engine")
    digest = hashlib.sha256(PATCH.read_bytes()).hexdigest()
    marker = ENGINE / ".shac-patch.json"
    applied = ENGINE / ".shac-applied.patch"
    already_applied = (
        subprocess.run(
            ["git", "apply", "--reverse", "--check", str(PATCH)], cwd=ENGINE, capture_output=True
        ).returncode
        == 0
    )
    if not already_applied:
        if applied.exists():
            subprocess.run(
                ["git", "apply", "--reverse", "--check", str(applied)], cwd=ENGINE, check=True
            )
            subprocess.run(["git", "apply", "--reverse", str(applied)], cwd=ENGINE, check=True)
        hashes = json.loads((ROOT / "patches/genesis-v1.4.5-source-hashes.json").read_text())
        for file, expected in hashes.items():
            actual = hashlib.sha256((ENGINE / file).read_bytes()).hexdigest()
            if actual != expected:
                raise RuntimeError(f"Unexpected engine contents in {file}; refusing to overwrite")
        subprocess.run(["git", "apply", "--check", str(PATCH)], cwd=ENGINE, check=True)
        subprocess.run(["git", "apply", str(PATCH)], cwd=ENGINE, check=True)
    applied.write_bytes(PATCH.read_bytes())
    tracked = json.loads((ROOT / "patches/genesis-v1.4.5-source-hashes.json").read_text())
    source_hashes = {
        file: hashlib.sha256((ENGINE / file).read_bytes()).hexdigest() for file in tracked
    }
    marker.write_text(
        json.dumps(
            {"upstream": commit, "patch_sha256": digest, "source_hashes": source_hashes}, indent=2
        )
        + "\n"
    )
    print(marker.read_text())


if __name__ == "__main__":
    prepare()
