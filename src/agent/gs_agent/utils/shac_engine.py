"""Load only the task's private Genesis copy, before any Genesis import."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def activate() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[4]
    engine = root / ".deps/genesis"
    marker = engine / ".shac-patch.json"
    patch = root / "patches/genesis-v1.4.5-walk-output-grad.patch"
    metadata = json.loads(marker.read_text())
    if metadata["patch_sha256"] != hashlib.sha256(patch.read_bytes()).hexdigest():
        raise RuntimeError("Engine patch changed; run examples/prepare_shac_engine.py first")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=engine, text=True).strip()
    if commit != "1b6dfa927f3836d49cf80df7a024116499e68d15" or metadata["upstream"] != commit:
        raise RuntimeError("Private Genesis engine is not the pinned upstream revision")
    expected = json.loads((root / "patches/genesis-v1.4.5-source-hashes.json").read_text())
    if set(metadata.get("source_hashes", {})) != set(expected):
        raise RuntimeError("Private engine source manifest is incomplete")
    for file, digest in metadata["source_hashes"].items():
        if hashlib.sha256((engine / file).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Private engine changed after preparation: {file}")
    changed = subprocess.check_output(
        ["git", "diff", "HEAD", "--name-only"], cwd=engine, text=True
    ).splitlines()
    if set(changed) != set(expected):
        raise RuntimeError("Private engine has changes outside the validated patch")
    if "genesis" in sys.modules:
        raise RuntimeError("Activate isolated SHAC engine before importing genesis")
    sys.path.insert(0, str(engine))
    os.environ["GS_ENABLE_ZEROCOPY"] = "0"
    return {**metadata, "source": str(engine), "zero_copy": "disabled"}
