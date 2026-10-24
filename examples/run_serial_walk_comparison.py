"""Run one paired comparison serially; the parent never initializes CUDA."""

import argparse
import fcntl
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(options: argparse.Namespace) -> None:
    destination = ROOT / options.output
    destination.mkdir(parents=True, exist_ok=False)
    lock_path = Path("/tmp/genesis-walk-comparison-queue.lock")
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for algorithm in ("ppo", "shac"):
            command = [
                sys.executable,
                "-u",
                "examples/run_walk_benchmark.py",
                "--algorithm",
                algorithm,
                "--num-envs",
                str(options.num_envs),
                "--iterations",
                str(options.iterations),
                "--seed",
                str(options.seed),
                "--output",
                str(destination / algorithm),
            ]
            if options.debug:
                command.append("--debug")
            (destination / "queue_status.json").write_text(
                json.dumps({"phase": algorithm, "command": command}, indent=2)
            )
            with (destination / f"{algorithm}.stdout.log").open("w") as stream:
                process = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            if process.returncode:
                (destination / "queue_status.json").write_text(
                    json.dumps(
                        {
                            "phase": "failed",
                            "algorithm": algorithm,
                            "returncode": process.returncode,
                        },
                        indent=2,
                    )
                )
                raise RuntimeError(
                    f"{algorithm} failed; see {destination / f'{algorithm}.stdout.log'}"
                )
        if not options.debug:
            subprocess.run(
                [
                    sys.executable,
                    "examples/compare_walk_runs.py",
                    "--ppo",
                    str(destination / "ppo"),
                    "--shac",
                    str(destination / "shac"),
                    "--output",
                    str(destination / "comparison.json"),
                ],
                cwd=ROOT,
                check=True,
            )
        (destination / "queue_status.json").write_text(
            json.dumps({"phase": "complete", "debug": options.debug}, indent=2)
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--iterations", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--debug", action="store_true")
    run(parser.parse_args())
