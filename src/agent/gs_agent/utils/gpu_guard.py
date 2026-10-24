"""Coordinate experiments using one physical GPU."""

import csv
import fcntl
import os
import subprocess
from pathlib import Path
from types import TracebackType


class ExclusiveGPU:
    """Hold an advisory lock and reject an already occupied GPU before CUDA starts."""

    def __init__(self, index: int = 0) -> None:
        self.index = index
        self._file = None

    def __enter__(self) -> "ExclusiveGPU":
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible is not None and visible != str(self.index):
            raise ValueError(
                "Use the physical GPU index directly; CUDA_VISIBLE_DEVICES remapping is unsupported"
            )
        path = Path(f"/tmp/genesis-walk-gpu-{self.index}.lock")
        self._file = path.open("a+")
        try:
            fcntl.flock(self._file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            output = subprocess.check_output(
                [
                    "nvidia-smi",
                    f"--id={self.index}",
                    "--query-compute-apps=pid,process_name",
                    "--format=csv,noheader",
                ],
                text=True,
            )
            graphics = {"gnome-remote-desktop-daemon", "Xorg", "Xwayland"}
            occupied = [
                row
                for row in csv.reader(output.splitlines())
                if len(row) >= 2
                and int(row[0]) != os.getpid()
                and Path(row[1].strip()).name not in graphics
            ]
            if occupied:
                raise RuntimeError(f"GPU {self.index} already has compute processes: {occupied}")
        except BaseException:
            self._file.close()
            self._file = None
            raise
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
