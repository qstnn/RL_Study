"""Read-only progress display for an existing R4a PPO run.

This process never imports or controls the simulator or training launcher.
Run it in a separate terminal; Ctrl+C stops only this display.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import re
import time


_CHECKPOINT = re.compile(r"model_(\d+)\.pt\Z")
_TAIL_CHUNK = 64 * 1024


class JsonlTail:
    """Remember the last complete JSONL row and read only newly appended bytes."""

    def __init__(self, path: Path):
        self.path = path
        self.inode: tuple[int, int] | None = None
        self.offset = 0
        self.partial = b""
        self.row: dict | None = None
        self.row_mtime: float | None = None
        self.problem: str | None = None

    def _accept(self, line: bytes, mtime: float) -> None:
        try:
            value = json.loads(line.decode("utf-8"))
            if not isinstance(value, dict):
                raise ValueError("JSONL row is not an object")
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
            self.problem = "最新完整更新行无效"
            return
        self.row = value
        self.row_mtime = mtime
        self.problem = None

    def _seed(self, stream, size: int, mtime: float) -> None:
        """Search backward once; do not read a long run from its beginning."""
        end = size
        tail = b""
        while end:
            start = max(0, end - _TAIL_CHUNK)
            stream.seek(start)
            tail = stream.read(end - start) + tail
            end = start
            final_newline = tail.rfind(b"\n")
            if final_newline < 0:
                continue
            complete = tail[:final_newline].split(b"\n")
            for index in range(len(complete) - 1, -1, -1):
                line = complete[index]
                if line.strip() and (index > 0 or end == 0):
                    self._accept(line, mtime)
                    self.partial = tail[final_newline + 1 :]
                    self.offset = size
                    if self.partial and self.problem is None:
                        self.problem = "等待更新行写完"
                    return
            if end == 0:
                break
        self.partial = tail if b"\n" not in tail else tail[tail.rfind(b"\n") + 1 :]
        self.offset = size
        self.problem = "等待首条完整更新"

    def poll(self) -> None:
        try:
            with self.path.open("rb") as stream:
                stat = stream.fileno()
                import os

                file_stat = os.fstat(stat)
                identity = (file_stat.st_dev, file_stat.st_ino)
                if self.inode != identity or file_stat.st_size < self.offset:
                    self.inode = identity
                    self.offset = 0
                    self.partial = b""
                    self.row = None
                    self.row_mtime = None
                    self.problem = None
                    self._seed(stream, file_stat.st_size, file_stat.st_mtime)
                    return
                stream.seek(self.offset)
                added = stream.read()
                self.offset += len(added)
        except (FileNotFoundError, PermissionError, IsADirectoryError):
            self.problem = "等待 updates.jsonl"
            return

        if not added:
            self.problem = ("等待更新行写完" if self.partial else
                            "等待首条完整更新" if self.row is None else None)
            return
        data = self.partial + added
        final_newline = data.rfind(b"\n")
        if final_newline < 0:
            self.partial = data
            self.problem = "等待更新行写完"
            return
        complete = data[:final_newline].split(b"\n")
        self.partial = data[final_newline + 1 :]
        for line in reversed(complete):
            if line.strip():
                self._accept(line, file_stat.st_mtime)
                break
        if self.partial and self.problem is None:
            self.problem = "等待更新行写完"


def _read_json(path: Path) -> dict | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, PermissionError, IsADirectoryError, OSError, UnicodeDecodeError,
            json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _latest_checkpoint(run: Path) -> str:
    try:
        candidates = (
            (int(match.group(1)), path.name)
            for path in (run / "checkpoints").iterdir()
            if path.is_file() and (match := _CHECKPOINT.fullmatch(path.name))
        )
        return max(candidates, default=(-1, "尚未保存"))[1]
    except (FileNotFoundError, PermissionError, NotADirectoryError, OSError):
        return "尚未保存"


def format_status(run: Path, tail: JsonlTail, now: float | None = None) -> str:
    now = time.time() if now is None else now
    request = _read_json(run / "request.json") or {}
    session = _read_json(run / "session.json") or {}
    row = tail.row or {}
    total = request.get("new_updates")
    local = row.get("local_update", 0)
    age = "未知" if tail.row_mtime is None else f"约{max(0, int(now - tail.row_mtime))}秒"
    state = session.get("status", "等待 session.json")
    progress = f"{local}/{total if isinstance(total, int) else '?'}"
    note = f"；{tail.problem}" if tail.problem else ""
    return (
        f"{datetime.fromtimestamp(now).astimezone().strftime('%Y-%m-%d %H:%M:%S')} "
        f"状态={state} 本次更新={progress} global={row.get('global_update', '?')} "
        f"Adam={row.get('actual_critic_adam_steps', '?')} 最近更新={age} "
        f"checkpoint={_latest_checkpoint(run)}{note}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path, help="Existing PPO run directory")
    parser.add_argument("--interval", type=float, default=5.0, help="Polling seconds (default: 5)")
    parser.add_argument("--once", action="store_true", help="Print one read-only snapshot and exit")
    args = parser.parse_args(argv)
    if not 0 < args.interval < float("inf"):
        parser.error("--interval must be finite and greater than zero")
    run = args.run.expanduser().resolve()
    tail = JsonlTail(run / "updates.jsonl")
    try:
        while True:
            tail.poll()
            print(format_status(run, tail), flush=True)
            if args.once:
                return 0
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("监控已退出；训练进程未受操作。", flush=True)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
