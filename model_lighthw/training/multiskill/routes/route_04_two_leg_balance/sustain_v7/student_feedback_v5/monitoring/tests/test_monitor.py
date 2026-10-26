"""Small file-tail contract checks; no simulator or training process."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from monitor import JsonlTail, format_status  # noqa: E402


class MonitorTests(unittest.TestCase):
    def test_partial_line_is_not_reported_until_newline(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / "request.json").write_text('{"new_updates": 1000}')
            (run / "session.json").write_text('{"status": "running"}')
            path = run / "updates.jsonl"
            path.write_bytes(b'{"local_update": 1, "global_update": 35, "actual_critic_adam_steps": 70}\n'
                             b'{"local_update": 2, "global_update": 36')
            tail = JsonlTail(path)
            tail.poll()
            self.assertEqual(tail.row["local_update"], 1)
            self.assertEqual(tail.problem, "等待更新行写完")
            first_offset = tail.offset
            self.assertIn("本次更新=1/1000", format_status(run, tail))
            with path.open("ab") as stream:
                stream.write(b', "actual_critic_adam_steps": 72}\n')
            tail.poll()
            self.assertGreater(tail.offset, first_offset)
            self.assertEqual(tail.row["local_update"], 2)
            self.assertEqual(tail.row["actual_critic_adam_steps"], 72)
            self.assertIsNone(tail.problem)

    def test_missing_then_incremental_append_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            tail = JsonlTail(run / "updates.jsonl")
            tail.poll()
            self.assertIsNone(tail.row)
            self.assertIn("等待", format_status(run, tail))
            (run / "updates.jsonl").write_text(json.dumps({"local_update": 1}) + "\n")
            tail.poll()
            self.assertEqual(tail.row["local_update"], 1)
            temporary = run / "updates.moved"
            (run / "updates.jsonl").rename(temporary)
            tail.poll()
            self.assertEqual(tail.row["local_update"], 1)
            self.assertEqual(tail.problem, "等待 updates.jsonl")
            temporary.rename(run / "updates.jsonl")
            tail.poll()
            self.assertIsNone(tail.problem)
            (run / "checkpoints").mkdir()
            (run / "checkpoints" / "model_000134.pt").touch()
            (run / "checkpoints" / "model_000034.pt").touch()
            self.assertIn("checkpoint=model_000134.pt", format_status(run, tail))


if __name__ == "__main__":
    unittest.main()
