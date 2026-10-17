"""Focused cache progress tests, using real disk payloads and tqdm output."""
import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace

from modules.dataLoader.moss.MOSSConditioningCache import MOSSConditioningCache

import torch


class ProgressTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = io.StringIO()
        self.summary = io.StringIO()
        self.prepared = 0
        self.encoded = []

    def payload(self):
        return {"tokens": torch.ones(8, dtype=torch.long), "tokens_mask": torch.ones(8),
                "text_encoder_hidden_state": torch.ones(8, 4)}

    def cache(self, rows, **kwargs):
        def prepare():
            self.assertIn("preparing encoder", self.output.getvalue())
            self.prepared += 1

        def encode(prompt):
            self.assertIn("stage=encoding", self.output.getvalue())
            self.encoded.append(prompt)
            return self.payload()

        cache = MOSSConditioningCache(SimpleNamespace(empty_text={}), 8, self.root,
                                      {"encoder": "test"}, prepare, **kwargs)
        cache._get_previous_length = lambda name: len(rows)
        cache._get_previous_item = lambda variation, name, index: rows[index]
        cache._encode = encode
        return cache

    def test_progress_counts_rows_while_reusing_duplicates_and_disk_hits(self):
        cache = self.cache(["new", "cached", "new"])
        cache._save(self.root / (cache._key("cached") + ".pt"), self.payload())
        with redirect_stderr(self.output), redirect_stdout(self.summary):
            cache.start(0)
        self.assertIn("caching MOSS text", self.output.getvalue())
        self.assertIn("3/3", self.output.getvalue())
        self.assertIn("100%", self.output.getvalue())
        self.assertIn("encoded=1", self.output.getvalue())
        self.assertIn("reused=1", self.output.getvalue())
        self.assertEqual(self.encoded, ["new"])
        self.assertEqual(self.prepared, 1)
        self.assertEqual(len(cache.entries), 3)
        self.assertIn("rows=3, unique=2, encoded=1, reused=1", self.summary.getvalue())

    def test_existing_cache_paths_and_payloads_are_unchanged(self):
        rows = ["first", "second"]
        cache = self.cache(rows)
        for prompt in rows:
            expected = hashlib.sha256(json.dumps(
                [{"encoder": "test"}, 8, prompt], sort_keys=True, ensure_ascii=False
            ).encode()).hexdigest()
            self.assertEqual(cache._key(prompt), expected)
            cache._save(self.root / (expected + ".pt"), self.payload())
        before = {p.name: (p.stat().st_mtime_ns, p.read_bytes()) for p in self.root.glob("*.pt")}
        with redirect_stderr(self.output), redirect_stdout(self.summary):
            cache.start(0)
        self.assertEqual(self.prepared, 0)
        self.assertEqual(self.encoded, [])
        self.assertIn("2/2", self.output.getvalue())
        self.assertIn("encoded=0", self.output.getvalue())
        self.assertIn("reused=2", self.output.getvalue())
        self.assertEqual(before, {p.name: (p.stat().st_mtime_ns, p.read_bytes())
                                  for p in self.root.glob("*.pt")})

    def test_encoding_error_is_propagated_without_claiming_completion(self):
        cache = self.cache(["first"])

        def fail(prompt):
            raise RuntimeError("test encoder failure")

        cache._encode = fail
        with redirect_stderr(self.output), redirect_stdout(self.summary):
            with self.assertRaisesRegex(RuntimeError, "test encoder failure"):
                cache.start(0)
        self.assertFalse(cache.initialized)
        self.assertIn("stage=encoding", self.output.getvalue())
        self.assertNotIn("[OT-MOSS-TEXT-CACHE]", self.summary.getvalue())
        self.assertEqual(list(self.root.glob("*.pt")), [])

    def test_disabled_cache_does_not_display_a_caching_bar(self):
        cache = self.cache(["first"], enabled=False)
        with redirect_stderr(self.output), redirect_stdout(self.summary):
            cache.start(0)
        self.assertEqual(self.output.getvalue(), "")
        self.assertEqual(self.encoded, [])


if __name__ == "__main__":
    unittest.main()
