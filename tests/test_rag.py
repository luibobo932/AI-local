"""Test RAG (rag.py) với embedder giả lập — không cần Ollama thật."""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import llm
import rag


def _fake_embed(self, inputs, model=""):
    """Embedding giả: vector đếm tần suất chữ a-z (deterministic)."""
    single = isinstance(inputs, str)
    items = [inputs] if single else list(inputs)

    def vec(t):
        t = t.lower()
        return [float(t.count(c)) for c in "abcdefghijklmnopqrstuvwxyz"]

    out = [vec(t) for t in items]
    return out[0] if single else out


class RagTest(unittest.TestCase):
    def setUp(self):
        self._orig_embed = llm.LLMClient.embed
        llm.LLMClient.embed = _fake_embed
        self._cwd = os.getcwd()
        self.work = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.work, "src"))
        with open(os.path.join(self.work, "src", "auth.py"), "w") as f:
            f.write("def login(user, password):\n"
                    "    # verify password hash and create session token\n"
                    "    return check_password(user, password)\n")
        with open(os.path.join(self.work, "src", "math_utils.py"), "w") as f:
            f.write("def fibonacci(n):\n    a, b = 0, 1\n"
                    "    for _ in range(n):\n        a, b = b, a + b\n    return a\n")
        os.chdir(self.work)

    def tearDown(self):
        os.chdir(self._cwd)
        llm.LLMClient.embed = self._orig_embed

    def _client(self):
        return llm.LLMClient("x", "m")

    def test_index_and_retrieve(self):
        idx = rag.index_project(".", client=self._client())
        self.assertEqual(len(idx["items"]), 2)
        self.assertEqual(idx["dim"], 26)
        self.assertTrue(os.path.exists(rag.INDEX_PATH))

        hits = rag.retrieve("password login session authentication",
                            client=self._client(), k=2)
        self.assertTrue(hits[0]["file"].endswith("auth.py"))

    def test_cache_reuse(self):
        rag.index_project(".", client=self._client())
        idx2 = rag.index_project(".", client=self._client())  # lần 2 dùng cache
        self.assertEqual(len(idx2["items"]), 2)

    def test_chunking_skips_empty(self):
        chunks = rag.chunk_file(os.path.join(self.work, "src", "auth.py"), self.work)
        self.assertTrue(all(c["text"].strip() for c in chunks))
        self.assertTrue(chunks[0]["file"].endswith("auth.py"))

    def test_format_context_limit(self):
        chunks = [{"file": "a.py", "start": 1, "end": 9, "text": "x" * 5000, "score": 1.0},
                  {"file": "b.py", "start": 1, "end": 9, "text": "y" * 5000, "score": 0.9}]
        ctx = rag.format_context(chunks, max_chars=6000)
        self.assertIn("a.py", ctx)
        self.assertNotIn("b.py", ctx)  # vượt giới hạn -> bị cắt


if __name__ == "__main__":
    unittest.main(verbosity=2)
