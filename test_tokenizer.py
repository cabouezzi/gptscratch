import tempfile
import unittest
from pathlib import Path

from tokenizer import Tokenizer


class TokenizerTest(unittest.TestCase):
    def test_round_trip(self):
        text = "Hello, tokenizer! 😄"
        tokenizer = Tokenizer.train(text, vocab_size=276)

        self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)
        self.assertEqual(
            tokenizer.decode_bytes(tokenizer.encode(text)), text.encode("utf-8")
        )

    def test_save_and_load(self):
        text = "the quick brown fox jumps over the lazy dog"
        tokenizer = Tokenizer.train(text, vocab_size=276)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokenizer.json"
            tokenizer.save(path)
            loaded = Tokenizer.load(path)

        self.assertEqual(loaded.encode(text), tokenizer.encode(text))
        self.assertEqual(loaded.decode(loaded.encode(text)), text)


if __name__ == "__main__":
    unittest.main()
