import argparse
import json
from pathlib import Path


class Tokenizer:
    def __init__(self, merges=None):
        self.merges = merges or {}
        self.vocab = self._build_vocab()

    @property
    def vocab_size(self):
        return 256 + len(self.merges)

    @staticmethod
    def _pair_count(token_ids):
        counts = {}
        for pair in zip(token_ids, token_ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    @staticmethod
    def _merge(token_ids, pair, token_id):
        merged = []
        index = 0

        while index < len(token_ids):
            if (
                index < len(token_ids) - 1
                and (token_ids[index], token_ids[index + 1]) == pair
            ):
                merged.append(token_id)
                index += 2
            else:
                merged.append(token_ids[index])
                index += 1

        return merged

    @classmethod
    def train(cls, text, vocab_size=276):
        if vocab_size < 256:
            raise ValueError("vocab_size must be at least 256")

        token_ids = list(text.encode("utf-8"))
        merges = {}

        for token_id in range(256, vocab_size):
            pair_counts = cls._pair_count(token_ids)
            if not pair_counts:
                break

            pair = max(pair_counts, key=pair_counts.get)
            token_ids = cls._merge(token_ids, pair, token_id)
            merges[pair] = token_id

        return cls(merges)

    def encode(self, text):
        token_ids = list(text.encode("utf-8"))

        while len(token_ids) >= 2:
            pair_counts = self._pair_count(token_ids)
            pair = min(
                pair_counts,
                key=lambda candidate: self.merges.get(candidate, float("inf")),
            )

            if pair not in self.merges:
                break

            token_ids = self._merge(token_ids, pair, self.merges[pair])

        return token_ids

    def decode(self, token_ids):
        return self.decode_bytes(token_ids).decode("utf-8", errors="replace")

    def decode_bytes(self, token_ids):
        return b"".join(self.vocab[token_id] for token_id in token_ids)

    def save(self, path):
        serialized_merges = [
            {"left": pair[0], "right": pair[1], "token": token_id}
            for pair, token_id in self.merges.items()
        ]
        Path(path).write_text(
            json.dumps({"merges": serialized_merges}, indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path):
        saved = json.loads(Path(path).read_text(encoding="utf-8"))
        merges = {
            (merge["left"], merge["right"]): merge["token"]
            for merge in saved["merges"]
        }
        return cls(merges)

    def _build_vocab(self):
        vocab = {token_id: bytes([token_id]) for token_id in range(256)}
        for (left, right), token_id in self.merges.items():
            vocab[token_id] = vocab[left] + vocab[right]
        return vocab


def main():
    parser = argparse.ArgumentParser(description="Train and save a BPE tokenizer")
    parser.add_argument("input", type=Path, help="UTF-8 training text")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tokenizer.json"),
        help="where to save the tokenizer (default: tokenizer.json)",
    )
    parser.add_argument("--vocab-size", type=int, default=276)
    args = parser.parse_args()

    text = args.input.read_text(encoding="utf-8")
    tokenizer = Tokenizer.train(text, args.vocab_size)
    tokenizer.save(args.output)
    print(f"Saved {tokenizer.vocab_size}-token vocabulary to {args.output}")


if __name__ == "__main__":
    main()
