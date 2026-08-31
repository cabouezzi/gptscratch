#!/usr/bin/env python3

import argparse
import io
import struct
from pathlib import Path

import torch


GGUF_VERSION = 3
GGML_TYPE_F32 = 0
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_STRING = 8
GGUF_TYPE_UINT64 = 10
ALIGNMENT = 32


def write_string(output: io.BytesIO, value: str) -> None:
    encoded = value.encode("utf-8")
    output.write(struct.pack("<Q", len(encoded)))
    output.write(encoded)


def write_metadata(output: io.BytesIO, key: str, value_type: int, value) -> None:
    write_string(output, key)
    output.write(struct.pack("<I", value_type))

    if value_type == GGUF_TYPE_STRING:
        write_string(output, value)
    elif value_type == GGUF_TYPE_UINT32:
        output.write(struct.pack("<I", value))
    elif value_type == GGUF_TYPE_UINT64:
        output.write(struct.pack("<Q", value))
    elif value_type == GGUF_TYPE_FLOAT32:
        output.write(struct.pack("<f", value))
    else:
        raise ValueError(f"Unsupported metadata type: {value_type}")


def align(value: int) -> int:
    return (value + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    inference_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(
        description="Export a gptscratch PyTorch checkpoint to GGUF v3."
    )
    parser.add_argument(
        "checkpoint",
        nargs="?",
        type=Path,
        default=project_root / "training" / "checkpoint.pt",
    )
    parser.add_argument(
        "output",
        nargs="?",
        type=Path,
        default=inference_root / "resources" / "model.gguf",
    )
    parser.add_argument("--head-count", type=int, default=6)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    tensors = []
    for name, tensor in state_dict.items():
        if name.endswith(".tril"):
            continue
        if len(name.encode("utf-8")) > 64:
            raise ValueError(f"GGUF tensor name exceeds 64 bytes: {name}")

        tensor = tensor.detach().cpu().to(torch.float32).contiguous()
        raw = tensor.numpy().astype("<f4", copy=False).tobytes(order="C")
        gguf_dimensions = list(reversed(tensor.shape))
        if not gguf_dimensions or len(gguf_dimensions) > 4:
            raise ValueError(f"Unsupported tensor rank for {name}: {tensor.shape}")
        tensors.append(
            {
                "name": name,
                "dimensions": gguf_dimensions,
                "data": raw,
            }
        )

    token_embeddings = state_dict["token_embedding_table.weight"]
    position_embeddings = state_dict["position_embedding_table.weight"]
    vocabulary_size, embedding_size = token_embeddings.shape
    context_length = position_embeddings.shape[0]
    block_indices = {
        int(name.split(".")[1])
        for name in state_dict
        if name.startswith("blocks.") and ".sa." in name
    }
    block_count = len(block_indices)

    metadata = [
        ("general.architecture", GGUF_TYPE_STRING, "gptscratch"),
        ("general.name", GGUF_TYPE_STRING, "gptscratch"),
        ("general.alignment", GGUF_TYPE_UINT32, ALIGNMENT),
        ("general.file_type", GGUF_TYPE_UINT32, 0),
        ("gptscratch.vocab_size", GGUF_TYPE_UINT64, vocabulary_size),
        ("gptscratch.context_length", GGUF_TYPE_UINT64, context_length),
        ("gptscratch.embedding_length", GGUF_TYPE_UINT64, embedding_size),
        ("gptscratch.block_count", GGUF_TYPE_UINT64, block_count),
        (
            "gptscratch.feed_forward_length",
            GGUF_TYPE_UINT64,
            embedding_size * 4,
        ),
        (
            "gptscratch.attention.head_count",
            GGUF_TYPE_UINT64,
            args.head_count,
        ),
        (
            "gptscratch.attention.layer_norm_epsilon",
            GGUF_TYPE_FLOAT32,
            1.0e-5,
        ),
        (
            "gptscratch.attention.scale_dimension",
            GGUF_TYPE_UINT64,
            embedding_size,
        ),
    ]

    data_offset = 0
    for tensor in tensors:
        data_offset = align(data_offset)
        tensor["offset"] = data_offset
        data_offset += len(tensor["data"])

    header = io.BytesIO()
    header.write(b"GGUF")
    header.write(struct.pack("<IQQ", GGUF_VERSION, len(tensors), len(metadata)))

    for key, value_type, value in metadata:
        write_metadata(header, key, value_type, value)

    for tensor in tensors:
        write_string(header, tensor["name"])
        header.write(struct.pack("<I", len(tensor["dimensions"])))
        for dimension in tensor["dimensions"]:
            header.write(struct.pack("<Q", dimension))
        header.write(struct.pack("<IQ", GGML_TYPE_F32, tensor["offset"]))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as output:
        header_bytes = header.getvalue()
        output.write(header_bytes)
        output.write(b"\0" * (align(len(header_bytes)) - len(header_bytes)))

        relative_position = 0
        for tensor in tensors:
            padding = tensor["offset"] - relative_position
            output.write(b"\0" * padding)
            output.write(tensor["data"])
            relative_position = tensor["offset"] + len(tensor["data"])

    print(f"Exported {len(tensors)} tensors to {args.output}")


if __name__ == "__main__":
    main()
