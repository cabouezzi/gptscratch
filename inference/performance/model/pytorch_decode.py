import argparse
import gc
import os
import statistics
import time
import types
from pathlib import Path

import torch


INFERENCE_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = INFERENCE_ROOT.parent
TRAINING_ROOT = REPOSITORY_ROOT / "training"


def load_training_module():
    module = types.ModuleType("gpt_runtime")
    module.__dict__["__file__"] = str(TRAINING_ROOT / "gpt.py")
    source = (TRAINING_ROOT / "gpt.py").read_text(encoding="utf-8")
    source = source.replace('losses["val"]', "losses['val']")
    previous_directory = Path.cwd()
    try:
        os.chdir(TRAINING_ROOT)
        exec(compile(source, module.__dict__["__file__"], "exec"), module.__dict__)
    finally:
        os.chdir(previous_directory)
    return module


class CachedModel(torch.nn.Module):
    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.config = config
        head_size = config.n_embed // config.num_head
        cache_shape = (
            config.n_layer,
            config.num_head,
            1,
            config.block_size,
            head_size,
        )
        cache_dtype = next(model.parameters()).dtype
        self.register_buffer(
            "key_cache",
            torch.empty(cache_shape, device=config.device, dtype=cache_dtype),
        )
        self.register_buffer(
            "value_cache",
            torch.empty(cache_shape, device=config.device, dtype=cache_dtype),
        )

    def forward(self, tokens, start_position):
        token_count = tokens.shape[1]
        end_position = start_position + token_count
        positions = torch.arange(start_position, end_position, device=self.config.device)
        x = self.model.token_embedding_table(tokens)
        x = x + self.model.position_embedding_table(positions)

        for layer_index in range(self.config.n_layer):
            block = self.model.blocks[layer_index]
            x = block.ln1(x)
            head_outputs = []
            for head_index, head in enumerate(block.sa.heads):
                query = head.query(x)
                key = head.key(x)
                value = head.value(x)
                self.key_cache[
                    layer_index, head_index, :, start_position:end_position, :
                ] = key
                self.value_cache[
                    layer_index, head_index, :, start_position:end_position, :
                ] = value
                cached_keys = self.key_cache[
                    layer_index, head_index, :, :end_position, :
                ]
                cached_values = self.value_cache[
                    layer_index, head_index, :, :end_position, :
                ]

                weights = query @ cached_keys.transpose(-2, -1)
                weights = weights * (self.config.n_embed**-0.5)
                if token_count > 1:
                    query_positions = positions.view(1, token_count, 1)
                    key_positions = torch.arange(
                        end_position, device=self.config.device
                    ).view(1, 1, end_position)
                    weights = weights.masked_fill(
                        key_positions > query_positions, float("-inf")
                    )
                weights = torch.nn.functional.softmax(weights, dim=-1)
                head_outputs.append(weights @ cached_values)

            attention = torch.cat(head_outputs, dim=-1)
            x = x + block.sa.proj(attention)
            x = block.ln2(x)
            x = x + block.ffwd(x)

        x = self.model.blocks[self.config.n_layer](x)
        return self.model.lm_head(x)


class PackedCachedModel(torch.nn.Module):
    def __init__(self, model, config, use_sdpa):
        super().__init__()
        self.model = model
        self.config = config
        self.use_sdpa = use_sdpa
        self.head_size = config.n_embed // config.num_head

        qkv_weights = []
        for layer_index in range(config.n_layer):
            heads = model.blocks[layer_index].sa.heads
            query_weight = torch.cat([head.query.weight for head in heads], dim=0)
            key_weight = torch.cat([head.key.weight for head in heads], dim=0)
            value_weight = torch.cat([head.value.weight for head in heads], dim=0)
            qkv_weights.append(
                torch.cat([query_weight, key_weight, value_weight], dim=0)
            )
        self.register_buffer("qkv_weights", torch.stack(qkv_weights))

        cache_shape = (
            config.n_layer,
            1,
            config.num_head,
            config.block_size,
            self.head_size,
        )
        cache_dtype = next(model.parameters()).dtype
        self.register_buffer(
            "key_cache",
            torch.empty(cache_shape, device=config.device, dtype=cache_dtype),
        )
        self.register_buffer(
            "value_cache",
            torch.empty(cache_shape, device=config.device, dtype=cache_dtype),
        )

    def forward(self, tokens, start_position):
        token_count = tokens.shape[1]
        end_position = start_position + token_count
        positions = torch.arange(start_position, end_position, device=self.config.device)
        x = self.model.token_embedding_table(tokens)
        x = x + self.model.position_embedding_table(positions)

        for layer_index in range(self.config.n_layer):
            block = self.model.blocks[layer_index]
            x = block.ln1(x)
            qkv = torch.nn.functional.linear(x, self.qkv_weights[layer_index])
            qkv = qkv.view(
                1,
                token_count,
                3,
                self.config.num_head,
                self.head_size,
            ).permute(2, 0, 3, 1, 4)
            query, key, value = qkv.unbind(0)

            self.key_cache[layer_index, :, :, start_position:end_position, :] = key
            self.value_cache[
                layer_index, :, :, start_position:end_position, :
            ] = value
            cached_keys = self.key_cache[layer_index, :, :, :end_position, :]
            cached_values = self.value_cache[layer_index, :, :, :end_position, :]

            if self.use_sdpa:
                attention = torch.nn.functional.scaled_dot_product_attention(
                    query,
                    cached_keys,
                    cached_values,
                    is_causal=token_count > 1,
                    scale=self.config.n_embed**-0.5,
                )
            else:
                weights = query @ cached_keys.transpose(-2, -1)
                weights = weights * (self.config.n_embed**-0.5)
                if token_count > 1:
                    query_positions = positions.view(1, 1, token_count, 1)
                    key_positions = torch.arange(
                        end_position, device=self.config.device
                    ).view(1, 1, 1, end_position)
                    weights = weights.masked_fill(
                        key_positions > query_positions, float("-inf")
                    )
                weights = torch.nn.functional.softmax(weights, dim=-1)
                attention = weights @ cached_values

            attention = attention.transpose(1, 2).contiguous()
            attention = attention.view(1, token_count, self.config.n_embed)
            x = x + block.sa.proj(attention)
            x = block.ln2(x)
            x = x + block.ffwd(x)

        x = self.model.blocks[self.config.n_layer](x)
        return self.model.lm_head(x)


def generate(forward, context, token_count):
    logits = forward(context, 0)
    generated = []
    for generated_index in range(token_count):
        next_token = logits[:, -1, :].argmax(dim=1, keepdim=True)
        generated.append(next_token)
        if generated_index + 1 < token_count:
            logits = forward(next_token, context.shape[1] + generated_index)
    return torch.cat(generated, dim=1).squeeze(0).tolist()


def synchronize(device):
    if device == "mps":
        torch.mps.synchronize()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=(
            "eager",
            "aot-eager",
            "packed",
            "packed-sdpa",
            "packed-sdpa-compile",
            "packed-fp16",
            "packed-sdpa-fp16",
            "packed-sdpa-fp16-compile",
        ),
        required=True,
    )
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--tokens", type=int, default=200)
    parser.add_argument("--verify", action="store_true")
    arguments = parser.parse_args()

    config = load_training_module()
    model = config.Model().to(config.device)
    checkpoint = torch.load(REPOSITORY_ROOT / "checkpoint.pt", map_location=config.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    packed = arguments.mode.startswith("packed")
    if packed and "fp16" in arguments.mode:
        model = model.half()
    cached_model = (
        PackedCachedModel(model, config, use_sdpa="sdpa" in arguments.mode)
        if packed
        else CachedModel(model, config)
    ).eval()

    forward = cached_model
    if arguments.mode == "aot-eager":
        forward = torch.compile(cached_model, backend="aot_eager", dynamic=True)
    elif arguments.mode in (
        "packed-sdpa-compile",
        "packed-sdpa-fp16-compile",
    ):
        forward = torch.compile(cached_model, dynamic=True)

    prompt = "ROMEO:"
    context = torch.tensor(
        [config.encode(prompt)], dtype=torch.long, device=config.device
    )

    with torch.inference_mode():
        if arguments.verify and packed:
            reference = CachedModel(model, config).eval()
            reference_logits = reference(context, 0)
            packed_logits = cached_model(context, 0)
            difference = (
                reference_logits.float() - packed_logits.float()
            ).abs().max().item()
            print(f"maximum logit difference: {difference:.8f}")

        generate(forward, context, 10)
        synchronize(config.device)

        elapsed_times = []
        final_tokens = []
        for sample in range(arguments.samples):
            start = time.perf_counter()
            final_tokens = generate(forward, context, arguments.tokens)
            synchronize(config.device)
            elapsed_times.append(time.perf_counter() - start)
            print(
                f"Completed sample {sample + 1}/{arguments.samples}: "
                f"{elapsed_times[-1]:.6f} seconds"
            )
            gc.collect()
            if config.device == "mps":
                torch.mps.empty_cache()

    print(f"mode: {arguments.mode}")
    print(f"device: {config.device}")
    print(f"median: {statistics.median(elapsed_times):.6f} seconds")
    print(f"mean: {statistics.mean(elapsed_times):.6f} seconds")
    print(f"minimum: {min(elapsed_times):.6f} seconds")
    print(f"maximum: {max(elapsed_times):.6f} seconds")
    if arguments.mode == "aot-eager" or arguments.mode.endswith("-compile"):
        print(f"graphs: {torch._dynamo.utils.counters['stats']['unique_graphs']}")
    print(prompt + config.decode(final_tokens))


if __name__ == "__main__":
    main()
