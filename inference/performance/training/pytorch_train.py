import gc
import os
import sys
import time
import types
from pathlib import Path

import torch


INFERENCE_ROOT = Path(__file__).resolve().parents[2]
TRAINING_ROOT = INFERENCE_ROOT.parent / "training"
START_PATH = INFERENCE_ROOT / "resources" / "random_model.pt"
OUTPUT_PATH = INFERENCE_ROOT / "resources" / "pytorch_trained.pt"
STEP_COUNT = 10
SEQUENCE_LENGTH = 256
LEARNING_RATE = 0.0001
SEED = 42
USE_COMPILE = False
USE_FOREACH = True


def load_training_module():
    module = types.ModuleType("gpt_training_benchmark")
    sys.modules[module.__name__] = module
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


def synchronize():
    torch.mps.synchronize()


def make_batch(config, step):
    maximum_start = len(config.train_data) - SEQUENCE_LENGTH - 1
    start = (SEED + step * 9973) % (maximum_start + 1)
    tokens = config.train_data[start : start + SEQUENCE_LENGTH].unsqueeze(0).to("mps")
    targets = config.train_data[start + 1 : start + SEQUENCE_LENGTH + 1].unsqueeze(0).to("mps")
    return tokens, targets


class LogitsOnly(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, tokens):
        logits, _ = self.model(tokens)
        return logits


torch.manual_seed(SEED)
config = load_training_module()
checkpoint = torch.load(START_PATH, map_location="cpu", weights_only=True)
model = config.Model().to("mps")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
logits_model = LogitsOnly(model)
training_model = torch.compile(logits_model, dynamic=False) if USE_COMPILE else logits_model
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, foreach=USE_FOREACH)

warmup_tokens, warmup_targets = make_batch(config, 0)
for _ in range(3):
    optimizer.zero_grad(set_to_none=True)
    warmup_logits = training_model(warmup_tokens)
    warmup_loss = torch.nn.functional.cross_entropy(warmup_logits.reshape(-1, warmup_logits.shape[-1]), warmup_targets.reshape(-1))
    warmup_loss.backward()
synchronize()

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, foreach=USE_FOREACH)
gc.collect()
torch.mps.empty_cache()
synchronize()

start_time = time.perf_counter()
loss = None
for step in range(STEP_COUNT):
    tokens, targets = make_batch(config, step)
    optimizer.zero_grad(set_to_none=True)
    logits = training_model(tokens)
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
    loss.backward()
    optimizer.step()
    print(f"Step {step} | loss {loss.item():.6f}")
synchronize()
elapsed = time.perf_counter() - start_time

torch.save({"model_state_dict": model.state_dict(), "steps": STEP_COUNT, "loss": loss.detach().cpu()}, OUTPUT_PATH)
mode = "compiled" if USE_COMPILE else "eager"
print(f"PyTorch {mode} MPS training: {elapsed:.6f} seconds")
print(f"Final loss: {loss.item():.6f}")
print(f"Saved trained checkpoint to {OUTPUT_PATH}")
