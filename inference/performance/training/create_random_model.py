import os
import types
from pathlib import Path

import torch


INFERENCE_ROOT = Path(__file__).resolve().parents[2]
TRAINING_ROOT = INFERENCE_ROOT.parent / "training"
OUTPUT_PATH = INFERENCE_ROOT / "resources" / "random_model.pt"
SEED = 42


def load_training_module():
    module = types.ModuleType("gpt_random_initialization")
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


torch.manual_seed(SEED)
config = load_training_module()
model = config.Model().cpu()
torch.save({"model_state_dict": model.state_dict(), "seed": SEED}, OUTPUT_PATH)
print(f"Saved deterministic random initialization to {OUTPUT_PATH}")
