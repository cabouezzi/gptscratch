import math
import os
import subprocess
import sys
import time
import types
import uuid
from collections import deque
from datetime import datetime
from pathlib import Path

import torch


INFERENCE_ROOT = Path(__file__).resolve().parent
TRAINING_ROOT = INFERENCE_ROOT.parent / "training"
START_PATH = INFERENCE_ROOT / "resources" / "5bae56b2-ea62-4365-bb5c-000b23e70c67.pt"
BATCH_SIZE = 64
SEQUENCE_LENGTH = 256
POPULATION_SIZE = 2
RANK = 1
SIGMA = 0.0001
LEARNING_RATE = 0.0001
SEED = 42
REPORT_INTERVAL_SECONDS = 60
ROLLING_WINDOW_SECONDS = 5 * 3600
CHECKPOINT_INTERVAL_SECONDS = 600
EVALUATION_BATCH_COUNT = 5
MINIMUM_VALIDATION_IMPROVEMENT = 0.001
EARLY_STOPPING_PATIENCE_REPORTS = 60


def load_training_module():
    module = types.ModuleType("gpt_eggroll_training")
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


def make_batch(data, generator):
    starts = torch.randint(len(data) - SEQUENCE_LENGTH, (BATCH_SIZE,), generator=generator)
    tokens = torch.stack([data[start : start + SEQUENCE_LENGTH] for start in starts]).to("mps")
    targets = torch.stack([data[start + 1 : start + SEQUENCE_LENGTH + 1] for start in starts]).to("mps")
    return tokens, targets


def make_population(linears, generator):
    candidate_size = sum((linear.weight.shape[0] + linear.weight.shape[1]) * RANK for linear in linears)
    random_values = torch.randn(POPULATION_SIZE * candidate_size, generator=generator).to("mps")
    population = []
    offset = 0
    for _ in range(POPULATION_SIZE):
        candidate = []
        for linear in linears:
            output_size, input_size = linear.weight.shape
            A = random_values[offset : offset + output_size * RANK].reshape(output_size, RANK)
            offset += output_size * RANK
            B = random_values[offset : offset + input_size * RANK].reshape(input_size, RANK)
            offset += input_size * RANK
            candidate.append((A, B))
        population.append(candidate)
    return population


torch.manual_seed(SEED)
torch.mps.manual_seed(SEED)
config = load_training_module()
checkpoint = torch.load(START_PATH, map_location="cpu", weights_only=True)
model = config.Model().to("mps")
model.load_state_dict(checkpoint["model_state_dict"])
model.train()
linears = [module for module in model.modules() if isinstance(module, torch.nn.Linear)]
active_perturbations = {}
active_epsilon = 0.0
rank_scale = 1.0 / math.sqrt(RANK)


def perturbation_hook(module, inputs, output):
    perturbation = active_perturbations.get(module)
    if perturbation is None:
        return output
    A, B = perturbation
    return output + active_epsilon * rank_scale * inputs[0].matmul(B).matmul(A.transpose(0, 1))


def make_evaluation_batches(data, seed):
    generator = torch.Generator().manual_seed(seed)
    return [make_batch(data, generator) for _ in range(EVALUATION_BATCH_COUNT)]


@torch.no_grad()
def evaluate(batches):
    model.eval()
    losses = []
    for tokens, targets in batches:
        _, loss = model(tokens, targets)
        losses.append(loss)
    model.train()
    return torch.stack(losses).mean().item()


def save_checkpoint(path, completed_steps, validation_loss):
    torch.save({"model_state_dict": model.state_dict(), "steps": completed_steps, "validation_loss": validation_loss, "algorithm": "EGGROLL"}, path)


hooks = [linear.register_forward_hook(perturbation_hook) for linear in linears]
starting_step = int(checkpoint.get("steps", 0))
data_generator = torch.Generator().manual_seed(SEED + starting_step)
perturbation_generator = torch.Generator().manual_seed(SEED + starting_step)
training_evaluation_batches = make_evaluation_batches(config.train_data, SEED + 1)
validation_evaluation_batches = make_evaluation_batches(config.val_data, SEED + 2)
run_id = str(uuid.uuid4())
checkpoint_path = INFERENCE_ROOT / "resources" / f"{run_id}.pt"
best_checkpoint_path = INFERENCE_ROOT / "resources" / f"{run_id}-best.pt"
gguf_path = INFERENCE_ROOT / "resources" / f"{run_id}.gguf"
log_path = INFERENCE_ROOT / "builddir" / f"eggroll-{run_id}.log"


def report(message):
    print(message, flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(message + "\n")


report(f"Training model {run_id}")
report(f"Monitoring log: {log_path}")
report(f"Resuming from step {starting_step} using {START_PATH}")

start_time = time.perf_counter()
next_report_time = start_time + REPORT_INTERVAL_SECONDS
next_checkpoint_time = start_time + CHECKPOINT_INTERVAL_SECONDS
loss_history = deque()
rolling_loss_sum = 0.0
completed_steps = starting_step
interrupted = False
converged = False
initial_validation_loss = evaluate(validation_evaluation_batches)
best_validation_loss = initial_validation_loss
latest_validation_loss = initial_validation_loss
best_step = starting_step
reports_without_improvement = 0
save_checkpoint(best_checkpoint_path, best_step, best_validation_loss)
report(f"Initial validation loss {initial_validation_loss:.6f}")

try:
    with torch.no_grad():
        step = starting_step
        while True:
            tokens, targets = make_batch(config.train_data, data_generator)
            active_perturbations = {}
            active_epsilon = 0.0
            _, baseline_loss = model(tokens, targets)
            population = make_population(linears, perturbation_generator)
            positive_losses = []
            negative_losses = []

            for candidate_index, candidate in enumerate(population):
                active_perturbations = dict(zip(linears, candidate))
                dropout_seed = SEED + step * POPULATION_SIZE + candidate_index
                torch.manual_seed(dropout_seed)
                torch.mps.manual_seed(dropout_seed)
                active_epsilon = SIGMA
                _, positive_loss = model(tokens, targets)
                torch.manual_seed(dropout_seed)
                torch.mps.manual_seed(dropout_seed)
                active_epsilon = -SIGMA
                _, negative_loss = model(tokens, targets)
                positive_losses.append(positive_loss)
                negative_losses.append(negative_loss)

            active_perturbations = {}
            active_epsilon = 0.0
            loss_differences = torch.stack(negative_losses) - torch.stack(positive_losses)
            signs = torch.sign(loss_differences).cpu().tolist()
            update_scale = LEARNING_RATE / POPULATION_SIZE * rank_scale
            for linear_index, linear in enumerate(linears):
                update_A = torch.cat([population[index][linear_index][0] * signs[index] for index in range(POPULATION_SIZE)], dim=1)
                update_B = torch.cat([population[index][linear_index][1] for index in range(POPULATION_SIZE)], dim=1)
                linear.weight.addmm_(update_A, update_B.transpose(0, 1), beta=1.0, alpha=update_scale)

            completed_steps = step + 1
            now = time.perf_counter()
            loss_value = baseline_loss.item()
            loss_history.append((now, loss_value))
            rolling_loss_sum += loss_value
            cutoff = now - ROLLING_WINDOW_SECONDS
            while loss_history and loss_history[0][0] < cutoff:
                _, expired_loss = loss_history.popleft()
                rolling_loss_sum -= expired_loss
            if now >= next_report_time:
                active_perturbations = {}
                active_epsilon = 0.0
                training_loss = evaluate(training_evaluation_batches)
                validation_loss = evaluate(validation_evaluation_batches)
                latest_validation_loss = validation_loss
                rolling_loss = rolling_loss_sum / len(loss_history)
                elapsed_hours = (now - start_time) / 3600.0
                if validation_loss < best_validation_loss - MINIMUM_VALIDATION_IMPROVEMENT:
                    best_validation_loss = validation_loss
                    best_step = completed_steps
                    reports_without_improvement = 0
                    save_checkpoint(best_checkpoint_path, best_step, best_validation_loss)
                else:
                    reports_without_improvement += 1
                report(f"[{datetime.now().isoformat(timespec='seconds')}] Step {completed_steps} | rolling 5h loss {rolling_loss:.6f} over {len(loss_history)} steps | train loss {training_loss:.6f} | validation loss {validation_loss:.6f} | best validation {best_validation_loss:.6f} at step {best_step} | patience {reports_without_improvement}/{EARLY_STOPPING_PATIENCE_REPORTS} | elapsed {elapsed_hours:.2f} hours")
                while next_report_time <= now:
                    next_report_time += REPORT_INTERVAL_SECONDS
                if reports_without_improvement >= EARLY_STOPPING_PATIENCE_REPORTS:
                    converged = True
                    report(f"Validation loss stopped improving; restoring the best checkpoint from step {best_step}")
                    break
            if now >= next_checkpoint_time:
                save_checkpoint(checkpoint_path, completed_steps, latest_validation_loss)
                report(f"Checkpoint saved to {checkpoint_path}")
                while next_checkpoint_time <= now:
                    next_checkpoint_time += CHECKPOINT_INTERVAL_SECONDS
            step += 1
except KeyboardInterrupt:
    interrupted = True
    report(f"Stopping after step {completed_steps}")
finally:
    active_perturbations = {}
    active_epsilon = 0.0
    for hook in hooks:
        hook.remove()

torch.mps.synchronize()
elapsed = time.perf_counter() - start_time
save_checkpoint(checkpoint_path, completed_steps, latest_validation_loss)
export_checkpoint_path = checkpoint_path if interrupted else best_checkpoint_path
subprocess.run([sys.executable, str(INFERENCE_ROOT / "scripts" / "export_gguf.py"), str(export_checkpoint_path), str(gguf_path)], check=True)
status = "interrupted" if interrupted else "converged" if converged else "complete"
report(f"EGGROLL training {status}: {elapsed:.6f} seconds")
report(f"Best validation loss {best_validation_loss:.6f} at step {best_step}")
report(f"Saved trained model to {gguf_path}")
