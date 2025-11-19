import torch
import asyncio
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from gpt import Model, decode, device, block_size

app = FastAPI()

# --- CORS (required for React frontend) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load model ---
model = Model().to(device)
checkpoint = torch.load("checkpoint.pt", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# --- Shared generation state ---
state = {
    "playing": False,
    "context": torch.zeros((1, 1), dtype=torch.long, device=device),
    "queue": asyncio.Queue(),  # tokens will be pushed here
    "gen_task": None,  # reference to generation loop task
}


# ===============================================================
# BACKGROUND TOKEN GENERATION LOOP
# ===============================================================
async def generation_loop():
    """
    Continuously generate tokens one at a time, respecting play/pause/reset.
    Updates context after each token so generation continues properly.
    """
    try:
        while state["playing"]:
            idx = state["context"]
            # Crop context to avoid position embedding overflow
            idx_cropped = idx[:, -block_size:]
            # Forward pass
            logits, _ = model(idx_cropped)
            logits = logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            # Update context
            idx = torch.cat((idx, idx_next), dim=1)
            state["context"] = idx[:, -block_size:]  # crop to block_size

            # Decode and push to SSE queue
            decoded = decode([idx_next.item()])
            await state["queue"].put(decoded)

            # Throttle token generation
            await asyncio.sleep(0.015)

    except asyncio.CancelledError:
        await asyncio.sleep(0.05)
    except Exception as e:
        print("Generation error:", e)
        await asyncio.sleep(0.05)

    await asyncio.sleep(0.05)


# ===============================================================
# PLAY / PAUSE
# ===============================================================
@app.post("/play")
async def play():
    state["playing"] = True

    # Start generation loop task if not running
    if state.get("gen_task") is None or state["gen_task"].done():
        state["gen_task"] = asyncio.create_task(generation_loop())

    return {"status": "playing"}


@app.post("/pause")
async def pause():
    state["playing"] = False
    return {"status": "paused"}


# ===============================================================
# RESET CONTEXT
# ===============================================================
@app.post("/reset")
async def reset():
    global state
    # Stop generation task
    state["playing"] = False
    if state.get("gen_task") and not state["gen_task"].done():
        state["gen_task"].cancel()
        try:
            await state["gen_task"]
        except asyncio.CancelledError:
            pass

    state = {
        "playing": False,
        "context": torch.zeros((1, 1), dtype=torch.long, device=device),
        "queue": asyncio.Queue(),
        "gen_task": None,
    }

    return {"status": "reset"}


# ===============================================================
# STREAM (SSE)
# ===============================================================
@app.get("/stream")
async def stream():
    async def event_generator():
        while True:
            token = await state["queue"].get()
            # preserve newlines properly for frontend
            safe = token.replace("\n", "\\n")
            print(safe, end="", flush=True)
            yield f"data: {safe}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ===============================================================
# Startup: optional warmup or just rely on play endpoint
# ===============================================================
@app.on_event("startup")
async def startup_event():
    # We don't start generation until play is called
    pass


# ===============================================================
# Run: uvicorn server:app --host 0.0.0.0 --port 8000
# ===============================================================
