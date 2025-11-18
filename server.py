import asyncio
import torch
from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse
from gpt import decode, Model, device

app = FastAPI()

# --- Load model ---
checkpoint_path = "checkpoint.pt"
model = Model().to(device)
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# --- Global state ---
is_playing = False
clients = []      # list of queues

@app.post("/play")
async def play():
    global is_playing
    is_playing = True
    return {"status": "playing"}

@app.post("/pause")
async def pause():
    global is_playing
    is_playing = False
    return {"status": "paused"}

@app.get("/stream")
async def stream():
    """
    Stream tokens via Server-Sent Events (SSE).
    Each client gets its own asyncio.Queue so broadcast is safe.
    """
    queue = asyncio.Queue()
    clients.append(queue)

    async def event_gen():
        try:
            while True:
                token = await queue.get()
                yield {"data": token}
        except asyncio.CancelledError:
            pass
        finally:
            clients.remove(queue)

    return EventSourceResponse(event_gen())


async def token_generator():
    """
    Background task that continuously generates tokens and broadcasts them.
    Only pushes tokens when is_playing == True
    """
    context = torch.zeros((1, 1), dtype=torch.long, device=device)

    while True:
        # Block until play is true
        if not is_playing:
            await asyncio.sleep(0.1)
            continue

        with torch.no_grad():
            output = model.generate(context, max_tokens=1, stream=True)
            for token in output:
                text = decode([token.item()])
                # broadcast to all connected clients
                for q in clients:
                    q.put_nowait(text)
        await asyncio.sleep(0)  # yield control


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(token_generator())


# run with uvicorn server:app --host 0.0.0.0 --port 8000