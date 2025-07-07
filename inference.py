import torch
from gpt import decode, Model, device

checkpoint_path = "checkpoint.pt"

model = Model()
model.to(device)

checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
with torch.no_grad():
    output = model.generate(context, max_tokens=1000, stream=True)
    with open("output.txt", "a", encoding="utf-8") as f:
        for token in output:
            token = decode([token.item()])
            print(token, end="", flush=True)
            f.write(token)
        f.write("\n---\n")

print("\nOutput written to output.txt")
