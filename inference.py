import codecs

import torch
from gpt import encode, Model, device, tokenizer

checkpoint_path = "checkpoint.pt"

model = Model()
model.to(device)

checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

context = torch.tensor([encode("\n")], dtype=torch.long, device=device)
with torch.no_grad():
    output = model.generate(context, max_tokens=1000, stream=True)
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    with open("output.txt", "a", encoding="utf-8") as f:
        for token in output:
            text = decoder.decode(tokenizer.decode_bytes([token.item()]))
            print(text, end="", flush=True)
            f.write(text)
        text = decoder.decode(b"", final=True)
        print(text, end="", flush=True)
        f.write(text)
        f.write("\n---\n")

print("\nOutput written to output.txt")
