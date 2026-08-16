import torch
import torch.nn as nn
from torch.nn import functional as F

block_size = 8
batch_size = 32
max_iters = 3000
eval_interval = 300
learning_rate = 1e-3
device = "mps" if torch.backends.mps.is_available() else "cpu"
eval_iters = 200
n_embed = 32

torch.manual_seed(1337)


with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
print(" ".join(chars))
print(vocab_size)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)
print(data[:20])

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split="train"):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x = x.to(device)
    y = y.to(device)
    return x, y


@torch.no_grad
def estimate_loss():
    out = {}
    m.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = m(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    m.train()
    return out


xb, yb = get_batch()
print(f"Input: {xb.shape}\n{xb}")
print(f"Output: {yb.shape}\n{yb}")

for b in range(batch_size):
    for t in range(block_size):
        context = xb[b, : t + 1]
        output = yb[b, t]
        print(f"When the input is {context.tolist()}, output is {output}")


class Bigram(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)

        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)  # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        logits = self.lm_head(x)

        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        else:
            loss = None
        return logits, loss

    def generate(self, idx, max_tokens):
        for _ in range(max_tokens):
            logits, loss = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


m = Bigram()
m = m.to(device)
logits, loss = m(xb, yb)
print(loss)

optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)


for step in range(max_iters):
    xb, yb = get_batch()
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if step % eval_interval == 0:
        losses = estimate_loss()
        print(
            f"Step {step}: Train loss {losses['train']:.4f}\tValidation loss {losses["val"]:.4f}"
        )


context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_tokens=500)[0].tolist()))
