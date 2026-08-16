def pair_count(data):
    count = {}
    for pair in zip(data, data[1:]):
        count[pair] = count.get(pair, 0) + 1
    return count

def merge(ids, pair, idx):
    """Replaces the `pair` with the `idx` whenever it appears in the `ids`"""
    newids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids


vocab_size = 276
num_merges = vocab_size - 256
ids = list(tokens)

merges = {}
for i in range(num_merges):
    stats = pair_count(ids)
    pair = max(stats, key=stats.get)
    idx = 256 + i
    ids = merge(ids, pair, idx)
    print(f"merging pair {pair} into a new token {idx}")
    merges[pair] = idx

def encode(text):
    tokens = list(text.encode("utf-8"))
    while len(tokens) >= 2:
        stats = pair_count(tokens)
        pair = min(stats, key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
            break
        idx = merges[pair]
        tokens = merge(tokens, pair, idx)
    return tokens