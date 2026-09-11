"""Text -> the encoder's input, without the checkpoint (the client's step).

    python encode.py <all_dir> <text.txt> <out_dir> [--tokens T]

One prompt per line of `text.txt`. Each becomes `[CLS] pieces [SEP]`, cut
to T and padded with `[PAD]`; the word + position + type embeddings and the
embedding LayerNorm are applied in the clear (export.py dumped the tables
and `vocab.txt`). Writes `inputs.f32` [N, T, H], `mask.u8` [N, T] (1 = real
token) and `ids.txt` -- exactly what `export.py --prompts` writes, so the
same crypto run reads both.
"""

import argparse
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("all_dir")
    ap.add_argument("text")
    ap.add_argument("out")
    ap.add_argument("--tokens", type=int, default=0)
    a = ap.parse_args()
    meta = json.load(open(os.path.join(a.all_dir, "meta.json")))
    H, eps = meta["channels"], meta["ln_eps"]
    T = a.tokens or meta["tokens"]
    V = meta["vocab"]

    def load(name, shape):
        arr = np.fromfile(os.path.join(a.all_dir, name), dtype=np.float32)
        return arr.astype(np.float64).reshape(shape)

    we = load("emb_word.f32", (V, H))
    wp = load("emb_pos.f32", (meta["max_position"], H))
    wt = load("emb_type.f32", (-1, H))
    g, b = load("emb_norm.f32", (H,)), load("emb_norm_bias.f32", (H,))

    from tokenizers import BertWordPieceTokenizer

    tok = BertWordPieceTokenizer(os.path.join(a.all_dir, "vocab.txt"), lowercase=True)
    cls_id, sep_id, pad_id = (tok.token_to_id(t) for t in ("[CLS]", "[SEP]", "[PAD]"))
    lines = [ln.strip() for ln in open(a.text, encoding="utf-8") if ln.strip()]
    ids, valid = [], np.ones((len(lines), T), dtype=np.uint8)
    for i, ln in enumerate(lines):
        body = tok.encode(ln, add_special_tokens=False).ids[:T - 2]
        n = len(body) + 2
        ids.append([cls_id] + body + [sep_id] + [pad_id] * (T - n))
        valid[i, n:] = 0
    xs = np.zeros((len(lines), T, H))
    for i, w in enumerate(ids):
        x = we[np.asarray(w)] + wp[:T] + wt[0][None, :]
        mu = x.mean(axis=1, keepdims=True)
        xc = x - mu
        var = (xc * xc).mean(axis=1, keepdims=True)
        xs[i] = xc / np.sqrt(var + eps) * g + b
    os.makedirs(a.out, exist_ok=True)
    np.ascontiguousarray(xs.astype(np.float32)).tofile(os.path.join(a.out, "inputs.f32"))
    valid.tofile(os.path.join(a.out, "mask.u8"))
    with open(os.path.join(a.out, "ids.txt"), "w") as f:
        for w in ids:
            f.write(" ".join(str(i) for i in w) + "\n")
    print("%d prompt(s) -> %s: [%d, %d, %d], real tokens %d..%d"
          % (len(lines), a.out, len(lines), T, H, int(valid.sum(axis=1).min()),
             int(valid.sum(axis=1).max())))


if __name__ == "__main__":
    main()
