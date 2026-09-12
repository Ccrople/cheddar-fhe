"""BERT-Base (google-bert/bert-base-uncased) -> f32 blobs in this tree's layout.

The shape is READ FROM THE CHECKPOINT's config.json, so the same file exports
any BERT of this family (tiny / mini / small / base): nothing here is 128 or
2 by hand.

    python export.py <out_dir> [--tokens 128] [--model google-bert/bert-base-uncased]
                     [--prompts N --corpus text.txt]

Writes
    meta.json          the shape (layers, channels, hidden, heads, head_dim,
                       tokens, ln_eps, vocab, max_position)
    emb_norm.f32, emb_norm_bias.f32
    L00/ .. L{n}/      wq wk wv wo (H x H), wint (H x I), wout (I x H) --
                       ORIENTATION [in, out] (HF's [out, in] transposed) --
                       and bq bk bv bo bint bout attn_norm attn_norm_bias
                       ffn_norm ffn_norm_bias
    head/              pooler (H x H + H), the NSP classifier (H x 2 + 2)
    input.f32          [T, H]: the recorded prompt after the embedding
                       LayerNorm (the encoder's real input, computed in the
                       clear as a client would); meta.json records the ids
    prompts/           with --prompts: inputs.f32 [N, T, H] and ids.txt, N
                       disjoint (T - 2)-piece windows of --corpus, each with
                       its own [CLS]/[SEP]; the population a calibration
                       reads

This is `bert_tiny/export.py` with the default checkpoint changed: the
shape comes from config.json either way, so a reader of one reads the other.
"""

import argparse
import json
import os

import numpy as np

TEXT = (
    "the quick brown fox jumps over the lazy dog while the sun rises slowly "
    "above the quiet river and the city begins to wake. a small boat drifts "
    "past the old stone bridge, and two children run along the bank shouting "
    "at the birds. later the market opens and the streets fill with people "
    "carrying baskets of bread, fruit and flowers, and the noise of the day "
    "settles into a steady rhythm that lasts until evening falls again over "
    "the water and the lamps are lit one by one along the road home. the "
    "baker closes his shutters, a train sounds far away beyond the hills, "
    "and somewhere a radio plays an old song that nobody listens to closely. "
    "by midnight the square is empty except for a cat asleep under a bench, "
    "and the first cold wind of autumn moves the leaves across the stones. "
    "in the morning the fog lifts from the fields and the farmers lead their "
    "horses down to the river where the water runs clear and cold over the "
    "smooth grey rocks, and the church bell rings nine times across the "
    "valley while the school children walk in pairs along the narrow road "
    "with their books held tight against the wind that comes from the sea. "
    "the old teacher waits at the gate with a lantern still burning from the "
    "night before, counting the heads as they pass, and when the last one "
    "is inside she closes the door and the day begins in earnest with the "
    "sound of chalk on slate and rain beginning to fall on the tin roof."
)


def snapshot(model):
    from huggingface_hub import snapshot_download

    return snapshot_download(model, allow_patterns=["*.json", "vocab.txt",
                                                    "*.safetensors"],
                             max_workers=1)


_HEADERS = {}


def _header(path):
    if path not in _HEADERS:
        with open(path, "rb") as f:
            n = int.from_bytes(f.read(8), "little")
            _HEADERS[path] = (n, json.loads(f.read(n)))
    return _HEADERS[path]


def build_index(path):
    table = {}
    for fn in sorted(os.listdir(path)):
        if fn.endswith(".safetensors"):
            full = os.path.join(path, fn)
            _, header = _header(full)
            for k in header:
                if k != "__metadata__":
                    table[k] = full
    return table


def get(table, name):
    """One tensor as f32; both the `bert.` prefix and the gamma/beta spelling."""
    cands = [name, "bert." + name]
    cands += [c.replace("LayerNorm.weight", "LayerNorm.gamma")
               .replace("LayerNorm.bias", "LayerNorm.beta") for c in cands]
    for c in cands:
        if c in table:
            name = c
            break
    else:
        raise KeyError(name)
    path = table[name]
    n, header = _header(path)
    info = header[name]
    start, end = info["data_offsets"]
    with open(path, "rb") as f:
        f.seek(8 + n + start)
        raw = f.read(end - start)
    dtype, shape = info["dtype"], tuple(info["shape"])
    if dtype == "F32":
        return np.frombuffer(raw, dtype=np.float32).reshape(shape)
    if dtype == "BF16":
        wide = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
        return wide.view(np.float32).reshape(shape)
    if dtype == "F16":
        return np.frombuffer(raw, dtype=np.float16).reshape(shape).astype(np.float32)
    raise TypeError(dtype)


def tokenizer(path):
    from tokenizers import BertWordPieceTokenizer

    return BertWordPieceTokenizer(os.path.join(path, "vocab.txt"), lowercase=True)


def windows(tok, text, T, n):
    """`n` disjoint id windows of exactly T tokens: [CLS] + (T-2) pieces + [SEP]."""
    ids = tok.encode(text, add_special_tokens=False).ids
    cls_id, sep_id = tok.token_to_id("[CLS]"), tok.token_to_id("[SEP]")
    body = T - 2
    assert len(ids) >= body * n, "the text has %d pieces, %d windows of %d need %d" % (
        len(ids), n, body, body * n)
    return [[cls_id] + ids[i * body:(i + 1) * body] + [sep_id] for i in range(n)]


def embed(table, ids, T, eps):
    we = get(table, "embeddings.word_embeddings.weight").astype(np.float64)
    wp = get(table, "embeddings.position_embeddings.weight").astype(np.float64)
    wt = get(table, "embeddings.token_type_embeddings.weight").astype(np.float64)
    g = get(table, "embeddings.LayerNorm.weight").astype(np.float64)
    b = get(table, "embeddings.LayerNorm.bias").astype(np.float64)
    x = we[np.asarray(ids)] + wp[:T] + wt[0][None, :]
    mu = x.mean(axis=1, keepdims=True)
    xc = x - mu
    var = (xc * xc).mean(axis=1, keepdims=True)
    return xc / np.sqrt(var + eps) * g + b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--model", default="google-bert/bert-base-uncased")
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--prompts", type=int, default=0)
    ap.add_argument("--corpus", default="")
    ap.add_argument("--min-len", type=int, default=0,
                    help="with --prompts: real tokens per prompt uniform in "
                         "[min-len, T] (padded with [PAD]); 0 = all exactly T")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    T = a.tokens

    path = snapshot(a.model)
    cfg = json.load(open(os.path.join(path, "config.json")))
    assert cfg["hidden_act"] == "gelu", cfg["hidden_act"]
    H, I = cfg["hidden_size"], cfg["intermediate_size"]
    NL, NH = cfg["num_hidden_layers"], cfg["num_attention_heads"]
    eps = cfg["layer_norm_eps"]
    assert T <= cfg["max_position_embeddings"]
    table = build_index(path)
    os.makedirs(a.out, exist_ok=True)

    def dump(name, arr):
        np.ascontiguousarray(arr.astype(np.float32)).tofile(os.path.join(a.out, name))

    mats = [("wq.f32", "attention.self.query.weight", (H, H)),
            ("wk.f32", "attention.self.key.weight", (H, H)),
            ("wv.f32", "attention.self.value.weight", (H, H)),
            ("wo.f32", "attention.output.dense.weight", (H, H)),
            ("wint.f32", "intermediate.dense.weight", (H, I)),
            ("wout.f32", "output.dense.weight", (I, H))]
    vecs = [("bq.f32", "attention.self.query.bias", (H,)),
            ("bk.f32", "attention.self.key.bias", (H,)),
            ("bv.f32", "attention.self.value.bias", (H,)),
            ("bo.f32", "attention.output.dense.bias", (H,)),
            ("bint.f32", "intermediate.dense.bias", (I,)),
            ("bout.f32", "output.dense.bias", (H,)),
            ("attn_norm.f32", "attention.output.LayerNorm.weight", (H,)),
            ("attn_norm_bias.f32", "attention.output.LayerNorm.bias", (H,)),
            ("ffn_norm.f32", "output.LayerNorm.weight", (H,)),
            ("ffn_norm_bias.f32", "output.LayerNorm.bias", (H,))]
    for L in range(NL):
        stem = "encoder.layer.%d." % L
        os.makedirs(os.path.join(a.out, "L%02d" % L), exist_ok=True)
        for fname, key, shape in mats:
            w = get(table, stem + key).T
            assert w.shape == shape, (fname, w.shape)
            dump("L%02d/%s" % (L, fname), w)
        for fname, key, shape in vecs:
            v = get(table, stem + key)
            assert v.shape == shape, (fname, v.shape)
            dump("L%02d/%s" % (L, fname), v)
    dump("emb_norm.f32", get(table, "embeddings.LayerNorm.weight"))
    dump("emb_norm_bias.f32", get(table, "embeddings.LayerNorm.bias"))
    # The embedding tables and the vocabulary, so `encode.py` can turn text
    # into the encoder's input without the checkpoint.
    dump("emb_word.f32", get(table, "embeddings.word_embeddings.weight"))
    dump("emb_pos.f32", get(table, "embeddings.position_embeddings.weight"))
    dump("emb_type.f32", get(table, "embeddings.token_type_embeddings.weight"))
    import shutil
    shutil.copyfile(os.path.join(path, "vocab.txt"), os.path.join(a.out, "vocab.txt"))
    os.makedirs(os.path.join(a.out, "head"), exist_ok=True)
    dump("head/pool_w.f32", get(table, "pooler.dense.weight").T)
    dump("head/pool_b.f32", get(table, "pooler.dense.bias"))
    dump("head/cls_w.f32", get(table, "cls.seq_relationship.weight").T)
    dump("head/cls_b.f32", get(table, "cls.seq_relationship.bias"))

    tok = tokenizer(path)
    # The recorded prompt: TEXT where it is long enough (T <= 256), else the
    # corpus's first window, so a T = 512 export has a real prompt too.
    if len(tok.encode(TEXT, add_special_tokens=False).ids) >= T - 2:
        ids = windows(tok, TEXT, T, 1)[0]
    else:
        assert a.corpus, "TEXT is short of T = %d pieces; give --corpus" % T
        text = open(a.corpus, encoding="utf-8", errors="ignore").read()
        ids = windows(tok, text, T, 1)[0]
    x0 = embed(table, ids, T, eps)
    dump("input.f32", x0)
    meta = {"model": a.model, "layers": NL, "channels": H, "hidden": I,
            "heads": NH, "head_dim": H // NH, "tokens": T, "ln_eps": eps,
            "vocab": cfg["vocab_size"],
            "max_position": cfg["max_position_embeddings"],
            "prompt_ids": ids, "input_absmax": float(np.abs(x0).max())}
    with open(os.path.join(a.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print("%s: %d layers, H %d, I %d, %d heads x %d; input |x| <= %.4f"
          % (a.model, NL, H, I, NH, H // NH, meta["input_absmax"]))

    if a.prompts > 0:
        text = open(a.corpus, encoding="utf-8", errors="ignore").read()
        ws = windows(tok, text, T, a.prompts)
        valid = np.ones((a.prompts, T), dtype=np.uint8)
        if a.min_len > 0:
            # Random real lengths: [CLS] body [SEP] then [PAD] to T. The mask
            # is what the attention multiplies its exp by; padded positions
            # are still computed (as BERT does) and never read.
            rng = np.random.default_rng(a.seed)
            pad_id, sep_id = tok.token_to_id("[PAD]"), tok.token_to_id("[SEP]")
            for i, w in enumerate(ws):
                n = int(rng.integers(a.min_len, T + 1))
                ws[i] = w[:n - 1] + [sep_id] + [pad_id] * (T - n)
                valid[i, n:] = 0
        xs = np.stack([embed(table, w, T, eps) for w in ws])
        pd = os.path.join(a.out, "prompts")
        os.makedirs(pd, exist_ok=True)
        np.ascontiguousarray(xs.astype(np.float32)).tofile(os.path.join(pd, "inputs.f32"))
        valid.tofile(os.path.join(pd, "mask.u8"))
        with open(os.path.join(pd, "ids.txt"), "w") as f:
            for w in ws:
                f.write(" ".join(str(i) for i in w) + "\n")
        print("prompts/: %d x [%d, %d], |x| <= %.4f, real tokens %d..%d"
              % (a.prompts, T, H, float(np.abs(xs).max()),
                 int(valid.sum(axis=1).min()), int(valid.sum(axis=1).max())))
    print("done ->", a.out)


if __name__ == "__main__":
    main()
