"""The held-out FAILURE RATE, per prompt, on the host.

    python failure.py <all_dir> <calib.json> --inputs held/inputs.f32
                      [--mask held/mask.u8] [--slice i/W] [--out part.json]
                      [--approx] [--head]

`sim.py` WRITES a calibration; this READS one and asks a different question:
of `N` prompts the calibration never saw, how many does the shipped model
serve correctly? It rebuilds every polynomial from `calib.json` at the
recorded interval and degree -- the model that ships, not a fresh fit -- and
attributes every escape to the PROMPT it came from.

WHY AN ESCAPE IS THE CRITERION. Inside its interval a fit is good to its
`fit_err` (1e-5 .. 1e-8 here), which is far under the crypto's own noise and
cannot decide anything. Outside it a Chebyshev polynomial of degree 255-511
is astronomical. So a prompt succeeds exactly when every one of its slots
lands inside every window, and the interesting number beside the rate is the
MARGIN -- the worst `|t|` any prompt reached, where 1.0 is the edge.

    default (fast)   the exact stream, every window checked, every prompt.
                     One float64 forward pass a prompt.
    --approx         the approximated stream as well, for the per-prompt rms
                     against float64. ~30x dearer (a degree-255 Clenshaw
                     over the whole feed-forward), so it is normally run on
                     a subsample.

The default is not a weaker test as long as the margins it reports are not
near 1.0: the approximated stream differs from the exact one by ~2^-10
relative, so a slot at 0.7 of its interval cannot cross and a slot at 0.999
might. The report prints the worst margin for exactly this reason.

THE BATCH. `T * B = 65536`, so B prompts share every ciphertext and one
escaped slot is a batch-wide event: a per-prompt rate `p` is a per-BATCH
rate `1 - (1 - p)^B`. Both are printed, for every B the layout allows.

`--slice i/W` takes prompts `i::W` (W processes over one prompt set, one
JSON part each; `merge` below combines them), because a degree-255 Clenshaw
in numpy is single-threaded and the box has 96 cores.
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import numpy.polynomial.chebyshev as C

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import Model, bits, gelu                          # noqa: E402
from sim import BandedGelu, ChoSoftmax, Poly, suppression    # noqa: E402,F401


# --------------------------------------------------------------- the tracer
class Tracer:
    """Per-PROMPT escape counts, worst margins, and which window it was.

    Axis 0 of every tensor in the chain is the prompt, so a boolean of the
    same shape reduces over the remaining axes into a per-prompt count.
    """

    def __init__(self, n):
        self.n = n
        self.esc = np.zeros(n, dtype=np.int64)      # escaped slots, per prompt
        self.margin = np.zeros(n, dtype=np.float64)  # worst |t|, per prompt
        self.where = {}                              # tag -> [slots, worst |t|]
        self.base = 0                                # this chunk's first prompt

    def note(self, tag, at):
        """`at` = |t|, the argument mapped onto [-1, 1]. 1.0 is the edge."""
        axes = tuple(range(1, at.ndim))
        worst = at.max(axis=axes) if axes else at
        sl = slice(self.base, self.base + worst.shape[0])
        np.maximum(self.margin[sl], worst, out=self.margin[sl])
        row = self.where.setdefault(tag, [0, 0.0])
        row[1] = max(row[1], float(worst.max()))
        if at.max() > 1.0:
            cnt = (at > 1.0).sum(axis=axes)
            self.esc[sl] += cnt
            row[0] += int(cnt.sum())


class Checked:
    """A shipped fit: its interval, its degree, and the exact function.

    `approximate` picks which value comes back -- the polynomial (the model
    that ships) or the exact function (the escape scan, which only needs the
    ARGUMENT). Either way the argument is traced.
    """

    def __init__(self, f, lo, hi, degree, tag, tracer, approximate):
        self.f, self.tag, self.tracer = f, tag, tracer
        self.lo, self.hi = float(lo), float(hi)
        self.a, self.b = 0.5 * (hi - lo), 0.5 * (hi + lo)
        self.approximate = approximate
        self.degree = int(degree)
        if approximate:
            p = Poly(f, lo, hi, tol=0.0, degree=self.degree, name=tag)
            self.c, self.err = p.c, p.err
        else:
            self.c, self.err = None, 0.0
        self.escapes = 0                              # sim.py's interface

    def __call__(self, x):
        t = (x - self.b) / self.a
        self.tracer.note(self.tag, np.abs(t))
        return C.chebval(t, self.c) if self.approximate else self.f(x)


def rebuild(m, cal, tracer, approximate, L, dt=np.float64):
    """Layer L's four hooks, from the calibration as it ships.

    `dt` is the stream's dtype: the shift and the fold estimates are public
    TABLES that multiply it, so they have to follow it or every layer
    silently promotes back to float64.
    """
    cj = cal["layers"][L]
    mk = lambda f, j, tag: Checked(f, j["lo"], j["hi"], j["degree"],  # noqa: E731
                                   "L%02d.%s" % (L, tag), tracer, approximate)
    inv_sqrt = lambda v: 1.0 / np.sqrt(v)                            # noqa: E731
    ln_f = lambda v: 1.0 / np.sqrt(v + m.eps)                        # noqa: E731

    s = cj["softmax"]
    est = [None if not e else np.asarray(e, dtype=dt) for e in s.get("est", [])] or None
    sm = ChoSoftmax(s["niter"], np.asarray(s["shift"], dtype=dt),
                    mk(np.exp, s["exp"], "exp"),
                    [mk(inv_sqrt, q, "inv%d" % j) for j, q in enumerate(s["inv"])],
                    est, s.get("est_live_pow") or None)

    g = cj["gelu"]
    gl = BandedGelu(np.asarray(g["perm"]), int(g["tile"]),
                    [mk(gelu, q, "gelu%d" % t) for t, q in enumerate(g["tiles"])])
    return {"softmax": sm, "gelu": gl,
            "ln1": mk(ln_f, cj["ln1"], "ln1"), "ln2": mk(ln_f, cj["ln2"], "ln2")}


# ------------------------------------------------------------------ the run
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("all_dir")
    ap.add_argument("calib")
    ap.add_argument("--inputs", default="")
    ap.add_argument("--ids", default="", help="ids.u32 instead of inputs.f32")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--f32", action="store_true",
                    help="run the scan in float32 (~2x); the verdict is a "
                         "ratio of order one, so f32's 1e-7 decides nothing")
    ap.add_argument("--mask", default="")
    ap.add_argument("--slice", default="0/1", help="i/W: prompts i::W")
    ap.add_argument("--out", default="")
    ap.add_argument("--chunk", type=int, default=0, help="0 = 8192 / T")
    ap.add_argument("--approx", action="store_true",
                    help="run the approximated stream too (per-prompt rms)")
    ap.add_argument("--head", action="store_true",
                    help="carry the pooler + NSP head (needs --approx to be "
                         "more than an escape check)")
    ap.add_argument("--layers", type=int, default=0)
    a = ap.parse_args()

    m = Model(a.all_dir)
    if a.f32:
        # The scan decides a RATIO of order one (is |t| over 1?), and f32's
        # 1e-7 is nowhere near any margin; sgemm is about twice dgemm, which
        # is half of a 50,000-prompt scan. The reference chain the crypto is
        # measured against stays float64 -- that is `reference.py`'s job.
        m.w = [{k: v.astype(np.float32) for k, v in w.items()} for w in m.w]
        if m.head_w is not None:
            m.head_w = {k: v.astype(np.float32) for k, v in m.head_w.items()}
    cal = json.load(open(a.calib))
    assert cal["tokens"] == m.T, (cal["tokens"], m.T)
    NL = a.layers or min(m.NL, cal["layers_total"])

    # Either embedded inputs (small sets, and what the card is handed) or
    # token ids (big scans; the embedding is the client's half anyway).
    src = a.ids or a.inputs
    ids = m.ids(a.ids) if a.ids else None
    xs = None if a.ids else (m.prompts(a.inputs) if a.inputs else m.input())
    valid = m.valid(a.mask or (os.path.join(os.path.dirname(src), "mask.u8")
                               if src else ""))
    i, W = (int(v) for v in a.slice.split("/"))
    total = ids.shape[0] if a.ids else xs.shape[0]
    idx = np.arange(total)[i::W]
    if a.limit:
        idx = idx[:a.limit]
    valid = None if valid is None else valid[idx]   # AFTER the limit: `am`
                                                    # is indexed by position
    n = idx.shape[0]
    chunk = a.chunk or max(1, 8192 // m.T)
    print("%d of %d prompts (slice %d/%d), T %d, %d layers, %s stream, chunk %d"
          % (n, total, i, W, m.T, NL,
             "approximated" if a.approx else "exact", chunk), flush=True)

    tracer = Tracer(n)
    dt = np.float32 if a.f32 else np.float64
    hooks = [rebuild(m, cal, tracer, a.approx, L, dt) for L in range(NL)]
    am = Model.additive_mask(valid)
    if am is not None and a.f32:
        am = am.astype(np.float32)          # -inf survives the cast
    rel = np.zeros(n) if a.approx else None       # the last layer's, per prompt
    agree = np.zeros(n, dtype=bool) if a.head else None
    tanh = None
    if a.head and m.head_w is not None:
        centre, radius = m.head_certified(NL - 1)
        tanh = Checked(np.tanh, float((centre - radius).min()),
                       float((centre + radius).max()),
                       cal.get("head", {}).get("tanh", {}).get("degree", 255),
                       "tanh", tracer, a.approx)
    path = a.out or "failure_%s.json" % a.slice.replace("/", "_")

    def save(done):
        fail = tracer.esc[:done] > 0
        out = {"calib": os.path.abspath(a.calib), "tokens": m.T, "layers": NL,
               "slice": a.slice, "prompts": int(done), "approx": bool(a.approx),
               "source": os.path.abspath(src) if src else "input.f32",
               "escaped_prompts": int(fail.sum()),
               "escaped_slots": int(tracer.esc[:done].sum()),
               "margin_max": float(tracer.margin[:done].max()),
               "margin_p50": float(np.median(tracer.margin[:done])),
               "margin_p99": float(np.quantile(tracer.margin[:done], 0.99)),
               "index": idx[:done].tolist(),
               "esc_per_prompt": tracer.esc[:done].tolist(),
               # .tolist() and not a comprehension: round() on an np.float64
               # gives an np.float64 back, which json cannot serialise
               "margin_per_prompt": np.round(tracer.margin[:done], 5).tolist(),
               "where": {k: v for k, v in sorted(tracer.where.items())}}
        if rel is not None:
            out["rel_per_prompt"] = [float(v) for v in rel[:done]]
        if agree is not None:
            out["labels_agree"] = int(agree[:done].sum())
        with open(path + ".tmp", "w") as f:
            json.dump(out, f)
        os.replace(path + ".tmp", path)           # never a half-written part
        return fail

    # ONE CHUNK THROUGH THE WHOLE CHAIN: the alternative (a layer at a time
    # over every prompt) holds [50000, 512, 768] = 157 GB.
    for c0 in range(0, n, chunk):
        c1 = min(n, c0 + chunk)
        tracer.base = c0
        sel = idx[c0:c1]
        h = m.embed_ids(ids[sel]) if a.ids else xs[sel]
        if a.f32:
            h = h.astype(np.float32)
        e = h
        mc = None if am is None else am[c0:c1]
        for L in range(NL):
            hk = hooks[L]
            hook = lambda tag, v, msk=None, hk=hk: (                 # noqa: E731
                hk[tag](v, msk) if tag == "softmax" else hk[tag](v))
            hn = m.layer(h, L, hook, mc)
            e = m.layer(e, L, None, mc) if a.approx else hn
            h = hn
        if a.approx:
            ax = (1, 2)
            rel[c0:c1] = np.sqrt(((h - e) ** 2).sum(axis=ax) / (e ** 2).sum(axis=ax))
        if tanh is not None:
            lg, _ = m.head(h, lambda tag, v: tanh(v))
            lg_ex, _ = m.head(e)
            agree[c0:c1] = lg.argmax(axis=1) == lg_ex.argmax(axis=1)
        if (c1 // chunk) % 16 == 0 or c1 == n:
            fail = save(c1)
            print("  %6d / %d: %d escaped, worst margin %.3f%s"
                  % (c1, n, fail.sum(), tracer.margin[:c1].max(),
                     "" if rel is None else ", rms 2^%.2f"
                     % -bits(float(np.sqrt((rel[:c1] ** 2).mean())))), flush=True)
    fail = save(n)
    print("%d / %d prompts escaped (%.4f %%), worst margin %.3f -> %s"
          % (fail.sum(), n, 100.0 * fail.mean(), tracer.margin.max(), path))


# ---------------------------------------------------------------- the merge
def merge(parts, B_list):
    """`python failure.py --merge 'dir/part_*.json'`: the rate and what a
    batch of B prompts then does."""
    esc, marg, rel, tot, where, agree, T = [], [], [], 0, {}, 0, 0
    for p in sorted(glob.glob(parts)):
        d = json.load(open(p))
        esc += d["esc_per_prompt"]
        marg += d["margin_per_prompt"]
        rel += d.get("rel_per_prompt", [])
        agree += d.get("labels_agree", 0)
        T = d["tokens"]
        tot += d["prompts"]
        for k, v in d["where"].items():
            row = where.setdefault(k, [0, 0.0])
            row[0] += v[0]
            row[1] = max(row[1], v[1])
    esc, marg = np.asarray(esc), np.asarray(marg)
    fail = esc > 0
    p = float(fail.mean())
    print("prompts %d, T %d" % (tot, T))
    print("  escaped prompts   %d  (%.4f %%)" % (fail.sum(), 100 * p))
    print("  escaped slots     %d" % esc.sum())
    print("  margin  p50 %.3f  p99 %.3f  max %.3f" % (
        np.median(marg), np.quantile(marg, 0.99), marg.max()))
    if rel:
        r = np.asarray(rel)
        print("  rms     p50 2^%.2f  worst 2^%.2f" % (
            -bits(float(np.median(r))), -bits(float(r.max()))))
    if agree:
        print("  labels agree      %d / %d" % (agree, tot))
    for k, v in sorted(where.items(), key=lambda kv: -kv[1][1])[:8]:
        print("  %-14s worst |t| %.3f%s" % (k, v[1],
                                            "" if not v[0] else "  ESCAPED %d slots" % v[0]))
    for B in B_list:
        if B * T == 65536 or B == 1:
            print("  B = %-4d (T*B = %d): a batch fails with probability "
                  "%.4f %%" % (B, B * T, 100 * (1 - (1 - p) ** B)))


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--merge":
        merge(sys.argv[2], [1, 65536 // int(sys.argv[3])] if len(sys.argv) > 3
              else [1, 128, 256, 512])
    else:
        main()
