"""The exact float64 references the encrypted run is measured against.

    python reference.py <all_dir> <out_dir> [--inputs prompts/inputs.f32]
                        [--mask mask.u8] [--chunk 64] [--prompts N]

Writes `h_L{k}.f64` ([N, T, H]; N = 1 for the recorded prompt) and
`stats.json`, the RAW per-layer ranges (no margins, no degrees -- those are
`sim.py`'s decisions and live in calib.json).

This is `bert_tiny/reference.py` made CHUNKED over prompts, which BERT-Base's
width forces: one layer's intermediates for 1000 prompts of 128 tokens are
`u` at [1000, 128, 3072] = 3.1 GB alone, and the twelve layers' records the
tiny version keeps would be 150 GB. Nothing about the numbers changes -- a
prompt's forward does not see the others -- only how many exist at once.
"""

import argparse
import json
import os

import numpy as np

from model import Model


class Stats:
    """The running per-layer ranges, accumulated a chunk at a time."""

    def __init__(self, layer):
        self.d = {"layer": layer}

    def _mx(self, key, value):
        self.d[key] = max(self.d.get(key, 0.0), float(value))

    def _lo(self, key, value):
        self.d[key] = min(self.d.get(key, np.inf), float(value))

    def add(self, r, z):
        s = r["s"]
        for key, v in (("q_absmax", r["q"]), ("k_absmax", r["k"]),
                       ("v_absmax", r["v"]), ("av_absmax", r["av"]),
                       ("o_absmax", r["o"]), ("h_pre_absmax", r["h_pre"]),
                       ("h_absmax", r["h"]), ("u_absmax", r["u"]),
                       ("y_absmax", r["y"]), ("z_pre_absmax", r["z_pre"]),
                       ("z_absmax", z)):
            self._mx(key, np.abs(v).max())
        self._mx("s_max", s.max())
        self._lo("s_min", s.min())
        self._mx("row_max_absmax", np.abs(s.max(axis=-1)).max())
        self._mx("p_max", r["p"].max())
        for tag, pre in (("ln1", r["h_pre"]), ("ln2", r["z_pre"])):
            var = pre.var(axis=-1)
            self._lo(tag + "_var_min", var.min())
            self._mx(tag + "_var_max", var.max())

    def json(self):
        return self.d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("all_dir")
    ap.add_argument("out")
    ap.add_argument("--inputs", default="")
    ap.add_argument("--mask", default="", help="mask.u8 beside --inputs (default)")
    ap.add_argument("--chunk", type=int, default=64, help="prompts at a time")
    ap.add_argument("--prompts", type=int, default=0, help="0 = all of --inputs")
    a = ap.parse_args()
    m = Model(a.all_dir)
    x = m.prompts(a.inputs) if a.inputs else m.input()
    mask_path = a.mask or (os.path.join(os.path.dirname(a.inputs), "mask.u8")
                           if a.inputs else "")
    valid = m.valid(mask_path)
    if a.prompts > 0:
        x = x[:a.prompts]
        if valid is not None:
            valid = valid[:a.prompts]
    os.makedirs(a.out, exist_ok=True)
    N = x.shape[0]
    am = Model.additive_mask(valid)
    print("%d prompt(s) x [%d, %d]; input |x| <= %.4f%s" % (
        N, m.T, m.H, float(np.abs(x).max()),
        "" if valid is None else "; padded (real tokens %d..%d)" % (
            int(valid.sum(axis=1).min()), int(valid.sum(axis=1).max()))))
    chunk = max(1, min(a.chunk, N))
    cur = x
    stats = []
    for L in range(m.NL):
        out = np.empty_like(cur)
        st = Stats(L)
        for c0 in range(0, N, chunk):
            c1 = min(N, c0 + chunk)
            r = {}
            out[c0:c1] = m.layer(cur[c0:c1], L, mask=None if am is None else am[c0:c1],
                                 record=r)
            st.add(r, out[c0:c1])
        out.astype(np.float64).tofile(os.path.join(a.out, "h_L%02d.f64" % L))
        d = st.json()
        stats.append(d)
        print("layer %d: s [%.2f, %.2f]  |h| %.2f  |u| %.2f  |z| %.2f  "
              "ln1 var %.3g..%.3g  ln2 var %.3g..%.3g"
              % (L, d["s_min"], d["s_max"], d["h_absmax"], d["u_absmax"],
                 d["z_absmax"], d["ln1_var_min"], d["ln1_var_max"],
                 d["ln2_var_min"], d["ln2_var_max"]))
        cur = out
    if m.head_w is not None:
        logits = np.empty((N, m.head_w["cw"].shape[1]))
        umax = 0.0
        for c0 in range(0, N, chunk):
            c1 = min(N, c0 + chunk)
            logits[c0:c1], u = m.head(cur[c0:c1])
            umax = max(umax, float(np.abs(u).max()))
        logits.astype(np.float64).tofile(os.path.join(a.out, "cls_logits.f64"))
        lab = logits.argmax(axis=1)
        print("head: logits [N, %d] written; labels 0/1 = %d/%d; |u| <= %.3f"
              % (logits.shape[1], int((lab == 0).sum()), int((lab == 1).sum()), umax))
    with open(os.path.join(a.out, "stats.json"), "w") as f:
        json.dump({"prompts": N, "tokens": m.T, "channels": m.H,
                   "layers": stats}, f, indent=1)
    print("done ->", a.out)


if __name__ == "__main__":
    main()
