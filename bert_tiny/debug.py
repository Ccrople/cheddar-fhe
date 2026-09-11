"""Compare the layer's tapped intermediates (BERT_TINY_DUMP) with the model.

    python debug.py <all_dir> <ref_dir with calib.json> <dump_dir>

Each `L<k>_<name>.f64` is [ct][live][T] in model units (the test divided by
the tap's factor); this recomputes the same quantity in float64 from the
recorded prompt and prints rms relative error, worst |err|, and a sample.
The first probe that is wrong is where the bug is.
"""

import json
import os
import sys

import numpy as np

from model import Model, gelu, bits

all_dir, ref_dir, dump = sys.argv[1], sys.argv[2], sys.argv[3]
m = Model(all_dir)
calib = json.load(open(os.path.join(ref_dir, "calib.json")))
x = m.input()
recs = []
outs = m.forward(x, records=recs)
T, H, NH, D = m.T, m.H, m.NH, m.D


def diag(mat):
    """[T, T] (query, key) -> [T diagonals][T query]: d = key - query mod T."""
    out = np.zeros((T, T))
    for d in range(T):
        for t in range(T):
            out[d, t] = mat[t, (t + d) % T]
    return out


def host(L, name):
    r, c = recs[L], calib["layers"][L]
    k = c["softmax"]["niter"]
    shift = np.asarray(c["softmax"]["shift"])            # [NH, T]
    e = c["softmax"]["exp"]
    a_e, b_e = 0.5 * (e["hi"] - e["lo"]), 0.5 * (e["hi"] + e["lo"])
    g = c["gelu"]
    a_g, b_g = 0.5 * (g["hi"] - g["lo"]), 0.5 * (g["hi"] + g["lo"])
    s0 = r["s"][0, 0]                                     # head 0, [T, T]
    u0 = (s0 - shift[0][:, None]) / 2 ** k
    y0 = np.exp(u0)
    pre1, pre2 = r["h_pre"][0], r["z_pre"][0]
    ln = {"ln1": pre1, "ln2": pre2}
    if name in ("q", "k", "v"):
        return r[name][0].reshape(T, H).T                 # [H][T]
    if name == "s":
        return diag(s0)
    if name == "t_exp":
        return ((diag(u0)[0] - b_e) / a_e)[None, :]
    if name == "y":
        return diag(y0)
    if name == "sq0":
        return (y0 * y0).sum(axis=1)[None, :]
    if name == "r0":
        return (1.0 / np.sqrt((y0 * y0).sum(axis=1)))[None, :]
    if name == "p":
        return diag(r["p"][0, 0])
    if name == "o_in":
        return r["av"][0].T
    if name == "attn":
        return r["o"][0].T
    if name == "h_pre":
        return pre1.T
    if name == "z_pre":
        return pre2.T
    for tag, pre in ln.items():
        if name == tag + "_cen":
            return (pre - pre.mean(axis=1, keepdims=True)).T
        if name == tag + "_var":
            return pre.var(axis=1)[None, :]
        if name == tag + "_r":
            return (1.0 / np.sqrt(pre.var(axis=1) + m.eps))[None, :]
        if name == tag + "_out":
            return (r["h"] if tag == "ln1" else r["z"])[0].T
    if name == "t_gelu":
        return ((r["u"][0] - b_g) / a_g).T
    if name == "g":
        return gelu(r["u"][0]).T
    if name == "y_ffn":
        return r["y"][0].T
    return None


for line in open(os.path.join(dump, "index.txt")):
    fname, n, factor = line.split()
    n = int(n)
    L = int(fname[1:fname.index("_")])
    name = fname[fname.index("_") + 1:]
    got = np.fromfile(os.path.join(dump, fname + ".f64"), dtype=np.float64)
    live = got.size // (n * T)
    got = got.reshape(n, live, T)
    want = host(L, name)
    if want is None:
        print("%-12s (no host reference)" % fname)
        continue
    want = np.asarray(want).reshape(n, T)
    err = got - want[:, None, :]
    rms = np.sqrt((err ** 2).sum() / max((want ** 2).sum() * live, 1e-300))
    worst = np.abs(err).max()
    flag = "" if rms < 1e-3 else "   <-- " + ("BAD" if rms > 0.1 else "check")
    print("%-12s n %4d live %d  rms 2^%6.2f  max|err| %10.3e  |want| <= %9.4g  "
          "sample got %s want %s%s"
          % (fname, n, live, -bits(rms), worst, np.abs(want).max(),
             np.array2string(got[0, 0, :3], precision=4),
             np.array2string(want[0, :3], precision=4), flag))
