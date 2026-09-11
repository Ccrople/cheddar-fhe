"""The exact float64 references the encrypted run is measured against.

    python reference.py <all_dir> <out_dir> [--inputs prompts/inputs.f32]

Writes `h_L{k}.f64` ([N, T, H]; N = 1 for the recorded prompt) and
`stats.json`, the RAW per-layer ranges (no margins, no degrees -- those are
`sim.py`'s decisions and live in calib.json).
"""

import argparse
import json
import os

import numpy as np

from model import Model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("all_dir")
    ap.add_argument("out")
    ap.add_argument("--inputs", default="")
    a = ap.parse_args()
    m = Model(a.all_dir)
    x = m.prompts(a.inputs) if a.inputs else m.input()
    os.makedirs(a.out, exist_ok=True)
    N = x.shape[0]
    print("%d prompt(s) x [%d, %d]; input |x| <= %.4f" % (N, m.T, m.H,
                                                          float(np.abs(x).max())))
    recs = []
    outs = m.forward(x, records=recs)
    stats = []
    for L, (z, r) in enumerate(zip(outs, recs)):
        z.astype(np.float64).tofile(os.path.join(a.out, "h_L%02d.f64" % L))
        s = r["s"]
        st = {"layer": L,
              "q_absmax": float(np.abs(r["q"]).max()),
              "k_absmax": float(np.abs(r["k"]).max()),
              "v_absmax": float(np.abs(r["v"]).max()),
              "s_min": float(s.min()), "s_max": float(s.max()),
              "row_max_absmax": float(np.abs(s.max(axis=-1)).max()),
              "p_max": float(r["p"].max()),
              "av_absmax": float(np.abs(r["av"]).max()),
              "o_absmax": float(np.abs(r["o"]).max()),
              "h_pre_absmax": float(np.abs(r["h_pre"]).max()),
              "h_absmax": float(np.abs(r["h"]).max()),
              "u_absmax": float(np.abs(r["u"]).max()),
              "y_absmax": float(np.abs(r["y"]).max()),
              "z_pre_absmax": float(np.abs(r["z_pre"]).max()),
              "z_absmax": float(np.abs(z).max())}
        for tag, pre in (("ln1", r["h_pre"]), ("ln2", r["z_pre"])):
            var = pre.var(axis=-1)
            st[tag + "_var_min"] = float(var.min())
            st[tag + "_var_max"] = float(var.max())
        stats.append(st)
        print("layer %d: s [%.2f, %.2f]  |h| %.2f  |u| %.2f  |z| %.2f  "
              "ln1 var %.3g..%.3g  ln2 var %.3g..%.3g"
              % (L, st["s_min"], st["s_max"], st["h_absmax"], st["u_absmax"],
                 st["z_absmax"], st["ln1_var_min"], st["ln1_var_max"],
                 st["ln2_var_min"], st["ln2_var_max"]))
    with open(os.path.join(a.out, "stats.json"), "w") as f:
        json.dump({"prompts": N, "tokens": m.T, "channels": m.H,
                   "layers": stats}, f, indent=1)
    print("done ->", a.out)


if __name__ == "__main__":
    main()
