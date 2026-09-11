"""The chain on the host with the crypto's APPROXIMATIONS in, and the knobs.

Everything the encrypted layer is told that is not a weight comes from here:
the softmax's row shift, Cho's pass count and each pass's inverse-square-root
window and degree, the exp polynomial's domain and degree, both LayerNorms'
variance windows and degrees, the GELU's interval and degree, and the maxima
the ride factors are sized from. `calib.json` is that list, per layer, in
MODEL units -- the ride bookkeeping (what is scaled by what before a
bootstrap) is the C++'s own and is not in the file.

    python sim.py <all_dir> <calib.json> [--inputs prompts/inputs.f32]
                  [--held-out other/inputs.f32] [--k K] [--tol 1e-5]
                  [--margin 1.3] [--exp-margin 1.0] [--gelu-margin 1.2]

Every polynomial is the Chebyshev INTERPOLANT on its stated domain (the
same construction as `extension/ChebyshevFit.h`, so the C++ builds the same
coefficients from (lo, hi, degree)), and the degree is the smallest of
7 / 15 / 31 / 63 / 127 / 255 whose worst error on the domain is under `tol`
(absolute for exp and GELU, relative for 1/sqrt). What the run prints is the
chain: layer L on layer L-1's approximated output, rms relative to the exact
float64 model -- the number `ci_bert_tiny_test` prints against `h_L{k}.f64`,
without a GPU -- and each operator's local error beside it.
"""

import argparse
import json
import os

import numpy as np
from numpy.polynomial import chebyshev as C

from model import Model, gelu, rms_rel, bits

DEGREES = [7, 15, 31, 63, 127, 255]


# ---------------------------------------------------------------- the fits
def cheb_interp(f, n):
    """chebfit::Interpolate: f on [-1, 1] at the n + 1 Chebyshev nodes."""
    m = n + 1
    j = np.arange(m)
    node = np.cos(np.pi * (j + 0.5) / m)
    val = f(node)
    k = np.arange(m)[:, None]
    c = (np.cos(np.pi * k * (j[None, :] + 0.5) / m) * val[None, :]).sum(axis=1)
    c *= 2.0 / m
    c[0] *= 0.5
    return c


class Poly:
    """`f` on [lo, hi] at the smallest degree under `tol` (or a fixed one)."""

    def __init__(self, f, lo, hi, tol, relative=False, degree=0, name=""):
        self.lo, self.hi, self.name = float(lo), float(hi), name
        self.a, self.b = 0.5 * (hi - lo), 0.5 * (hi + lo)
        g = lambda t: f(self.a * t + self.b)  # noqa: E731
        probe = np.linspace(-1, 1, 20001)
        want = g(probe)
        for d in ([degree] if degree else DEGREES):
            c = cheb_interp(g, d)
            got = C.chebval(probe, c)
            err = np.abs(got - want) / (np.abs(want) if relative else 1.0)
            self.err = float(err.max())
            self.degree, self.c = d, c
            if self.err <= tol:
                break
        self.levels = int(np.ceil(np.log2(self.degree + 1)))
        self.escapes = 0

    def __call__(self, x):
        t = (x - self.b) / self.a
        self.escapes += int((np.abs(t) > 1.0).sum())
        return C.chebval(t, self.c)

    def json(self):
        return {"lo": self.lo, "hi": self.hi, "degree": self.degree,
                "levels": self.levels, "fit_err": self.err}

    def __str__(self):
        return "%-6s [%9.4g, %9.4g] deg %3d (%d lv) err %.1e" % (
            self.name, self.lo, self.hi, self.degree, self.levels, self.err)


# ------------------------------------------------------------- the softmax
class ChoSoftmax:
    """y0 = exp((s - shift) / 2^k); k times: y = (y / ||y||)^2; P = y_k.

    The shift is PUBLIC, per head and query position: the population row
    maximum. The normalisation is the Euclidean norm (Cho), so each pass is
    one sum of squares over the keys, one inverse square root on it, and
    one square -- what the batched layout does with T ciphertext adds, one
    narrow polynomial and one wide multiply.
    """

    def __init__(self, k, shift, exp_poly, inv_polys):
        self.k, self.shift, self.exp, self.inv = k, shift, exp_poly, inv_polys
        self.sq_seen = [[np.inf, -np.inf] for _ in range(k)]

    def __call__(self, s, mask=None):
        u = (s - self.shift[None, :, :, None]) / float(2 ** self.k)
        y = self.exp(u)
        if mask is not None:                               # pads: exp'd, then zeroed
            y = y * np.isfinite(mask)
        for j in range(self.k):
            sq = (y * y).sum(axis=-1)
            self.sq_seen[j][0] = min(self.sq_seen[j][0], float(sq.min()))
            self.sq_seen[j][1] = max(self.sq_seen[j][1], float(sq.max()))
            r = self.inv[j](sq)
            y = (y * r[..., None]) ** 2
        return y


def exact_cho_ranges(s, shift, k, valid=None):
    """The exact Cho walk, to size the windows before any polynomial. The
    exp domain is over EVERY key (pads are exp'd in the crypto too); the
    sums are over the real keys."""
    u = (s - shift[None, :, :, None]) / float(2 ** k)
    y = np.exp(u)
    if valid is not None:
        y = y * valid[:, None, None, :]
    rng = []
    for _ in range(k):
        sq = (y * y).sum(axis=-1)
        rng.append((float(sq.min()), float(sq.max())))
        y = (y / np.sqrt(sq)[..., None]) ** 2
    return (float(u.min()), float(u.max())), rng


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("all_dir")
    ap.add_argument("calib")
    ap.add_argument("--inputs", default="")
    ap.add_argument("--held-out", default="")
    ap.add_argument("--k", type=int, default=0, help="Cho passes; 0 = auto")
    ap.add_argument("--tol", type=float, default=1e-5)
    ap.add_argument("--margin", type=float, default=1.3,
                    help="window margin on 1/sqrt domains (each side)")
    ap.add_argument("--exp-margin", type=float, default=1.0,
                    help="exp domain: lo - this, hi + this/4")
    ap.add_argument("--gelu-margin", type=float, default=1.2)
    ap.add_argument("--sq-ratio", type=float, default=300.0,
                    help="auto k: the smallest k whose first window is under this")
    a = ap.parse_args()

    def mask_beside(path):
        return os.path.join(os.path.dirname(path), "mask.u8") if path else ""

    m = Model(a.all_dir)
    x = m.prompts(a.inputs) if a.inputs else m.input()
    valid = m.valid(mask_beside(a.inputs))
    amask = Model.additive_mask(valid)
    N = x.shape[0]
    print("calibration: %d prompt(s), T %d, H %d%s" % (
        N, m.T, m.H, "" if valid is None else ", padded (real tokens %d..%d)" % (
            int(valid.sum(axis=1).min()), int(valid.sum(axis=1).max()))))
    recs = []
    exact = m.forward(x, mask=amask, records=recs)

    layers, hooks = [], []
    for L, r in enumerate(recs):
        s = r["s"]
        # the population row maximum over the REAL keys (the shift is public)
        sv = s if valid is None else np.where(valid[:, None, None, :], s, -np.inf)
        shift = sv.max(axis=(0, 3))                         # [NH, T]
        # Cho passes: the smallest k whose first window is narrow enough.
        ks = [a.k] if a.k else [1, 2, 3, 4]
        for k in ks:
            (u_lo, u_hi), rng = exact_cho_ranges(s, shift, k, valid)
            if rng[0][1] / rng[0][0] <= a.sq_ratio or k == ks[-1]:
                break
        exp_poly = Poly(np.exp, u_lo - a.exp_margin, max(u_hi, 0.0) + a.exp_margin / 4,
                        a.tol, name="exp")
        inv_polys = [Poly(lambda v: 1.0 / np.sqrt(v), lo / a.margin, hi * a.margin,
                          a.tol, relative=True, name="inv%d" % j)
                     for j, (lo, hi) in enumerate(rng)]
        sm = ChoSoftmax(k, shift, exp_poly, inv_polys)

        lns = {}
        for tag, pre in (("ln1", r["h_pre"]), ("ln2", r["z_pre"])):
            var = pre.var(axis=-1)
            lo, hi = float(var.min()), float(var.max())
            lns[tag] = Poly(lambda v: 1.0 / np.sqrt(v + m.eps), lo / a.margin,
                            hi * a.margin, a.tol, relative=True, name=tag)
        umax = float(np.abs(r["u"]).max()) * a.gelu_margin
        gl = Poly(gelu, -umax, umax, a.tol, name="gelu")

        print("layer %d: Cho k=%d, shift |.| <= %.3f, u [%.2f, %.2f]"
              % (L, k, float(np.abs(shift).max()), u_lo, u_hi))
        for p in [exp_poly] + inv_polys + [lns["ln1"], gl, lns["ln2"]]:
            print("   ", p)
        hooks.append({"softmax": sm, "ln1": lns["ln1"], "ln2": lns["ln2"], "gelu": gl})
        layers.append({
            "layer": L,
            "in_absmax": float(np.abs(x if L == 0 else exact[L - 1]).max()),
            "softmax": {"niter": k, "shift": shift.tolist(),
                        "exp": exp_poly.json(),
                        "inv": [p.json() for p in inv_polys],
                        "sq_range": rng, "s_min": float(s.min()),
                        "s_max": float(s.max())},
            "v_absmax": float(np.abs(r["v"]).max()),
            "av_absmax": float(np.abs(r["av"]).max()),
            "h_pre_absmax": float(np.abs(r["h_pre"]).max()),
            "ln1": dict(lns["ln1"].json(), r_max=float(
                1.0 / np.sqrt(lns["ln1"].lo + m.eps)),
                out_absmax=float(np.abs(r["h"]).max())),
            "gelu": gl.json(),
            "u_absmax": float(np.abs(r["u"]).max()),
            "z_pre_absmax": float(np.abs(r["z_pre"]).max()),
            "ln2": dict(lns["ln2"].json(), r_max=float(
                1.0 / np.sqrt(lns["ln2"].lo + m.eps)),
                out_absmax=float(np.abs(r["z"]).max())),
        })

    # The head: the pooler's tanh on the CERTIFIED interval (the last LN's
    # sphere), so nothing about it is a statistic of any prompt.
    head = None
    if m.head_w is not None:
        centre, radius = m.head_certified(m.NL - 1)
        lo, hi = float((centre - radius).min()), float((centre + radius).max())
        tanh_poly = Poly(np.tanh, lo, hi, a.tol, name="tanh")
        print("head: pooler input certified in [%.2f, %.2f]" % (lo, hi))
        print("   ", tanh_poly)
        head = {"tanh": tanh_poly.json(), "u_certified_lo": lo, "u_certified_hi": hi,
                "centre_absmax": float(np.abs(centre).max()),
                "radius_max": float(radius.max())}
        head_poly = tanh_poly

    # The chain with the approximations in: layer L reads L-1's approximated
    # output, so each hook is the layer's own.
    def run(xs, vmask, label):
        h = xs
        am = Model.additive_mask(vmask)
        ex = m.forward(xs, mask=am)
        print("%s: %d prompt(s)" % (label, xs.shape[0]))
        for L in range(m.NL):
            hk = hooks[L]
            hook = lambda tag, v, mask=None, hk=hk: (  # noqa: E731
                hk[tag](v, mask) if tag == "softmax" else hk[tag](v))
            h = m.layer(h, L, hook, am)
            e = rms_rel(h, ex[L])
            esc = sum(p.escapes for p in (hk["ln1"], hk["ln2"], hk["gelu"],
                                          hk["softmax"].exp) + tuple(hk["softmax"].inv))
            print("  layer %d: chain rms 2^%.2f%s" % (
                L, -bits(e), "" if esc == 0 else "  ESCAPES %d" % esc))
        if head is not None:
            lg, u = m.head(h, lambda tag, v: head_poly(v))
            lg_ex, u_ex = m.head(ex[-1])
            agree = int((lg.argmax(axis=1) == lg_ex.argmax(axis=1)).sum())
            print("  head: logits rms 2^%.2f, |u| <= %.2f (certified %.2f), labels "
                  "agree %d / %d%s" % (-bits(rms_rel(lg, lg_ex)), float(np.abs(u_ex).max()),
                                       max(abs(head["u_certified_lo"]), abs(head["u_certified_hi"])),
                                       agree, xs.shape[0],
                                       "" if head_poly.escapes == 0 else "  ESCAPES %d" % head_poly.escapes))
        return h

    run(x, valid, "calibration prompts")
    if a.held_out:
        run(m.prompts(a.held_out), m.valid(mask_beside(a.held_out)), "held-out prompts")

    out = {"model": m.meta["model"], "tokens": m.T, "channels": m.H,
           "hidden": m.I, "heads": m.NH, "head_dim": m.D, "ln_eps": m.eps,
           "calibration_prompts": N,
           "knobs": {"tol": a.tol, "margin": a.margin, "exp_margin": a.exp_margin,
                     "gelu_margin": a.gelu_margin, "sq_ratio": a.sq_ratio},
           "padded": valid is not None,
           # the shortest prompt the windows were sized on: a served prompt
           # shorter than this is outside the calibration (the first Cho
           # window is a statistic of the row sum, and an escape is batch-wide)
           "min_real_tokens": int(valid.sum(axis=1).min()) if valid is not None else m.T,
           "head": head,
           "layers": layers}
    with open(a.calib, "w") as f:
        json.dump(out, f, indent=1)
    print("calib ->", a.calib)


if __name__ == "__main__":
    main()
