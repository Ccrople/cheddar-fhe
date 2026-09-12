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
def live_of(mask, T, N):
    """The instance's real-token count -- PUBLIC (the server holds the mask)."""
    if mask is None:
        return np.full(N, float(T))
    return np.isfinite(mask).sum(axis=1).astype(float)


class ChoSoftmax:
    """y0 = exp((s - shift) / 2^k); k times: y = (y / ||y||)^2; P = y_k.

    The shift is PUBLIC, per head and query position: the population row
    maximum. The normalisation is the Euclidean norm (Cho), so each pass is
    one sum of squares over the keys, one inverse square root on it, and
    one square -- what the batched layout does with T ciphertext adds, one
    narrow polynomial and one wide multiply.

    THE FOLD. `1/sqrt(sq) = (1/sqrt(est)) (1/sqrt(sq/est))` for any public
    `est > 0`, so the polynomial can be given the RATIO instead of `sq`. With
    `est[b][h][t] = ehat[h][t] * live_b^p` the two things that make the first
    window wide leave it: the instance's real-token count (p = 1 at pass 0,
    where `sq_0` is a sum over exactly `live_b` keys) and the row's own place
    in the population (`ehat`). What is left is one row's spread.
    """

    def __init__(self, k, shift, exp_poly, inv_polys, est=None, live_pow=None):
        self.k, self.shift, self.exp, self.inv = k, shift, exp_poly, inv_polys
        self.est = est
        self.live_pow = live_pow if live_pow is not None else fold_live_pow(k)
        self.sq_seen = [[np.inf, -np.inf] for _ in range(k)]

    def __call__(self, s, mask=None):
        u = (s - self.shift[None, :, :, None]) / float(2 ** self.k)
        y = self.exp(u)
        if mask is not None:                               # pads: exp'd, then zeroed
            y = y * np.isfinite(mask)
        live = live_of(mask, s.shape[-1], s.shape[0])
        for j in range(self.k):
            sq = (y * y).sum(axis=-1)
            self.sq_seen[j][0] = min(self.sq_seen[j][0], float(sq.min()))
            self.sq_seen[j][1] = max(self.sq_seen[j][1], float(sq.max()))
            if self.est is None or self.est[j] is None:
                r = self.inv[j](sq)
            else:
                e = self.est[j][None] * (live[:, None, None] ** self.live_pow[j])
                r = self.inv[j](sq / e) / np.sqrt(e)
            y = (y * r[..., None]) ** 2
        return y


def fold_live_pow(k):
    """Pass 0's sum runs over `live` keys; the later passes' argument is a
    normalised distribution, whose size the count no longer sets."""
    return [1] + [0] * (k - 1)


def fold_passes(which, k):
    """Which passes carry a fold. The LAST pass pays a narrow level for its
    `1/sqrt(est)` that the walk would otherwise spend on `P` (the plan needs
    `l_p >= 4`), and the later windows are near-theorems anyway -- the wide
    one is the first. `first` is therefore the default."""
    if which == "none":
        return [False] * k
    if which == "all":
        return [True] * k
    return [j == 0 for j in range(k)]


def exact_cho_ranges(s, shift, k, valid=None, fold=True):
    """The exact Cho walk, to size the windows before any polynomial. The
    exp domain is over EVERY key (pads are exp'd in the crypto too); the
    sums are over the real keys.

    Returns the exp domain, each pass's raw `sq` range, the fold estimates
    `[pass][head][token]` and the ranges the polynomials actually see. The
    estimate `sqrt(min * max)` over the population of that row is the one
    that MINIMISES the ratio window -- it leaves `[sqrt(min/max),
    sqrt(max/min)]`, so the window's ratio becomes the worst ROW's spread
    instead of the whole population's.
    """
    u = (s - shift[None, :, :, None]) / float(2 ** k)
    y = np.exp(u)
    if valid is not None:
        y = y * valid[:, None, None, :]
    N, T = s.shape[0], s.shape[-1]
    live = valid.sum(axis=1).astype(float) if valid is not None else np.full(N, float(T))
    pw = fold_live_pow(k)
    folds = fold if isinstance(fold, (list, tuple)) else [bool(fold)] * k
    rng, est_tabs, seen = [], [], []
    for j in range(k):
        sq = (y * y).sum(axis=-1)                        # [N, NH, T]
        rng.append((float(sq.min()), float(sq.max())))
        if folds[j]:
            w = live[:, None, None] ** pw[j]
            z = sq / w
            zlo, zhi = z.min(axis=0), z.max(axis=0)       # [NH, T]
            # the geometric mean IN LOGS: the product underflows outright
            # once the walk has squared the row a few times. A row whose
            # smallest member has underflowed to zero has no estimate worth
            # the name -- the walk there is degenerate, not the fold -- so it
            # takes the largest, and the window it opens is reported.
            zlo = np.where(zlo > 0, zlo, zhi)
            est = np.exp(0.5 * (np.log(zlo) + np.log(zhi)))
            rho = z / est[None]
            seen.append((float(rho.min()), float(rho.max())))
            est_tabs.append(est)
        else:
            seen.append(rng[-1])
            est_tabs.append(None)
        y = (y / np.sqrt(sq)[..., None]) ** 2
    return (float(u.min()), float(u.max())), rng, est_tabs, seen


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
    ap.add_argument("--no-chain", action="store_true",
                    help="write the calibration without running the host "
                         "chain (the windows need only the exact forward)")
    ap.add_argument("--fold-passes", default="first",
                    choices=["first", "all", "none"],
                    help="which Cho passes divide by the public row estimate")
    ap.add_argument("--no-fold", action="store_true",
                    help="do not divide the Cho sums by a public row estimate")
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
        # Cho passes: the smallest k whose first window is narrow enough --
        # of what the POLYNOMIAL sees, which the fold is what decides.
        ks = [a.k] if a.k else [1, 2, 3, 4]
        for k in ks:
            (u_lo, u_hi), rng, est_tabs, seen = exact_cho_ranges(
                s, shift, k, valid,
                fold=fold_passes("none" if a.no_fold else a.fold_passes, k))
            if seen[0][1] / seen[0][0] <= a.sq_ratio or k == ks[-1]:
                break
        exp_poly = Poly(np.exp, u_lo - a.exp_margin, max(u_hi, 0.0) + a.exp_margin / 4,
                        a.tol, name="exp")
        inv_polys = [Poly(lambda v: 1.0 / np.sqrt(v), lo / a.margin, hi * a.margin,
                          a.tol, relative=True, name="inv%d" % j)
                     for j, (lo, hi) in enumerate(seen)]
        est = None if a.no_fold else est_tabs
        sm = ChoSoftmax(k, shift, exp_poly, inv_polys, est)
        folded = [e is not None for e in est_tabs] if est else [False] * k

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
        for j in range(k):
            print("     window %d: raw %.3g x  ->  seen %.3g x%s" % (
                j, rng[j][1] / rng[j][0], seen[j][1] / seen[j][0],
                "" if est is None or est_tabs[j] is None
                else "  (fold, live^%d)" % fold_live_pow(k)[j]))
        for p in [exp_poly] + inv_polys + [lns["ln1"], gl, lns["ln2"]]:
            print("   ", p)
        hooks.append({"softmax": sm, "ln1": lns["ln1"], "ln2": lns["ln2"], "gelu": gl})
        layers.append({
            "layer": L,
            "in_absmax": float(np.abs(x if L == 0 else exact[L - 1]).max()),
            "softmax": {"niter": k, "shift": shift.tolist(),
                        "exp": exp_poly.json(),
                        "inv": [p.json() for p in inv_polys],
                        # the fold: est[pass][head][token], multiplied at
                        # serve time by the instance's live count to the
                        # power est_live_pow[pass] (both public)
                        "est": [] if est is None else
                               [[] if e is None else e.tolist() for e in est],
                        "est_live_pow": [] if est is None else fold_live_pow(k),
                        "sq_range": rng, "sq_seen": seen,
                        "s_min": float(s.min()),
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

    if not a.no_chain:
        run(x, valid, "calibration prompts")
        if a.held_out:
            run(m.prompts(a.held_out), m.valid(mask_beside(a.held_out)),
                "held-out prompts")

    out = {"model": m.meta["model"], "tokens": m.T, "channels": m.H,
           "hidden": m.I, "heads": m.NH, "head_dim": m.D, "ln_eps": m.eps,
           "calibration_prompts": N,
           "knobs": {"tol": a.tol, "margin": a.margin, "exp_margin": a.exp_margin,
                     "gelu_margin": a.gelu_margin, "sq_ratio": a.sq_ratio,
                     "fold": "none" if a.no_fold else a.fold_passes},
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
