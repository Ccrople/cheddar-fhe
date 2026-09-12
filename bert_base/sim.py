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
                  [--chunk 64] [--no-chain]

Every polynomial is the Chebyshev INTERPOLANT on its stated domain (the
same construction as `extension/ChebyshevFit.h`, so the C++ builds the same
coefficients from (lo, hi, degree)), and the degree is the smallest of
7 / 15 / 31 / 63 / 127 / 255 / 511 whose worst error on the domain is under
`tol` (absolute for exp and GELU, relative for 1/sqrt). What the run prints
is the chain: layer L on layer L-1's approximated output, rms relative to
the exact float64 model -- the number `ci_bert_base_test` prints against
`h_L{k}.f64`, without a GPU -- and each operator's local error beside it.

This is `bert_tiny/sim.py` made CHUNKED over prompts (BERT-Base's `u` is
[N, T, 3072]) and layer by layer: only the scores of the layer being
calibrated are held whole, because the row shift is a maximum over the
population and the windows are then read off it. The decisions are
identical -- a prompt's forward does not see the others.
"""

import argparse
import json
import os

import numpy as np
from numpy.polynomial import chebyshev as C

from model import Model, gelu, rms_rel, bits

DEGREES = [7, 15, 31, 63, 127, 255, 511]


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

    def __init__(self, f, lo, hi, tol, relative=False, degree=0, name="",
                 max_degree=0):
        self.lo, self.hi, self.name = float(lo), float(hi), name
        self.a, self.b = 0.5 * (hi - lo), 0.5 * (hi + lo)
        g = lambda t: f(self.a * t + self.b)  # noqa: E731
        probe = np.linspace(-1, 1, 20001)
        want = g(probe)
        pool = [degree] if degree else [d for d in DEGREES
                                        if not max_degree or d <= max_degree]
        for d in pool:
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
    """The instance's real-token count, `[N]` -- PUBLIC (the server holds the
    mask).

    The mask reaches the hook in the model's ADDITIVE form, `[N, 1, 1, T]`
    (0 on a real key, -inf on a pad), so the count is over the LAST axis and
    the singleton head and query axes are dropped: summing over axis 1 gives
    a `[N, 1, T]` of ones, which broadcasts into a five-dimensional `y` two
    lines later and is what `bert_tiny/sim.py` did (it never ran a chain
    with both a fold and a mask -- every folded calibration there was
    `--no-chain`).
    """
    if mask is None:
        return np.full(N, float(T))
    return np.isfinite(mask).sum(axis=-1).reshape(-1).astype(float)


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

    def __call__(self, s, mask=None):
        u = (s - self.shift[None, :, :, None]) / float(2 ** self.k)
        y = self.exp(u)
        if mask is not None:                               # pads: exp'd, then zeroed
            y = y * np.isfinite(mask)
        live = live_of(mask, s.shape[-1], s.shape[0])
        for j in range(self.k):
            sq = (y * y).sum(axis=-1)
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


def exact_cho_ranges(s, shift, k, valid=None, fold=True, chunk=64):
    """The exact Cho walk, to size the windows before any polynomial. The
    exp domain is over EVERY key (pads are exp'd in the crypto too); the
    sums are over the real keys.

    Returns the exp domain, each pass's raw `sq` range, the fold estimates
    `[pass][head][token]` and the ranges the polynomials actually see. The
    estimate `sqrt(min * max)` over the population of that row is the one
    that MINIMISES the ratio window -- it leaves `[sqrt(min/max),
    sqrt(max/min)]`, so the window's ratio becomes the worst ROW's spread
    instead of the whole population's. That identity is also why this can
    run a chunk of prompts at a time: what the polynomial sees is decided by
    each ROW's extremes, and a minimum is associative.
    """
    N, NH, T = s.shape[0], s.shape[1], s.shape[-1]
    pw = fold_live_pow(k)
    folds = fold if isinstance(fold, (list, tuple)) else [bool(fold)] * k
    u_lo, u_hi = np.inf, -np.inf
    rlo, rhi = [np.inf] * k, [-np.inf] * k
    zlo = [np.full((NH, T), np.inf) for _ in range(k)]
    zhi = [np.full((NH, T), -np.inf) for _ in range(k)]
    for c0 in range(0, N, chunk):
        c1 = min(N, c0 + chunk)
        u = (s[c0:c1] - shift[None, :, :, None]) / float(2 ** k)
        u_lo, u_hi = min(u_lo, float(u.min())), max(u_hi, float(u.max()))
        y = np.exp(u)
        if valid is not None:
            y = y * valid[c0:c1][:, None, None, :]
            live = valid[c0:c1].sum(axis=1).astype(float)
        else:
            live = np.full(c1 - c0, float(T))
        for j in range(k):
            sq = (y * y).sum(axis=-1)                      # [n, NH, T]
            rlo[j], rhi[j] = min(rlo[j], float(sq.min())), max(rhi[j], float(sq.max()))
            if folds[j]:
                z = sq / (live[:, None, None] ** pw[j])
                np.minimum(zlo[j], z.min(axis=0), out=zlo[j])
                np.maximum(zhi[j], z.max(axis=0), out=zhi[j])
            y = (y / np.sqrt(sq)[..., None]) ** 2
    rng = [(rlo[j], rhi[j]) for j in range(k)]
    est_tabs, seen = [], []
    for j in range(k):
        if not folds[j]:
            est_tabs.append(None)
            seen.append(rng[j])
            continue
        assert (zhi[j] > 0).all(), "a row's sum of squares underflowed to 0 " \
                                   "at pass %d: the walk there is degenerate" % j
        # the geometric mean IN LOGS: the product underflows outright once the
        # walk has squared the row a few times. A row whose smallest member
        # has underflowed has no estimate worth the name -- it takes the
        # largest, and the window that opens is reported.
        lo_c = np.where(zlo[j] > 0, zlo[j], zhi[j])
        est = np.exp(0.5 * (np.log(lo_c) + np.log(zhi[j])))
        seen.append((float((zlo[j] / est).min()), float((zhi[j] / est).max())))
        est_tabs.append(est)
    return (u_lo, u_hi), rng, est_tabs, seen


def suppression(mx, cap):
    """The PUBLIC per-channel suppression of a stream: `d_c` a power of two in
    [1, cap], sized so the largest channel rides at `max / cap`.

    BERT's residual stream is dimension-wise skewed -- at BERT-Base layer 4 the
    LayerNorm output reaches 102 where a typical channel is ~1 -- and one ride
    sizes every channel by the largest, so the typical channel sits at 1/60 of
    the bootstrap's input height and loses those bits. In the batched layout a
    channel is a whole ciphertext, so its own scale is free: it rides the
    weights of whatever reads or writes the stream, and the LayerNorm undoes it
    with an INTEGER multiply, which is why these are powers of two.

    It is not free in EvalMod, though, which is why `cap` exists: the boot's
    error is cubic in the message, so lifting every channel to the full ride
    also lifts every channel to the full cubic term. `cap` is the trade and is
    a knob -- 16 is where the single-prompt line settled.
    """
    mx = np.asarray(mx, dtype=float)
    if cap <= 1.0:
        return np.ones_like(mx)
    ref = mx.max() / cap
    d = np.exp2(np.floor(np.log2(np.maximum(mx, 1e-300) / max(ref, 1e-300))))
    return np.clip(d, 1.0, cap)


# ------------------------------------------------------ the exact sweep
class BandedGelu:
    """One GELU fit per feed-forward TILE, the hidden channels sorted by
    their own `|u|`.

    A hidden channel is a whole ciphertext in the batched layout, so this
    costs nothing at all -- no mask, no band plaintext, no low-rank mode
    plan: the permutation rides `wint`'s columns and `wout`'s rows, which
    leaves the layer's output exactly unchanged. It matters because BERT's
    outliers are dimension-wise: at layer 10 one channel reaches |u| = 130
    where the 99th percentile is 5.5, and a Chebyshev interpolant's error
    depends on `radius / degree` alone -- so ONE shared interval charges
    every channel for the worst one (1e-1 absolute at degree 255).
    """

    def __init__(self, perm, tile, polys):
        self.perm, self.tile, self.polys = perm, tile, polys

    def __call__(self, u):
        out = np.empty_like(u)
        for t, p in enumerate(self.polys):
            idx = self.perm[t * self.tile:(t + 1) * self.tile]
            out[..., idx] = p(u[..., idx])
        return out

    @property
    def escapes(self):
        return sum(p.escapes for p in self.polys)


class LayerPass:
    """One layer's exact forward over the population, a chunk at a time: the
    scores (kept whole -- the row shift is a maximum over the population),
    both variances' ranges, the per-CHANNEL GELU input maxima and the
    maxima the rides are sized from."""

    def __init__(self, N, NH, T, I):
        self.s = np.empty((N, NH, T, T))
        self.var = {"ln1": np.empty((N, T)), "ln2": np.empty((N, T))}
        self.uch = np.zeros(I)
        self.hch = np.zeros(0)      # per-channel |h| and |z| (the two streams
        self.zch = np.zeros(0)      # a LayerNorm writes), for the suppression
        self.hpch = np.zeros(0)     # and the two RESIDUALS, which a wide-window
        self.zpch = np.zeros(0)     # norm bootstraps and so must ride too
        self.mx = {}
        self.s_min, self.s_max = np.inf, -np.inf

    def _mx(self, key, v):
        self.mx[key] = max(self.mx.get(key, 0.0), float(np.abs(v).max()))

    def add(self, c0, c1, r, z):
        self.s[c0:c1] = r["s"]
        self.s_min = min(self.s_min, float(r["s"].min()))
        self.s_max = max(self.s_max, float(r["s"].max()))
        self.var["ln1"][c0:c1] = r["h_pre"].var(axis=-1)
        self.var["ln2"][c0:c1] = r["z_pre"].var(axis=-1)
        np.maximum(self.uch, np.abs(r["u"]).max(axis=(0, 1)), out=self.uch)
        for name, v in (("hch", r["h"]), ("zch", z), ("hpch", r["h_pre"]),
                        ("zpch", r["z_pre"])):
            mx = np.abs(v).max(axis=(0, 1))
            cur = getattr(self, name)
            setattr(self, name, mx if cur.size == 0 else np.maximum(cur, mx))
        for key, v in (("v", r["v"]), ("av", r["av"]), ("h_pre", r["h_pre"]),
                       ("h", r["h"]), ("u", r["u"]), ("z_pre", r["z_pre"]),
                       ("z", z)):
            self._mx(key, v)


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
                    help="exp domain: lo - this")
    ap.add_argument("--exp-margin-hi", type=float, default=0.0,
                    help="exp domain: hi + this (0 = --exp-margin / 4). The "
                         "TOP is where a held-out prompt escapes, because "
                         "`shift` is the population's row maximum and a row "
                         "that beats it lands above `hi` by the excess over "
                         "2^k -- four layers of BERT-Base did exactly that.")
    ap.add_argument("--gelu-margin", type=float, default=1.2)
    ap.add_argument("--ffn-tile", type=int, default=256,
                    help="hidden channels a feed-forward tile: the layer's "
                         "memory peak AND the GELU's band, since a hidden "
                         "channel is a whole ciphertext here")
    ap.add_argument("--sq-ratio", type=float, default=300.0,
                    help="auto k: the smallest k whose first window is under this")
    ap.add_argument("--inv-tol", type=float, default=0.0,
                    help="the inverse square roots' relative tolerance "
                         "(0 = --tol); a degree is a LEVEL, so a ring with a "
                         "lower landing buys its plan back here")
    ap.add_argument("--exp-tol", type=float, default=0.0,
                    help="the exp fit's absolute tolerance (0 = --tol)")
    ap.add_argument("--gelu-tol", type=float, default=0.0,
                    help="the GELU fit's absolute tolerance (0 = --tol); at "
                         "BERT-Base's |u| an absolute tolerance is a hard ask")
    ap.add_argument("--gelu-max-degree", type=int, default=511,
                    help="the GELU's level budget: ceil(log2(deg+1)) levels "
                         "under the boot's landing")
    ap.add_argument("--inv-max-degree", type=int, default=255,
                    help="the Cho inverse square roots' level budget. The "
                         "LAST pass's degree caps the whole plan -- the walk "
                         "lands `P` at `top - 3 - Levels(degree)` and it needs "
                         "4 -- so 255 (8 levels) is the ceiling at landing 16, "
                         "and a wider window is paid in fit error, not levels.")
    ap.add_argument("--ln-max-degree", type=int, default=511,
                    help="the LayerNorms' inverse square roots run on the "
                         "NARROW path (booted), so a degree is free of the "
                         "wide plan -- this is only the ladder's own room")
    ap.add_argument("--chan-cap", type=float, default=16.0,
                    help="the public per-channel suppression's ceiling "
                         "(1 = off): a stream channel is a whole ciphertext, "
                         "so its own scale is free everywhere but EvalMod")
    ap.add_argument("--chunk", type=int, default=64, help="prompts at a time")
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
    chunk = max(1, min(a.chunk, N))
    print("calibration: %d prompt(s), T %d, H %d, %d layers%s" % (
        N, m.T, m.H, m.NL, "" if valid is None else
        ", padded (real tokens %d..%d)" % (int(valid.sum(axis=1).min()),
                                           int(valid.sum(axis=1).max()))))

    layers, hooks = [], []
    cur = x
    in_chan = suppression(np.abs(x).max(axis=(0, 1)), a.chan_cap)
    for L in range(m.NL):
        # ---- the exact layer over the population, a chunk at a time -------
        nxt = np.empty_like(cur)
        p = LayerPass(N, m.NH, m.T, m.I)
        # the ride is sized on the SUPPRESSED stream: a suppression and its
        # ride are one decision
        in_absmax = float((np.abs(cur).max(axis=(0, 1)) / in_chan).max())
        for c0 in range(0, N, chunk):
            c1 = min(N, c0 + chunk)
            r = {}
            nxt[c0:c1] = m.layer(cur[c0:c1], L, mask=None if amask is None else amask[c0:c1],
                                 record=r)
            p.add(c0, c1, r, nxt[c0:c1])
        s = p.s
        # the population row maximum over the REAL keys (the shift is public)
        sv = s if valid is None else np.where(valid[:, None, None, :], s, -np.inf)
        shift = sv.max(axis=(0, 3))                         # [NH, T]
        del sv
        # Cho passes: the smallest k whose first window is narrow enough --
        # of what the POLYNOMIAL sees, which the fold is what decides.
        ks = [a.k] if a.k else [1, 2, 3, 4]
        for k in ks:
            (u_lo, u_hi), rng, est_tabs, seen = exact_cho_ranges(
                s, shift, k, valid,
                fold=fold_passes("none" if a.no_fold else a.fold_passes, k),
                chunk=chunk)
            if seen[0][1] / seen[0][0] <= a.sq_ratio or k == ks[-1]:
                break
        p.s = None
        del s
        # The exp's tolerance is ABSOLUTE, and at small k its domain spans
        # many orders of magnitude: at k = 1 the smallest true value is
        # e^-17 = 4e-8 while a degree-15 interpolant is good to 3e-6, so the
        # tail of y is fit error (of either sign) and the Cho walk that
        # follows it has nothing to normalise. Cho's k is exactly the
        # compression of that range -- `--exp-tol` is what buys k back.
        exp_hi_margin = a.exp_margin_hi if a.exp_margin_hi > 0 else a.exp_margin / 4
        exp_poly = Poly(np.exp, u_lo - a.exp_margin, max(u_hi, 0.0) + exp_hi_margin,
                        a.exp_tol if a.exp_tol > 0 else a.tol, name="exp")
        probe = np.linspace(exp_poly.lo, exp_poly.hi, 4001)
        print("     exp: worst RELATIVE error over the domain %.2e"
              % float(np.abs(exp_poly(probe) / np.exp(probe) - 1).max()))
        inv_polys = [Poly(lambda v: 1.0 / np.sqrt(v), lo / a.margin, hi * a.margin,
                          a.inv_tol if a.inv_tol > 0 else a.tol,
                          relative=True, name="inv%d" % j,
                          max_degree=a.inv_max_degree)
                     for j, (lo, hi) in enumerate(seen)]
        est = None if a.no_fold else est_tabs
        sm = ChoSoftmax(k, shift, exp_poly, inv_polys, est)

        lns = {}
        for tag in ("ln1", "ln2"):
            var = p.var[tag]
            lo, hi = float(var.min()), float(var.max())
            lns[tag] = Poly(lambda v: 1.0 / np.sqrt(v + m.eps), lo / a.margin,
                            hi * a.margin, a.tol, relative=True, name=tag,
                            max_degree=a.ln_max_degree)
        # the GELU, one fit per feed-forward tile, channels sorted by |u|
        perm = np.argsort(-p.uch)
        tile = min(a.ffn_tile, m.I)
        ntile = int(np.ceil(m.I / tile))
        gtol = a.gelu_tol if a.gelu_tol > 0 else a.tol
        gts = []
        for t in range(ntile):
            R = float(p.uch[perm[t * tile:(t + 1) * tile]].max()) * a.gelu_margin
            gts.append(Poly(gelu, -R, R, gtol, name="gelu%d" % t,
                            max_degree=a.gelu_max_degree))
        gl = BandedGelu(perm, tile, gts)

        print("layer %d: Cho k=%d, shift |.| <= %.3f, u [%.2f, %.2f]"
              % (L, k, float(np.abs(shift).max()), u_lo, u_hi))
        for j in range(k):
            print("     window %d: raw %.3g x  ->  seen %.3g x%s" % (
                j, rng[j][1] / rng[j][0], seen[j][1] / seen[j][0],
                "" if est is None or est_tabs[j] is None
                else "  (fold, live^%d)" % fold_live_pow(k)[j]))
        for q in [exp_poly] + inv_polys + [lns["ln1"], lns["ln2"]]:
            print("   ", q)
        print("     gelu   %d tiles of %d: |u| %.1f .. %.1f, deg %s, worst err "
              "%.1e" % (ntile, tile, gts[0].hi, gts[-1].hi,
                        "/".join(str(q.degree) for q in gts),
                        max(q.err for q in gts)))
        h_chan = suppression(p.hch, a.chan_cap)
        z_chan = suppression(p.zch, a.chan_cap)
        h_absmax = float((p.hch / h_chan).max())
        z_absmax = float((p.zch / z_chan).max())
        print("     chan: suppression <= %.0fx  -> |x| %.2f, |h| %.2f -> %.2f, "
              "|z| %.2f -> %.2f" % (a.chan_cap, in_absmax, p.mx["h"], h_absmax,
                                    p.mx["z"], z_absmax))
        hooks.append({"softmax": sm, "ln1": lns["ln1"], "ln2": lns["ln2"], "gelu": gl})
        layers.append({
            "layer": L,
            "in_absmax": in_absmax,
            # the three streams' public per-channel suppression; the next
            # layer's in_chan IS this layer's ln2_chan (the stream is one)
            "in_chan": in_chan.tolist(),
            "ln1_chan": h_chan.tolist(),
            "ln2_chan": z_chan.tolist(),
            "softmax": {"niter": k, "shift": shift.tolist(),
                        "exp": exp_poly.json(),
                        "inv": [q.json() for q in inv_polys],
                        # the fold: est[pass][head][token], multiplied at
                        # serve time by the instance's live count to the
                        # power est_live_pow[pass] (both public)
                        "est": [] if est is None else
                               [[] if e is None else e.tolist() for e in est],
                        "est_live_pow": [] if est is None else fold_live_pow(k),
                        "sq_range": rng, "sq_seen": seen,
                        "s_min": p.s_min, "s_max": p.s_max},
            "v_absmax": p.mx["v"],
            "av_absmax": p.mx["av"],
            # the residuals in the units their own stream carries: a norm
            # whose variance window is wide boots its input, and then the
            # carry has to cover the residual and not just the stream
            "h_pre_absmax": float((p.hpch / in_chan).max()),
            "ln1": dict(lns["ln1"].json(),
                        r_max=float(1.0 / np.sqrt(lns["ln1"].lo + m.eps)),
                        out_absmax=h_absmax),
            # one fit per feed-forward tile, in this permuted channel order:
            # the layer applies the permutation to wint's columns and wout's
            # rows, which leaves the layer's output exactly unchanged
            "gelu": {"tile": tile, "perm": perm.tolist(),
                     "tiles": [q.json() for q in gts],
                     "u_per_channel_max": float(p.uch.max()),
                     "u_per_channel_p99": float(np.quantile(p.uch, 0.99))},
            "u_absmax": p.mx["u"],
            "z_pre_absmax": float((p.zpch / h_chan).max()),
            "ln2": dict(lns["ln2"].json(),
                        r_max=float(1.0 / np.sqrt(lns["ln2"].lo + m.eps)),
                        out_absmax=z_absmax),
        })
        cur = nxt
        in_chan = z_chan

    # The head: the pooler's tanh on the CERTIFIED interval (the last LN's
    # sphere), so nothing about it is a statistic of any prompt.
    head, head_poly = None, None
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
    # output, so each hook is the layer's own. The exact chain runs beside it
    # rather than all at once -- twelve layers of 1000 prompts is 9 GB.
    def run(xs, vmask, label):
        h, e = xs, xs
        am = Model.additive_mask(vmask)
        n = xs.shape[0]
        cs = max(1, min(chunk, n))
        print("%s: %d prompt(s)" % (label, n))
        for L in range(m.NL):
            hk = hooks[L]
            hook = lambda tag, v, mask=None, hk=hk: (  # noqa: E731
                hk[tag](v, mask) if tag == "softmax" else hk[tag](v))
            hn, en = np.empty_like(h), np.empty_like(e)
            se = sr = 0.0
            for c0 in range(0, n, cs):
                c1 = min(n, c0 + cs)
                mc = None if am is None else am[c0:c1]
                hn[c0:c1] = m.layer(h[c0:c1], L, hook, mc)
                en[c0:c1] = m.layer(e[c0:c1], L, None, mc)
                se += float(((hn[c0:c1] - en[c0:c1]) ** 2).sum())
                sr += float((en[c0:c1] ** 2).sum())
            esc = sum(q.escapes for q in (hk["ln1"], hk["ln2"], hk["gelu"],
                                          hk["softmax"].exp) + tuple(hk["softmax"].inv))
            print("  layer %d: chain rms 2^%.2f%s" % (
                L, -bits(np.sqrt(se / sr)), "" if esc == 0 else "  ESCAPES %d" % esc))
            h, e = hn, en
        if head is not None:
            lg, _ = m.head(h, lambda tag, v: head_poly(v))
            lg_ex, u_ex = m.head(e)
            agree = int((lg.argmax(axis=1) == lg_ex.argmax(axis=1)).sum())
            print("  head: logits rms 2^%.2f, |u| <= %.2f (certified %.2f), labels "
                  "agree %d / %d%s" % (
                      -bits(rms_rel(lg, lg_ex)), float(np.abs(u_ex).max()),
                      max(abs(head["u_certified_lo"]), abs(head["u_certified_hi"])),
                      agree, n,
                      "" if head_poly.escapes == 0 else "  ESCAPES %d" % head_poly.escapes))
        return h

    if not a.no_chain:
        run(x, valid, "calibration prompts")
        if a.held_out:
            run(m.prompts(a.held_out), m.valid(mask_beside(a.held_out)),
                "held-out prompts")

    out = {"model": m.meta["model"], "tokens": m.T, "channels": m.H,
           "hidden": m.I, "heads": m.NH, "head_dim": m.D, "ln_eps": m.eps,
           "layers_total": m.NL,
           "ffn_tile": min(a.ffn_tile, m.I),
           "calibration_prompts": N,
           "knobs": {"tol": a.tol, "margin": a.margin, "exp_margin": a.exp_margin,
                     "gelu_margin": a.gelu_margin, "sq_ratio": a.sq_ratio,
                     "exp_margin_hi": exp_hi_margin,
                     "inv_max_degree": a.inv_max_degree,
                     "chan_cap": a.chan_cap,
                     "ffn_tile": min(a.ffn_tile, m.I),
                     "gelu_tol": a.gelu_tol, "gelu_max_degree": a.gelu_max_degree,
                     "inv_tol": a.inv_tol, "ln_max_degree": a.ln_max_degree,
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
