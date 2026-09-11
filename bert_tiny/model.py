"""The BERT encoder in float64, batched over prompts, with hooks.

One forward for everything host-side: `reference.py` runs it exactly and
writes the references, `sim.py` runs it with the crypto's approximations
substituted through the hooks and reports what each costs. The shapes come
from `meta.json` (export.py), never from a constant here.

    x       [N, T, H]   N prompts, T tokens, H channels
    layer   post-LN:  h = LN1(x + O(attn(x)));  z = LN2(h + Wout gelu(Wint h))
"""

import json
import os

import numpy as np

try:
    from scipy.special import erf as _erf
except ImportError:  # pragma: no cover
    import math

    _erf = np.vectorize(math.erf)


def gelu(x):
    return 0.5 * x * (1.0 + _erf(x / np.sqrt(2.0)))


class Model:
    def __init__(self, all_dir):
        self.dir = all_dir
        self.meta = json.load(open(os.path.join(all_dir, "meta.json")))
        m = self.meta
        self.H, self.I, self.NL = m["channels"], m["hidden"], m["layers"]
        self.NH, self.D, self.eps = m["heads"], m["head_dim"], m["ln_eps"]
        self.T = m["tokens"]
        self.w = [self._layer(L) for L in range(self.NL)]

    def _load(self, name, shape):
        a = np.fromfile(os.path.join(self.dir, name), dtype=np.float32)
        assert a.size == int(np.prod(shape)), (name, a.size, shape)
        return a.astype(np.float64).reshape(shape)

    def _layer(self, L):
        H, I = self.H, self.I
        d = "L%02d/" % L
        names = {"wq": (H, H), "wk": (H, H), "wv": (H, H), "wo": (H, H),
                 "wint": (H, I), "wout": (I, H), "bq": (H,), "bk": (H,),
                 "bv": (H,), "bo": (H,), "bint": (I,), "bout": (H,),
                 "attn_norm": (H,), "attn_norm_bias": (H,), "ffn_norm": (H,),
                 "ffn_norm_bias": (H,)}
        return {k: self._load(d + k + ".f32", s) for k, s in names.items()}

    def input(self):
        return self._load("input.f32", (1, self.T, self.H))

    def prompts(self, path=None):
        path = path or os.path.join(self.dir, "prompts", "inputs.f32")
        a = np.fromfile(path, dtype=np.float32).astype(np.float64)
        return a.reshape(-1, self.T, self.H)

    # ------------------------------------------------------------ the layer
    def layer_norm(self, x, g, b, hook=None, tag=""):
        mu = x.mean(axis=-1, keepdims=True)
        xc = x - mu
        var = (xc * xc).mean(axis=-1)                    # [N, T]
        r = hook(tag, var) if hook else 1.0 / np.sqrt(var + self.eps)
        return xc * r[..., None] * g + b

    def attention(self, x, w, hook=None, mask=None):
        """`hook(tag, value)` may replace the softmax; returns [N, T, H]."""
        N, T, H = x.shape
        NH, D = self.NH, self.D
        q = (x @ w["wq"] + w["bq"]).reshape(N, T, NH, D)
        k = (x @ w["wk"] + w["bk"]).reshape(N, T, NH, D)
        v = (x @ w["wv"] + w["bv"]).reshape(N, T, NH, D)
        s = np.einsum("nthd,nshd->nhts", q, k) / np.sqrt(float(D))
        if mask is not None:
            s = s + mask                                   # [N, 1, 1, T]: 0 / -inf
        if hook is not None:
            p = hook("softmax", s)
        else:
            p = np.exp(s - s.max(axis=-1, keepdims=True))
            p = p / p.sum(axis=-1, keepdims=True)
        av = np.einsum("nhts,nshd->nthd", p, v).reshape(N, T, H)
        return av @ w["wo"] + w["bo"], (q, k, v, s, p, av)

    def layer(self, x, L, hook=None, mask=None, record=None):
        w = self.w[L]
        o, parts = self.attention(x, w, hook, mask)
        h_pre = x + o
        h = self.layer_norm(h_pre, w["attn_norm"], w["attn_norm_bias"], hook, "ln1")
        u = h @ w["wint"] + w["bint"]
        g = hook("gelu", u) if hook else gelu(u)
        y = g @ w["wout"] + w["bout"]
        z_pre = h + y
        z = self.layer_norm(z_pre, w["ffn_norm"], w["ffn_norm_bias"], hook, "ln2")
        if record is not None:
            record.update(q=parts[0], k=parts[1], v=parts[2], s=parts[3],
                          p=parts[4], av=parts[5], o=o, h_pre=h_pre, h=h, u=u,
                          y=y, z_pre=z_pre, z=z)
        return z

    def forward(self, x, hook=None, mask=None, records=None):
        outs = []
        for L in range(self.NL):
            rec = {} if records is not None else None
            x = self.layer(x, L, hook, mask, rec)
            if records is not None:
                records.append(rec)
            outs.append(x)
        return outs


def rms_rel(got, want):
    return float(np.sqrt(((got - want) ** 2).sum() / (want ** 2).sum()))


def bits(rel):
    return -np.log2(max(rel, 1e-300))
