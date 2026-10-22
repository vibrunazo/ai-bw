"""Do a role model's decisions hold control groups together as the pros' do? On the held-out snapshots of a response
dataset with keys (ingest/key_labels.py), units sharing a key: how often two key-mates take the same decision, and
when one takes a new decision (not keep) how often a key-mate takes the same; for the pros' labels, for the same
decisions shuffled within each snapshot (no grouping), and for the model's decisions drawn from its odds (with the
keys shown to a model trained with them, and hidden). The offline check before the key input goes into play
(docs/PROGRESS.md, 2026-10-22).

    python -m eval.key_coherence --model runs/roles/<run>/model.pt --data data/interim/roles/response_T_att_again_keys.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

import train.roles as TR
from gary.policy.roles import RoleModel


def coherence(dec: np.ndarray, keys: np.ndarray, mask: np.ndarray, keep: int) -> tuple[float, float]:
    """(key-mates' same-decision share, a new decision's key-mates taking it too) over snapshots."""
    pairs = same = new_n = new_same = 0
    for i in range(len(dec)):
        m = mask[i]
        k, y = keys[i][m], dec[i][m]
        for key in set(k.tolist()) - {-1, -2}:
            a = y[k == key]
            n = len(a)
            if n < 2:
                continue
            pairs += n * (n - 1) / 2
            for v in set(a.tolist()):
                c = int((a == v).sum())
                same += c * (c - 1) / 2
                if v != keep:
                    new_same += c * (c - 1)
                    new_n += c * (n - 1)
    return same / max(1, pairs), new_same / max(1, new_n)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", nargs="+", required=True, help="role models (each with and, if trained so, without keys)")
    ap.add_argument("--data", default="data/interim/roles/response_T_att_again_keys.npz")
    ap.add_argument("--snapshots", type=int, default=20000)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    d = np.load(args.data)
    rng = np.random.default_rng(0)
    first = RoleModel.load(args.model[0], device)
    cfg = first.config
    _, val, names = TR.load(Path(args.data), True, cfg["net"].get("move", False), cfg.get("attended", False),
                            cfg.get("budget", False), "again" in cfg["roles"], use_keys=True)
    keep = names.index("keep")
    n = min(args.snapshots, len(val[0]))
    idx = np.arange(n)
    labels = val[2][:n]
    mask = labels >= 0
    keys = val[TR.COLS.index("key")][:n].astype(int)
    print(f"{n} held-out snapshots, {int(mask.sum())} labeled units, {np.mean(keys[mask] >= 0):.0%} on a key")
    print(f"  pros' decisions:      key-mates same {coherence(labels, keys, mask, keep)[0]:.1%}, "
          f"a new decision shared {coherence(labels, keys, mask, keep)[1]:.1%}")
    shuffled = labels.copy()
    for i in range(n):
        m = mask[i]
        shuffled[i][m] = rng.permutation(labels[i][m])
    s, t = coherence(shuffled, keys, mask, keep)
    print(f"  shuffled in snapshot: key-mates same {s:.1%}, a new decision shared {t:.1%}")
    for path in args.model:
        model = RoleModel.load(path, device)
        for show in ([True, False] if model.config.get("keys") else [None]):
            decs: dict[str, np.ndarray] = {}
            for s0 in range(0, n, 256):
                b = idx[s0:s0 + 256]
                cols, saved = TR.COLS[:], None
                if not model.config.get("keys"):     # (a model without keys: its inputs without the key one-hot)
                    TR.COLS[:] = [c for c in cols if c != "key"]
                elif not show:                       # (keys hidden: every unit on no key)
                    saved = val[TR.COLS.index("key")][b].copy()
                    val[TR.COLS.index("key")][b] = -1
                got = TR.batch(val, b, device)
                TR.COLS[:] = cols
                if saved is not None:
                    val[TR.COLS.index("key")][b] = saved
                with torch.no_grad():
                    out = model.net(*got[0])
                    logits = out[0] if isinstance(out, tuple) else out
                    p = torch.softmax(logits, -1).cpu().numpy()
                w = p.shape[1]
                cdf = np.cumsum(p, -1)
                for mode in ("independent", "shared", "likeliest"):
                    if mode == "independent":
                        u = rng.random(p.shape[:2])
                    elif mode == "shared":            # (one uniform per key: key-mates' draws coupled)
                        u = rng.random(p.shape[:2])
                        kb = keys[b, :w]
                        for i in range(len(b)):
                            for key in set(kb[i].tolist()) - {-1, -2}:
                                u[i, kb[i] == key] = u[i, kb[i] == key][0]
                    if mode == "likeliest":
                        draw = p.argmax(-1)
                    else:
                        draw = np.minimum((cdf < u[..., None]).sum(-1), p.shape[-1] - 1)
                    decs.setdefault(mode, np.full(labels.shape, -1, np.int64))[b, :w] = draw
            label = Path(path).parent.name + ("" if show is None else (", keys shown" if show else ", keys hidden"))
            for mode, dm in decs.items():
                s, t = coherence(dm, keys, mask, keep)
                agree = np.mean((dm == labels)[mask])
                print(f"  model {label}, {mode} draws: key-mates same {s:.1%}, a new decision shared {t:.1%}; "
                      f"the pro's decision {agree:.1%}")


if __name__ == "__main__":
    main()
