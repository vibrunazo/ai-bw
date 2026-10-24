"""Learning the army's commit decision from locked paired branches, offline (docs/SELF_PLAY.md, "Learning beyond
imitation", rules 2 and 3): at a moment of a league game the Terran's commitment was forced to engage in one arm and
to move in the other, each held and reversed at the pros' hazard (eval/pairs.py, commit_t), and both arms were
played 90 s with the dice shared after the forced decision. The arm with the better outcome labels the moment.

The commit model's engage-against-move odds are fine-tuned on those labels, each weighted by how much the better arm
won (advantage-weighted, its weight capped), anchored to where it started: a KL term on the pros' own ticks (army
dataset), so the rest of what it learned from them stays. Games split by file: a fifth held out.

The gate written before it runs (docs/PROGRESS.md, the pilot's minimum detectable effect): on held-out moments the
fine-tuned model picks the better arm more often than chance, and more often than its start, the interval above;
then 96 paired games.

    python -m train.commit_pairs --rounds CP110a CP120a --out runs/commit_exp/TvZ_T_pairs
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "ingest"))

from eval import monitor, priority  # noqa: E402
from gary.policy.army import GRID, ArmyModel  # noqa: E402

ENGAGE, MOVE = 1, 2
MIN_DIFF = 0.5                  # a pair whose arms differ less than this labels nothing (a tie)
WEIGHT_CAP = 3.0                # a pair's weight: its difference over the median difference, capped


def pairs(rounds: list[str]) -> dict:
    """The training pairs of these rounds: state (grid, glob) at the branch, which arm won, by how much, the game."""
    out = {"grid": [], "glob": [], "engage_won": [], "diff": [], "game": []}
    for r in rounds:
        for f in sorted(glob.glob(str(REPO / "runs" / "league" / r / "*" / "*.pairs.npz"))):
            z = np.load(f)
            if "dec/pair" not in z.files:
                continue
            game = int(hashlib.sha1(f.encode()).hexdigest()[:8], 16)
            for k in range(len(z["reward"])):
                if z["ab"][k]:
                    continue
                rows = np.nonzero(z["dec/pair"] == k)[0]
                arm0 = [j for j in rows if z["dec/arm"][j] == 0 and z["dec/choice"][j] == ENGAGE]
                arm1 = [j for j in rows if z["dec/arm"][j] == 1 and z["dec/choice"][j] == MOVE]
                if not arm0 or not arm1:
                    continue
                d = float(z["reward"][k][0] - z["reward"][k][1])
                if abs(d) < MIN_DIFF:
                    continue
                out["grid"].append(z["dec/grid"][arm0[0]].astype(np.float32))
                out["glob"].append(z["dec/glob"][arm0[0]].astype(np.float32))
                out["engage_won"].append(float(d > 0))
                out["diff"].append(abs(d))
                out["game"].append(game)
    return {k: np.array(v) for k, v in out.items()}


def engage_logit(net, grid: torch.Tensor, glob_: torch.Tensor, channels: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The model's engage-against-move logit and its keep / engage / move log-odds."""
    _, kind, _ = net(grid.view(-1, channels, GRID, GRID), glob_)
    logp = F.log_softmax(kind, -1)
    return logp[:, ENGAGE] - logp[:, MOVE], logp


def score(net, data: dict, idx: np.ndarray, channels: int) -> dict:
    """On these pairs: how often the model's likelier of engage and move is the arm that won (and the AUC)."""
    from sklearn.metrics import roc_auc_score
    with torch.no_grad():
        lg, _ = engage_logit(net, torch.from_numpy(data["grid"][idx]), torch.from_numpy(data["glob"][idx]), channels)
    lg = lg.numpy()
    y = data["engage_won"][idx]
    acc = float(((lg > 0) == (y > 0.5)).mean())
    se = float(np.sqrt(acc * (1 - acc) / max(1, len(y))))
    auc = float(roc_auc_score(y, lg)) if 0 < y.sum() < len(y) else None
    return {"n": int(len(y)), "picks_winner": round(acc, 3), "interval": [round(acc - 2 * se, 3), round(acc + 2 * se, 3)],
            "auc": None if auc is None else round(auc, 3), "engage_won_share": round(float(y.mean()), 3)}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rounds", nargs="+", required=True)
    ap.add_argument("--model", default=None, help="the commit model to start from (default: the deployed, newest)")
    ap.add_argument("--anchor", type=float, default=1.0, help="the KL term's weight, on the pros' ticks")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=3)
    priority.add_argument(ap)
    args = ap.parse_args()
    priority.apply(args.priority)
    from gary.agent.manifests import model_path
    from train.commit import jobs, load
    start = Path(args.model) if args.model else model_path("commit/latest", "TvZ", "T")
    model = ArmyModel.load(start)
    spec, net = model.spec, model.net
    ch = spec.channels
    data = pairs(args.rounds)
    n = len(data["engage_won"])
    held = (data["game"] % 5) == 0
    tr, te = np.nonzero(~held)[0], np.nonzero(held)[0]
    print(f"{n} pairs labelled ({tr.size} train, {te.size} held out); engage won {data['engage_won'].mean():.2f}", flush=True)
    before = {"train": score(net, data, tr, ch), "held": score(net, data, te, ch)}
    print("start:", json.dumps(before), flush=True)
    # the anchor: the start model's odds on the pros' ticks (a sample), kept by a KL term
    pro_jobs = jobs("TvZ", "T", version="v3")
    pro = load([j for j in pro_jobs if not j[2]][:400] + [next(j for j in pro_jobs if j[2])], spec, args.workers)["train"]
    pg = torch.from_numpy(pro[0][:60000].astype(np.float32))
    px = torch.from_numpy(pro[1][:60000].astype(np.float32))
    with torch.no_grad():
        _, ref = engage_logit(net, pg, px, ch)
    w = data["diff"] / np.median(data["diff"][tr])
    w = np.minimum(w, WEIGHT_CAP)
    G, X = torch.from_numpy(data["grid"]), torch.from_numpy(data["glob"])
    Y, W = torch.from_numpy(data["engage_won"]).float(), torch.from_numpy(w).float()
    opt = torch.optim.AdamW([p for name, p in net.named_parameters() if name.startswith(("kind.", "glob."))],
                            lr=args.lr, weight_decay=1e-4)
    rng = np.random.default_rng(0)
    net.train()
    for epoch in monitor.track(range(args.epochs), args.epochs, "epochs"):
        perm = rng.permutation(tr)
        for s in range(0, len(perm), 256):
            b = perm[s:s + 256]
            lg, _ = engage_logit(net, G[b], X[b], ch)
            loss = (F.binary_cross_entropy_with_logits(lg, Y[b], reduction="none") * W[b]).mean()
            a = rng.choice(len(pg), 1024, replace=False)
            _, lp = engage_logit(net, pg[a], px[a], ch)
            loss = loss + args.anchor * F.kl_div(lp, ref[a], log_target=True, reduction="batchmean")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        net.eval()
        if (epoch + 1) % 5 == 0:
            print(f"epoch {epoch + 1}: held {json.dumps(score(net, data, te, ch))}", flush=True)
        net.train()
    net.eval()
    with torch.no_grad():
        _, lp = engage_logit(net, pg, px, ch)
        kl = float(F.kl_div(lp, ref, log_target=True, reduction="batchmean"))
    after = {"train": score(net, data, tr, ch), "held": score(net, data, te, ch), "kl_on_pros": round(kl, 4)}
    print("after:", json.dumps(after), flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ArmyModel(spec, net.cpu(), {**model.config, "pairs": {"from": str(start), "rounds": args.rounds}}).save(out / "model.pt")
    (out / "report.json").write_text(json.dumps({"from": str(start), "rounds": args.rounds, "before": before,
                                                 "after": after}, indent=1), encoding="utf-8")
    print(f"saved {out / 'model.pt'}")


if __name__ == "__main__":
    main()
