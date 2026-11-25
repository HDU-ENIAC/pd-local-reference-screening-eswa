"""Replay the published source-only training schedule on prepared waveforms."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from pd_screening.model import LocalReferenceNet, LogPowerSTFT, normalize_reference


def batches(seed):
    rows = []
    for offset, steps in [(1, 3600), (1001, 600)]:
        rng = np.random.default_rng(seed+offset)
        rows.extend(np.r_[rng.integers(9216, size=32),
                          9216+rng.integers(4608, size=32)] for _ in range(steps))
    return np.stack(rows)


def prepare(path, device):
    with np.load(path, allow_pickle=False) as z:
        x, refs, ci, y = [z[k].copy() for k in ["x", "refs", "context", "y"]]
    assert x.shape == (13824, 512) and refs.shape == (108, 8, 512)
    assert ci.shape == y.shape == (13824,)
    assert set(np.unique(y)) == {0, 1} and y.sum() == 6912
    for c in range(len(refs)):
        ix = np.flatnonzero(ci == c)
        x[ix], refs[c], _ = normalize_reference(x[ix], refs[c])
    tf = LogPowerSTFT().to(device)
    def convert(v):
        with torch.no_grad():
            return torch.cat([tf(torch.from_numpy(v[i:i+128]).to(device))
                              for i in range(0, len(v), 128)])
    return (convert(x), convert(refs.reshape(-1, 512)).reshape(108, 8, 32, 128),
            torch.tensor(ci, dtype=torch.long, device=device),
            torch.tensor(y, dtype=torch.float32, device=device))


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--seed", choices=[611013, 611014, 611015], type=int, default=611013)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--steps", type=int, default=4200, help="Shorter runs are smoke tests, not paper results")
    p.add_argument("--output", type=Path, default=Path("outputs/training"))
    a = p.parse_args()
    if not 1 <= a.steps <= 4200:
        p.error("steps must be in 1..4200")
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(a.seed)
    net = LocalReferenceNet().to(a.device)
    # These are random, untrained initial states, not pretrained encoders.
    # They preserve the original initialization RNG sequence after refactoring.
    net.load_state_dict(torch.load(Path(__file__).parent / "initial_states" / f"{a.seed}.pt",
                                   map_location=a.device, weights_only=True))
    net.train()
    tx, tr, ci, y = prepare(a.data, a.device)
    fixed = {k: v.clone() for k, v in net.named_buffers()}
    def optimizer():
        return torch.optim.AdamW(net.parameters(), lr=.001, weight_decay=.0001)
    opt = optimizer()
    history = []
    for step, ids in enumerate(batches(a.seed)[:a.steps]):
        if step == 3600:
            opt = optimizer()
        opt.zero_grad(set_to_none=True)
        unique, inverse = torch.unique(ci[ids], return_inverse=True)
        scores = net.window_scores(tx[ids], tr[unique], inverse)
        loss = F.binary_cross_entropy_with_logits(scores, y[ids])
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.)
        opt.step()
        if step == 0 or (step+1) % 300 == 0:
            row = dict(step=step+1, loss=float(loss.detach()))
            history.append(row)
            print(row, flush=True)
    assert all(torch.equal(v, fixed[k]) for k, v in net.named_buffers())
    torch.save(net.eval().state_dict(), a.output/f"{a.seed}.pt")
    (a.output/"training.json").write_text(json.dumps(dict(seed=a.seed, steps=a.steps,
        full_schedule=a.steps == 4200, history=history), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
