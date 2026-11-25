"""Inference on an NPZ with raw x, refs, and calibration arrays."""
import argparse
from pathlib import Path
import numpy as np
import torch
from pd_screening.model import LocalReferenceNet, score_windows
from pd_screening.calibration import threshold


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--checkpoint", default="checkpoints/611013.pt", type=Path)
    p.add_argument("--output", default="outputs/predictions.npz", type=Path)
    p.add_argument("--alpha", type=float, default=.05)
    p.add_argument("--normal-budget", type=int, default=32)
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    torch.set_num_threads(4)
    if a.output.exists():
        raise FileExistsError(a.output)
    net = LocalReferenceNet().to(a.device)
    net.load_state_dict(torch.load(a.checkpoint, map_location=a.device, weights_only=True))
    with np.load(a.input, allow_pickle=False) as z:
        cal = z["calibration"].reshape(-1, 512)
        if not 1 <= a.normal_budget <= len(cal):
            raise ValueError("normal-budget exceeds available calibration windows")
        scores = score_windows(net, z["x"], z["refs"])
        normal = score_windows(net, cal[:a.normal_budget], z["refs"])
    cut = threshold(normal, a.alpha)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.output, scores=scores, calibration_scores=normal,
                        threshold=cut, admitted=scores > cut)
    print(f"queries={len(scores)}, admitted={int((scores > cut).sum())}, threshold={cut:.6g}")


if __name__ == "__main__":
    main()
