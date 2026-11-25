"""Exercise the API with artificial Gaussian backgrounds; not a scientific test."""
from pathlib import Path
import numpy as np
import torch
from pd_screening.model import LocalReferenceNet, score_windows
from pd_screening.simulation import normal_response, pd_pulse
from pd_screening.calibration import threshold


def main():
    torch.set_num_threads(4)
    rng = np.random.default_rng(142)
    g = dict(f=.10,tau=24.,amp=10.)
    def window():
        return (rng.normal(size=512)+normal_response(g,rng,True)).astype(np.float32)
    refs, cal = np.stack([window() for _ in range(8)]),np.stack([window() for _ in range(32)])
    x = np.stack([window() for _ in range(4)])
    x[1,64:320] += 3*pd_pulse(.12,12.)
    net = LocalReferenceNet()
    net.load_state_dict(torch.load(Path(__file__).parent/'checkpoints/611013.pt',map_location='cpu',weights_only=True))
    cut = threshold(score_windows(net,cal,refs))
    scores = score_windows(net,x,refs)
    print('API demonstration only. These are not the paper evaluation inputs.')
    print('scores:',scores,'threshold:',cut,'admitted:',scores>cut)


if __name__ == '__main__':
    main()
