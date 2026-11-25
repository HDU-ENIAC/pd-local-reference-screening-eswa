"""Run the frozen proposed model on all prepared main-evaluation contexts."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from pd_screening.model import LocalReferenceNet, score_windows


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--data', required=True, type=Path, help='Directory with manifest.json and per-context NPZs')
    p.add_argument('--output', type=Path, default=Path('outputs/fresh_scores'))
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--limit', type=int, default=0, help='Nonzero limit is a smoke test only')
    p.add_argument('--check-archived', action='store_true', help='Compare main-evaluation scores against bundled originals')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    root = Path(__file__).resolve().parent
    entries = json.loads((a.data/'manifest.json').read_text(encoding='utf-8'))
    if a.limit:
        entries = entries[:a.limit]
    old = pd.read_csv(root/'results/main_proposed.csv.gz',float_precision='round_trip') if a.check_archived else None
    checks = []
    for seed in [611013,611014,611015]:
        net = LocalReferenceNet().to(a.device)
        net.load_state_dict(torch.load(root/'checkpoints'/f'{seed}.pt',map_location=a.device,weights_only=True))
        rows = []
        for e in entries:
            with np.load(a.data/(e['tag']+'.npz'),allow_pickle=False) as z:
                q = score_windows(net,z['x'],z['refs'])
                c = score_windows(net,z['calibration'].reshape(-1,512),z['refs'])
                info = dict(seed=seed,method='proposed',**e)
                rows.extend(dict(info,role='query',bank=-1,index=i,y=int(z['y'][i]),pair=int(z['pair'][i]),
                                 snr=int(z['snr'][i]),score=float(v)) for i,v in enumerate(q))
                rows.extend(dict(info,role='calibration',bank=i//128,index=i%128,y=0,pair=-1,snr=-1,
                                 score=float(v)) for i,v in enumerate(c))
        d = pd.DataFrame(rows)
        d.to_csv(a.output/f'{seed}.csv.gz',index=False,compression='gzip')
        if old is not None:
            keys = ['seed','tag','role','bank','index']
            expected = old[(old.seed == seed) & old.tag.isin([e['tag'] for e in entries])]
            j = d.merge(expected[keys+['score']],on=keys,suffixes=('_new','_old'),validate='one_to_one')
            assert len(j) == len(d) == len(expected)
            gap = float(abs(j.score_new-j.score_old).max())
            assert gap < 3e-4, gap
            checks.append(dict(seed=seed,score_rows=len(j),maximum_absolute_logit_error=gap))
        print(seed,'scored',len(d),'windows',flush=True)
    (a.output/'check.json').write_text(json.dumps(dict(full_run=not a.limit,
        contexts=len(entries),device=a.device,checks=checks),indent=2),encoding='utf-8')


if __name__ == '__main__':
    main()
