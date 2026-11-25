"""Construct the published main-evaluation windows from separately obtained data."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from pd_screening.simulation import main_conditions


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--raw-dir', type=Path, required=True,
                   help='Directory of <idMeasurement>.npy files from the original data provider')
    p.add_argument('--output', type=Path, default=Path('outputs/main_data'))
    a = p.parse_args()
    records = pd.read_csv(Path(__file__).parent/'data/main_records.csv', dtype={'site':str})
    a.output.mkdir(parents=True, exist_ok=False)
    entries = []
    for row in records.itertuples(index=False):
        raw = np.load(a.raw_dir/f'{row.measurement}.npy', allow_pickle=False)
        for condition, arrays in main_conditions(raw, row.record):
            if condition not in {'nominal', 'nuisance_shift'}:
                continue
            tag = f'{condition}_{row.record}'
            np.savez_compressed(a.output/f'{tag}.npz', **arrays)
            entries.append(dict(tag=tag, record=row.record, site=row.site, condition=condition))
        print(row.record, 'complete', flush=True)
    (a.output/'manifest.json').write_text(json.dumps(entries, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
