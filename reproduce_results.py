"""Recalculate paper tables and plot data from frozen per-window scores.

This does not retrain baselines or rerun their inference.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from pd_screening.calibration import threshold, metrics

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT/'results'
NAMES = ['quant', 'moderntcn', 'cnn_bilstm', 'pd_cnn', 'mantis', 'proposed']
METRICS = ['fpr', 'retention', 'ba']


def read(name):
    return pd.read_csv(RESULTS/f'{name}.csv.gz', dtype={'site':str}, float_precision='round_trip')


def summarize(frame, keys):
    # Average within recording before averaging over the balanced recording groups.
    parent = frame.groupby(keys+['site','record'])[METRICS].mean().reset_index()
    return parent.groupby(keys)[METRICS].mean().reset_index()


def main_table():
    rows = []
    for method in NAMES:
        d = read('main_'+method)
        for (seed, tag), g in d.groupby(['seed','tag']):
            condition = g.condition.iloc[0]
            if condition not in ['nominal','nuisance_shift']:
                continue
            q = g[g.role == 'query']
            for bank in range(4):
                c = g[(g.role == 'calibration') & (g.bank == bank)].sort_values('index').score.to_numpy()[:32]
                values = metrics(q[q.y == 0].score, q[(q.y == 1) & (q.snr == 5)].score, threshold(c))
                rows.append(dict(method=method, condition=condition, seed=seed, bank=bank,
                    site=g.site.iloc[0], record=g.record.iloc[0], **values))
    return summarize(pd.DataFrame(rows), ['method','condition'])


def design_table():
    rows = []
    for method in ['candidate_only','mean_only','global','proposed']:
        q, c = read(f'design_{method}_query'), read(f'design_{method}_calibration')
        queries = dict(tuple(q.groupby(['seed','tag'])))
        for (seed, tag, bank), group in c.groupby(['seed','tag','bank']):
            v = queries[seed, tag]
            if v.condition.iloc[0] != 'nuisance_shift':
                continue
            cut = threshold(group.sort_values('index').score.to_numpy()[:32])
            values = metrics(v[v.y == 0].score, v[(v.y == 1) & (v.snr == 5)].score, cut)
            rows.append(dict(method=method, seed=seed, bank=bank,
                site=v.site.iloc[0], record=v.record.iloc[0], **values))
    return summarize(pd.DataFrame(rows), ['method'])


def downstream_tables():
    gq, gc = read('design_proposed_query'), read('design_proposed_calibration')
    bq, bc = read('backend_query'), read('backend_calibration')
    qkeys = ['seed','tag','query','condition','site','record','pair','y','snr']
    ckeys = ['seed','tag','bank','index','condition','site','record']
    rows = []
    for backend in ['mantis_query_full','mantis_reference_full']:
        q = gq[qkeys+['score']].merge(bq[bq.method == backend][qkeys+['score']],
             on=qkeys, validate='one_to_one', suffixes=('_gate','_backend'))
        c = gc[ckeys+['score']].merge(bc[bc.method == backend][ckeys+['score']],
             on=ckeys, validate='one_to_one', suffixes=('_gate','_backend'))
        assert len(q) == len(gq) and len(c) == len(gc)
        query = dict(tuple(q.groupby(['seed','tag'])))
        for (seed, tag, bank), v in c.groupby(['seed','tag','bank']):
            z = query[seed, tag]
            if z.condition.iloc[0] != 'nuisance_shift':
                continue
            v = v.sort_values('index')
            a, b = v.score_gate.to_numpy(), v.score_backend.to_numpy()
            assert len(a) == 128
            ga, bb = threshold(a), threshold(b)
            m = int(((a > ga) & (b > bb)).sum())
            tight = max(bb, float(np.sort(b)[len(b)-m-1])) if m < len(b) else bb
            assert (b > tight).sum() <= m
            pos = ((z.y == 1) & (z.snr == 5)).to_numpy()
            neg = (z.y == 0).to_numpy()
            decisions = dict(original=z.score_backend.to_numpy() > bb,
                raised=z.score_backend.to_numpy() > tight,
                admission=(z.score_gate.to_numpy() > ga) & (z.score_backend.to_numpy() > bb))
            for strategy, mask in decisions.items():
                fpr, ret = float(mask[neg].mean()), float(mask[pos].mean())
                rows.append(dict(backend=backend, strategy=strategy, seed=seed, bank=bank,
                    site=z.site.iloc[0], record=z.record.iloc[0], fpr=fpr, retention=ret, ba=(1-fpr+ret)/2))
    return summarize(pd.DataFrame(rows), ['backend','strategy'])


def calibration_sweeps():
    rows = []
    settings = [('normal_budget', n, .05) for n in [32,64,128,256,512]]
    settings += [('alpha', 128, a) for a in [.01,.025,.05,.075,.1]]
    for method in ['proposed','pd_cnn']:
        d = read('main_'+method)
        d = d[d.condition == 'nuisance_shift']
        for (seed,tag), g in d.groupby(['seed','tag']):
            q = g[g.role == 'query']
            c = g[g.role == 'calibration'].sort_values(['bank','index']).score.to_numpy()
            assert len(c) == 512
            for sweep, n, alpha in settings:
                for first in range(4):
                    order = np.concatenate([np.arange(128)+128*((first+j)%4) for j in range(4)])
                    cut = threshold(c[order[:n]], alpha)
                    v = metrics(q[q.y == 0].score, q[(q.y == 1) & (q.snr == 5)].score, cut)
                    rows.append(dict(method=method, sweep=sweep, normal_budget=n, alpha=alpha,
                        seed=seed, bank=first, site=g.site.iloc[0], record=g.record.iloc[0], **v))
    return summarize(pd.DataFrame(rows), ['method','sweep','normal_budget','alpha'])


def check_expected(table1, table2, downstream):
    expected = {
        'quant': [[2.20,54.59,76.20],[2.38,54.26,75.94]],
        'moderntcn': [[2.75,57.42,77.34],[2.51,52.64,75.07]],
        'cnn_bilstm': [[2.78,61.04,79.13],[2.93,56.41,76.74]],
        'pd_cnn': [[3.08,90.40,93.66],[3.63,71.00,83.68]],
        'mantis': [[2.88,73.67,85.39],[2.02,64.06,81.02]],
        'proposed': [[3.01,91.21,94.10],[3.22,88.77,92.77]],
    }
    for method, values in expected.items():
        for condition, wanted in zip(['nominal','nuisance_shift'], values):
            row = table1[(table1.method == method) & (table1.condition == condition)]
            np.testing.assert_allclose(row[METRICS].to_numpy()[0]*100, wanted, atol=.005001, rtol=0)
    for method, wanted in {'candidate_only':[3.97,72.79,84.41], 'mean_only':[3.29,83.98,90.35],
                            'global':[3.14,81.09,88.97], 'proposed':[3.19,86.04,91.42]}.items():
        np.testing.assert_allclose(table2[table2.method == method][METRICS].to_numpy()[0]*100,
                                   wanted, atol=.005001, rtol=0)
    for backend, vals in {
        'mantis_query_full': [[5.29,70.28,82.50],[.99,56.18,77.60],[.46,67.45,83.50]],
        'mantis_reference_full': [[6.04,77.86,85.91],[1.92,66.60,82.34],[.49,73.70,86.60]]}.items():
        for strategy, wanted in zip(['original','raised','admission'], vals):
            row = downstream[(downstream.backend == backend) & (downstream.strategy == strategy)]
            np.testing.assert_allclose(row[METRICS].to_numpy()[0]*100, wanted, atol=.005001, rtol=0)
    return dict(table_I='36 rounded values matched', table_II='12 rounded values matched',
                tables_III_IV='18 rounded values matched',
                basis='Recalculated from archived per-window scores; not new training')


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--output', type=Path, default=Path('outputs/result_replay'))
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    t1, t2, back = main_table(), design_table(), downstream_tables()
    checks = check_expected(t1, t2, back)
    curves = calibration_sweeps()
    for name, d in [('table_I',t1),('table_II',t2),('tables_III_IV',back),('figure_4',curves)]:
        d.to_csv(a.output/f'{name}.csv',index=False)
    auc = read('reference_auc')
    auc = auc[(auc.condition == 'nuisance_shift') & (auc.snr == 5)]
    auc['reference'] = np.where(auc.record == auc.donor_record,'matched','replacement')
    auc.groupby(['reference','site','record']).auc.mean().reset_index().to_csv(a.output/'figure_5.csv',index=False)
    timing = read('timing_results')
    timing = timing[(timing.snr == 5) & (timing.policy == 'recalibrated') & (timing.normal_budget == 32)]
    timing.groupby('offset')[['fpr','retention']].mean().reset_index().to_csv(a.output/'figure_6.csv',index=False)
    (a.output/'checks.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    print(json.dumps(checks,indent=2))
    print(t1.to_string(index=False))


if __name__ == '__main__':
    main()
