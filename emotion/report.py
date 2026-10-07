"""Aggregate Phase 5 (XGBoost) results over the available folds.

    python -m emotion.report [--folds gkf5_0 gkf5_1 ...] [--out save/emotion/report]

- summary table of the 4 conditions: test macro-F1 / balanced accuracy, mean +- std over folds (fold value =
  mean over the generation / augmentation seeds); with a single fold the std is over seeds; chance = 1/7;
- paired Wilcoxon signed-rank test over folds (baseline vs each condition) with the matched-pairs
  rank-biserial correlation as effect size;
- per-fold and per-class CSV, per-class F1 table;
- confusion matrix per condition (aggregated over folds and seeds, row-normalised);
- XGBoost feature importance (normalised total gain) for the baseline and for real + LoRA.
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon

from emotion.data import EMOTIONS
from emotion.run_fold import CONDITIONS

CHANCE = 1 / len(EMOTIONS)


def rank_biserial(diff):
    d = np.asarray(diff)
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    r = rankdata(np.abs(d))
    return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='save/emotion')
    p.add_argument('--folds', nargs='*', default=None)
    p.add_argument('--out', default='')
    args = p.parse_args()
    folds = args.folds or sorted(os.path.basename(os.path.dirname(f))
                                 for f in glob.glob(os.path.join(args.root, '*', 'runs.csv')))
    out = args.out or os.path.join(args.root, 'report_' + folds[0] if len(folds) == 1 else 'report')
    os.makedirs(out, exist_ok=True)
    load = lambda name: pd.concat([pd.read_csv(os.path.join(args.root, f, name)) for f in folds])
    runs, pc, imp = load('runs.csv'), load('per_class.csv'), load('importance.csv')
    runs['n_train'] = runs.n_real + runs.n_added
    lines = ['# Phase 5 — downstream emotion recognition (XGBoost)', '',
             f'Folds: {", ".join(folds)}; test actors: {load("subjects.csv").actor.nunique()}; '
             f'chance level (7 classes) = {CHANCE:.3f}.', '']

    # ---- summary
    per_fold = runs.groupby(['fold', 'condition'])[['macro_f1', 'bal_acc', 'val_macro_f1', 'best_iteration',
                                                     'n_train']].mean().reset_index()
    seed_sd = runs.groupby(['fold', 'condition']).macro_f1.std().groupby('condition').mean()
    per_fold.to_csv(os.path.join(out, 'per_fold.csv'), index=False)
    multi = len(folds) > 1
    src = per_fold if multi else runs
    spread = 'folds' if multi else 'seeds (single fold)'
    base = per_fold[per_fold.condition == 'real'].set_index('fold')
    lines += [f'## Test macro-F1 / balanced accuracy (mean ± std over {spread})', '',
              '| condition | n train | macro-F1 | Δ vs real | balanced acc. | seed sd | val macro-F1 | trees |'
              + (' Wilcoxon p (folds) | r | folds improved |' if multi else ''),
              '|---' * (8 + 3 * multi) + '|']
    wil_rows = []
    for cond in CONDITIONS:
        g, gf = src[src.condition == cond], per_fold[per_fold.condition == cond].set_index('fold')
        if g.empty:
            continue
        d = (gf.macro_f1 - base.macro_f1.reindex(gf.index)).values
        row = (f'| {cond} | {gf.n_train.mean():.0f} | {g.macro_f1.mean():.3f} ± {g.macro_f1.std(ddof=1) if len(g) > 1 else 0:.3f} '
               f'| {d.mean():+.3f} | {g.bal_acc.mean():.3f} ± {g.bal_acc.std(ddof=1) if len(g) > 1 else 0:.3f} '
               f'| {seed_sd.get(cond, np.nan) if cond != "real" else 0:.3f} | {gf.val_macro_f1.mean():.3f} '
               f'| {gf.best_iteration.mean() + 1:.0f} |')
        if multi:
            if cond != 'real' and np.any(d != 0):
                stat, pval = wilcoxon(d)
                r = rank_biserial(d)
                wil_rows.append(dict(condition=cond, n_folds=len(d), mean_delta=d.mean(), W=stat, p=pval, r=r,
                                     improved=int((d > 0).sum()), worsened=int((d < 0).sum())))
                row += f' {pval:.3f} | {r:+.2f} | {(d > 0).sum()}/{len(d)} |'
            else:
                row += ' — | — | — |'
        lines.append(row)
    lines += ['', 'seed sd = std of test macro-F1 over the 3 augmentation / generation seeds within a fold '
                  '(the baseline is deterministic: fixed XGBoost seed). val = validation actor (early stopping).']
    if multi:
        pd.DataFrame(wil_rows).to_csv(os.path.join(out, 'wilcoxon.csv'), index=False)
        lines.append(f'Wilcoxon over {len(folds)} folds: the exact two-sided p-value cannot go below '
                     f'{2 / 2 ** len(folds):.3f}.')
    lines.append('')

    # ---- per-class F1
    pcm = pc.groupby(['condition', 'emotion'])[['precision', 'recall', 'f1']].mean().reset_index()
    pcm.to_csv(os.path.join(out, 'per_class_mean.csv'), index=False)
    lines += ['## Per-class F1 (mean over folds and seeds)', '',
              '| condition | ' + ' | '.join(EMOTIONS) + ' |', '|---' * (len(EMOTIONS) + 1) + '|']
    for cond in CONDITIONS:
        g = pcm[pcm.condition == cond].set_index('emotion')
        if not g.empty:
            lines.append(f'| {cond} | ' + ' | '.join(f'{g.loc[e, "f1"]:.2f}' for e in EMOTIONS) + ' |')
    lines.append('')

    # ---- feature importance
    im = imp.groupby(['condition', 'feature']).gain.mean().reset_index()
    im.pivot(index='feature', columns='condition', values='gain').sort_values('real', ascending=False).to_csv(
        os.path.join(out, 'importance_mean.csv'))
    lines += ['## XGBoost feature importance (normalised total gain, mean over folds/seeds) — top 10', '',
              '| rank | real | gain | real+lora | gain |', '|---' * 5 + '|']
    top = {c: im[im.condition == c].sort_values('gain', ascending=False).reset_index(drop=True)
           for c in ('real', 'real+lora')}
    for k in range(10):
        lines.append(f'| {k + 1} | ' + ' | '.join(f'{top[c].feature[k]} | {top[c].gain[k]:.3f}'
                                                  if len(top[c]) > k else '— | —' for c in top) + ' |')
    lines.append('')

    # ---- plots
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    conf = {}
    for f in folds:
        for c, mats in json.load(open(os.path.join(args.root, f, 'confusion.json'))).items():
            conf[c] = conf.get(c, 0) + np.sum(mats, axis=0) / len(mats)       # mean over seeds, sum over folds
    fig, axes = plt.subplots(1, len(conf), figsize=(4.2 * len(conf), 4.2))
    for ax, c in zip(np.atleast_1d(axes), [c for c in CONDITIONS if c in conf]):
        m = conf[c] / conf[c].sum(1, keepdims=True)
        ax.imshow(m, vmin=0, vmax=1, cmap='Blues')
        for i in range(len(EMOTIONS)):
            for j in range(len(EMOTIONS)):
                ax.text(j, i, f'{m[i, j]:.2f}', ha='center', va='center', fontsize=7,
                        color='white' if m[i, j] > 0.5 else 'black')
        ax.set_xticks(range(len(EMOTIONS)))
        ax.set_yticks(range(len(EMOTIONS)))
        ax.set_xticklabels([e[:3] for e in EMOTIONS])
        ax.set_yticklabels([e[:3] for e in EMOTIONS])
        f1 = src[src.condition == c].macro_f1.mean()
        ax.set_title(f'{c} (macro-F1 {f1:.3f})', fontsize=9)
        ax.set_xlabel('predicted')
    np.atleast_1d(axes)[0].set_ylabel('true')
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'confusion.png'), dpi=120)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    for ax, c in zip(axes, ('real', 'real+lora')):
        t = top[c].head(20)[::-1]
        ax.barh(t.feature, t.gain, color='tab:blue' if c == 'real' else 'tab:orange')
        ax.set_title(f'{c}: XGBoost total gain (normalised)', fontsize=9)
        ax.tick_params(axis='y', labelsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'importance.png'), dpi=120)
    plt.close(fig)

    with open(os.path.join(out, 'report.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print(f'written to {out}')


if __name__ == '__main__':
    main()
