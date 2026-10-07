"""Downstream emotion recognition with XGBoost, one fold end-to-end (Phase 5).

Stages (outputs in save/emotion/<fold>/):
  select_cfg  real + LoRA synthetic 1:1 for each candidate CFG scale, macro-F1 on the validation actor
              -> cfg_choice.json (test windows enter only the split assertions, never features or models)
  run         the 4 conditions on the test actors (anti-leakage assertions first) -> runs.csv, per_class.csv,
              subjects.csv, predictions.csv, confusion.json, importance.csv

    python -m emotion.run_fold select_cfg --fold gkf5_0 --lora_ckpt CKPT --pools 1.5=DIR 2.5=DIR 4=DIR
    python -m emotion.run_fold run --fold gkf5_0 --lora_ckpt CKPT --lora_pools D0 D1 D2 --base_pools D0 D1 D2

Conditions: real | real+aug | real+lora | real+base. In 2-4 the added windows are 1:1 with the real training
windows and class-balanced (round(n_train / 7) per emotion). XGBoost has fixed hyper-parameters and a fixed
seed; seed s of conditions 2-4 = augmentation RNG s / synthetic pool generated with seed s (LoRA and base pools
with the same seed share prompts, lengths and initial noise). The baseline is deterministic and run once.
"""
import argparse
import csv
import json
import os
import time
from multiprocessing import Pool

import numpy as np
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)

from emotion.data import (EMOTIONS, augmented_set, balanced_subset, check_lora, check_pool, check_split,
                          check_training_sets, load_fold, load_synthetic)
from emotion.features import FEATURE_NAMES, window_features
from emotion.models import XGB_PARAMS, fit_xgb, gain_importance

CONDITIONS = ['real', 'real+aug', 'real+lora', 'real+base']
LABELS = list(range(len(EMOTIONS)))


def macro_f1(y, p):
    return f1_score(y, p, average='macro', labels=LABELS, zero_division=0)


def _feat(j):
    return np.array([window_features(j)[n] for n in FEATURE_NAMES], np.float32)


def add_features(windows, workers=16):
    todo = [w for w in windows if 'x' not in w]
    if todo:
        with Pool(workers) as pool:
            feats = pool.map(_feat, [w['joints'] for w in todo], chunksize=16)
        for w, f in zip(todo, feats):
            w['x'] = f
    return windows


def xy(windows):
    return np.stack([w['x'] for w in windows]), np.array([w['label'] for w in windows])


def n_per_class(train):
    return int(round(len(train) / len(EMOTIONS)))


def write_csv(path, rows):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


# ------------------------------------------------------------------ stages

def stage_select_cfg(args, spec, sets, out_dir):
    check_split(spec, sets)
    check_lora(args.fold, spec, args.lora_ckpt)
    tr, va = add_features(sets['train']), add_features(sets['val'])
    train_ids = {w['id'] for w in tr}
    Xva, yva = xy(va)
    model = fit_xgb(*xy(tr), Xva, yva)
    res = {'real': dict(val_macro_f1=macro_f1(yva, model.predict(Xva)), best_iteration=model.best_iteration)}
    print(f"real only: val macro-F1 {res['real']['val_macro_f1']:.3f}", flush=True)
    for item in args.pools:
        scale, d = item.split('=', 1)
        pool = add_features(load_synthetic(d))
        check_pool('lora', pool, args.fold, spec, train_ids, args.lora_ckpt)
        assert all(float(w['meta']['scale']) == float(scale) for w in pool)
        syn = balanced_subset(pool, n_per_class(tr), np.random.default_rng(0))
        check_training_sets(sets, tr + syn, va, sets['test'])
        model = fit_xgb(*xy(tr + syn), Xva, yva)
        res[scale] = dict(val_macro_f1=macro_f1(yva, model.predict(Xva)), best_iteration=model.best_iteration,
                          pool=d)
        print(f'CFG {scale}: val macro-F1 (real + LoRA 1:1) {res[scale]["val_macro_f1"]:.3f}', flush=True)
    best = max((s for s in res if s != 'real'), key=lambda s: res[s]['val_macro_f1'])
    json.dump(dict(scores=res, chosen=best, val_actors=spec['val'],
                   criterion='macro-F1 on the validation actor, XGBoost on real + LoRA synthetic 1:1 (pool seed 0)'),
              open(os.path.join(out_dir, 'cfg_choice.json'), 'w'), indent=1)
    print('chosen CFG scale:', best)


def stage_run(args, spec, sets, out_dir):
    # ---- anti-leakage
    check_split(spec, sets)
    check_lora(args.fold, spec, args.lora_ckpt)
    tr, va, te = (add_features(sets[s]) for s in ('train', 'val', 'test'))
    train_ids = {w['id'] for w in tr}
    cfg = json.load(open(os.path.join(out_dir, 'cfg_choice.json')))['chosen']
    assert len(args.lora_pools) == len(args.base_pools)
    pools = {'lora': [add_features(load_synthetic(d)) for d in args.lora_pools],
             'base': [add_features(load_synthetic(d)) for d in args.base_pools]}
    for kind, ps in pools.items():
        for p in ps:
            check_pool(kind, p, args.fold, spec, train_ids, args.lora_ckpt)
            assert len({w['meta']['seed'] for w in p}) == 1, 'pool mixes generation seeds'
            assert all(float(w['meta']['scale']) == float(cfg) for w in p), 'pool not at the chosen CFG scale'
        assert len({p[0]['meta']['seed'] for p in ps}) == len(ps), f'{kind} pools share a generation seed'
    for pl, pb in zip(pools['lora'], pools['base']):           # same prompts / lengths / seed for LoRA and base
        key = lambda w: (w['meta']['seed'], w['meta']['prompt'], w['meta']['n_frames'], w['meta']['source_window'])
        assert {w['id']: key(w) for w in pl} == {w['id']: key(w) for w in pb}, 'LoRA and base prompts differ'
    print('anti-leakage checks passed', flush=True)

    Xva, yva = xy(va)
    Xte, yte = xy(te)
    npc = n_per_class(tr)
    rows, pc_rows, subj_rows, pred_rows, imp_rows, conf = [], [], [], [], [], {}
    t0 = time.time()
    for cond in CONDITIONS:
        seeds = [0] if cond == 'real' else list(range(len(pools['lora'])))
        for seed in seeds:
            rng = np.random.default_rng(seed)
            if cond == 'real':
                added = []
            elif cond == 'real+aug':
                added = add_features(augmented_set(tr, npc, rng))
            else:
                added = balanced_subset(pools[cond.split('+')[1]][seed], npc, rng)
            train = tr + added
            check_training_sets(sets, train, va, te)
            model = fit_xgb(*xy(train), Xva, yva)
            proba = model.predict_proba(Xte)
            pred = proba.argmax(1)
            key = dict(fold=args.fold, condition=cond, seed=seed)
            prec, rec, f1, sup = precision_recall_fscore_support(yte, pred, labels=LABELS, zero_division=0)
            rows.append(dict(key, n_real=len(tr), n_added=len(added), best_iteration=model.best_iteration,
                             val_macro_f1=macro_f1(yva, model.predict(Xva)), macro_f1=macro_f1(yte, pred),
                             bal_acc=balanced_accuracy_score(yte, pred), accuracy=float((pred == yte).mean()),
                             **{f'f1_{e}': v for e, v in zip(EMOTIONS, f1)}))
            pc_rows += [dict(key, emotion=e, precision=prec[k], recall=rec[k], f1=f1[k], support=int(sup[k]))
                        for k, e in enumerate(EMOTIONS)]
            for a in spec['test']:
                idx = [i for i, w in enumerate(te) if w['actor'] == a]
                subj_rows.append(dict(key, actor=a, n=len(idx), macro_f1=macro_f1(yte[idx], pred[idx]),
                                      bal_acc=balanced_accuracy_score(yte[idx], pred[idx])))
            pred_rows += [dict(key, id=w['id'], actor=w['actor'], label=EMOTIONS[w['label']],
                               pred=EMOTIONS[p], **{f'p_{e}': round(float(q), 5) for e, q in zip(EMOTIONS, pr)})
                          for w, p, pr in zip(te, pred, proba)]
            imp_rows += [dict(key, feature=n, gain=g) for n, g in gain_importance(model, FEATURE_NAMES).items()]
            conf.setdefault(cond, []).append(confusion_matrix(yte, pred, labels=LABELS).tolist())
            print(f'[{time.time() - t0:.0f}s] {cond} seed {seed}: n_train {len(train)} trees {model.best_iteration + 1} '
                  f"val F1 {rows[-1]['val_macro_f1']:.3f} | test macro-F1 {rows[-1]['macro_f1']:.3f} "
                  f"bal.acc {rows[-1]['bal_acc']:.3f}", flush=True)
    for name, data in (('runs.csv', rows), ('per_class.csv', pc_rows), ('subjects.csv', subj_rows),
                       ('predictions.csv', pred_rows), ('importance.csv', imp_rows)):
        write_csv(os.path.join(out_dir, name), data)
    json.dump(conf, open(os.path.join(out_dir, 'confusion.json'), 'w'))
    json.dump(dict(xgb=XGB_PARAMS, cfg_scale=cfg, lora_ckpt=args.lora_ckpt, lora_pools=args.lora_pools,
                   base_pools=args.base_pools, spec=spec, n_per_class_added=npc, features=FEATURE_NAMES),
              open(os.path.join(out_dir, 'run_config.json'), 'w'), indent=1)
    print(f'done in {time.time() - t0:.0f}s')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['select_cfg', 'run'])
    p.add_argument('--fold', default='gkf5_0')
    p.add_argument('--data', default='data/kdaee_hml3d')
    p.add_argument('--out', default='save/emotion')
    p.add_argument('--lora_ckpt', required=True)
    p.add_argument('--pools', nargs='*', default=[], help='select_cfg: scale=dir of seed-0 LoRA pools')
    p.add_argument('--lora_pools', nargs='*', default=[], help='run: one LoRA pool per generation seed')
    p.add_argument('--base_pools', nargs='*', default=[], help='run: one base-MDM pool per generation seed')
    args = p.parse_args()
    out_dir = os.path.join(args.out, args.fold)
    os.makedirs(out_dir, exist_ok=True)
    spec, sets = load_fold(args.data, args.fold)
    if args.stage == 'select_cfg':
        stage_select_cfg(args, spec, sets, out_dir)             # test windows only enter the split assertions
    else:
        stage_run(args, spec, sets, out_dir)


if __name__ == '__main__':
    main()
