"""Subject-wise splits for KDAEE (never clip-level random splits).

Folds:
  - 'sanity': 2 test + 2 validation actors (1 F + 1 M each, seeded draw), 18 training actors.
  - 'gkf5_<k>': GroupKFold by actor, 5 folds stratified by gender (4-5 test actors), 1 validation actor drawn
    from the remaining ones (gender alternating with the fold index), the rest for training (Phase 5, development).
  - 'loso_<actor>': 1 test actor, 1 validation actor (the next actor in sorted order), 20 training actors.
    One LoRA adapter per fold; the validation actor is used for LoRA early stopping, XGBoost early stopping and
    the choice of the CFG scale.

Writes <data>/splits/<fold>/{train,val,test}.txt (window ids, HumanML3D split-file format) and
<data>/splits/folds.json ({fold: {train: [actors], val: [...], test: [...]}}).

Usage (from repo root):
    python -m scripts.kdaee.splits --data data/kdaee_hml3d [--seed 0]
"""
import argparse
import csv
import json
import os
import random


def sanity_fold(females, males, seed):
    rng = random.Random(seed)
    f, m = rng.sample(females, 2), rng.sample(males, 2)
    return {'test': sorted([f[0], m[0]]), 'val': sorted([f[1], m[1]])}


def loso_folds(actors):
    """Test = each actor; val = the next actor in sorted order (cyclic)."""
    return {f'loso_{a}': {'test': [a], 'val': [actors[(i + 1) % len(actors)]]} for i, a in enumerate(actors)}


def group_kfolds(females, males, k, seed):
    """Gender-stratified GroupKFold over actors (one group per actor): shuffle each gender with `seed`, deal
    actors round-robin to k folds (males continue where females stopped, so fold sizes differ by <= 1)."""
    rng = random.Random(seed)
    f, m = rng.sample(females, len(females)), rng.sample(males, len(males))
    tests = [[] for _ in range(k)]
    for i, a in enumerate(f + m):
        tests[i % k].append(a)
    folds = {}
    for i, test in enumerate(tests):
        rest_f = [a for a in f if a not in test]
        rest_m = [a for a in m if a not in test]
        val = [rest_f[i % len(rest_f)]] if i % 2 == 0 else [rest_m[i % len(rest_m)]]
        folds[f'gkf{k}_{i}'] = {'test': sorted(test), 'val': sorted(val)}
    return folds


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='data/kdaee_hml3d')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    meta = list(csv.DictReader(open(os.path.join(args.data, 'meta.csv'))))
    gender = {m['actor']: m['gender'] for m in meta}
    actors = sorted(gender)
    females = [a for a in actors if gender[a] == 'Female']
    males = [a for a in actors if gender[a] == 'Male']

    folds = {'sanity': sanity_fold(females, males, args.seed), **group_kfolds(females, males, 5, args.seed),
             **loso_folds(actors)}
    for spec in folds.values():
        held = set(spec['test']) | set(spec['val'])
        assert not set(spec['test']) & set(spec['val'])
        spec['train'] = [a for a in actors if a not in held]

    root = os.path.join(args.data, 'splits')
    for name, spec in folds.items():
        os.makedirs(os.path.join(root, name), exist_ok=True)
        for split in ('train', 'val', 'test'):
            ids = [m['id'] for m in meta if m['actor'] in spec[split]]
            with open(os.path.join(root, name, f'{split}.txt'), 'w') as f:
                f.write('\n'.join(ids) + '\n')
    with open(os.path.join(root, 'all.txt'), 'w') as f:
        f.write('\n'.join(m['id'] for m in meta) + '\n')
    with open(os.path.join(root, 'folds.json'), 'w') as f:
        json.dump(folds, f, indent=1)
    print(f'{len(folds)} folds written to {root}')


if __name__ == '__main__':
    main()
