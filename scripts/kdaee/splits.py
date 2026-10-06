"""Subject-wise splits for KDAEE (never clip-level random splits).

Folds:
  - 'sanity': 2 test + 2 validation actors (1 F + 1 M each, seeded draw), 18 training actors.
  - 'loso_<actor>': 1 test actor, 2 validation actors (1 F + 1 M, deterministic rotation), 19 training actors.
    One LoRA adapter per fold.

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


def loso_folds(females, males):
    """Test = each actor; val = next actor of the same gender + actor at the same rank of the other gender."""
    folds = {}
    for same, other in ((females, males), (males, females)):
        for i, actor in enumerate(same):
            val = [same[(i + 1) % len(same)], other[i % len(other)]]
            folds[f'loso_{actor}'] = {'test': [actor], 'val': sorted(val)}
    return dict(sorted(folds.items()))


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

    folds = {'sanity': sanity_fold(females, males, args.seed), **loso_folds(females, males)}
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
