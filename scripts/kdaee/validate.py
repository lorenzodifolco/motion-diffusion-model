"""Physical validation of converted motions (KDAEE or generated) against a HumanML3D sample.

Usage (from repo root):
    python -m scripts.kdaee.validate --data data/kdaee_hml3d [--hml_n 2000] [--render N_PER_EMOTION] [--render_ids id ...]

Works on HumanML3D-style new_joints (n_frames, 22, 3), metres, Y-up, 20 fps.
Writes <data>/validation/{window_stats.csv, summary.md, renders/*.mp4}.
"""
import argparse
import csv
import glob
import os
import random
from multiprocessing import Pool

import numpy as np

from data_loaders.humanml.utils.paramUtil import t2m_kinematic_chain
from scripts.kdaee.hml3d import HML_PARENTS

FPS = 20
FOOT_JOINTS = {'l_ankle': 7, 'r_ankle': 8, 'l_foot': 10, 'r_foot': 11}
CONTACT_HEIGHT = {7: 0.10, 8: 0.10, 10: 0.05, 11: 0.05}   # m above floor, height-based contact
SKATE_SPEED = 0.10       # m/s: horizontal foot speed above which a contact frame counts as skating
FLOAT_HEIGHT = 0.10      # m: lowest foot above this = both feet off the ground


def motion_stats(j):
    """j: (T, 22, 3). Returns dict of per-window physical statistics."""
    bones = np.stack([np.linalg.norm(j[:, i] - j[:, p], axis=-1) for i, p in enumerate(HML_PARENTS) if p >= 0], 1)
    lowest = j[:, list(FOOT_JOINTS.values()), 1].min(1)
    skate_v, n_contact, n_skate = [], 0, 0
    for k in FOOT_JOINTS.values():
        v = np.linalg.norm(np.diff(j[:, k, [0, 2]], axis=0), axis=1) * FPS
        c = j[1:, k, 1] < CONTACT_HEIGHT[k]
        skate_v.append(v[c])
        n_contact += c.sum()
        n_skate += (v[c] > SKATE_SPEED).sum()
    skate_v = np.concatenate(skate_v) if n_contact else np.zeros(1)
    vel = np.linalg.norm(np.diff(j, axis=0), axis=-1) * FPS                 # (T-1, 22)
    acc = np.linalg.norm(np.diff(j, n=2, axis=0), axis=-1) * FPS ** 2
    jerk = np.linalg.norm(np.diff(j, n=3, axis=0), axis=-1) * FPS ** 3
    return {
        'n_frames': len(j),
        'bone_len_std_mm': float(bones.std(0).mean() * 1000),
        'bone_len_std_max_mm': float(bones.std(0).max() * 1000),
        'lowest_foot_min': float(lowest.min()),
        'lowest_foot_med': float(np.median(lowest)),
        'float_frac': float((lowest > FLOAT_HEIGHT).mean()),
        'penetration_frac': float((lowest < -0.01).mean()),
        'skate_speed_mean': float(skate_v.mean()),
        'skate_frac': float(n_skate / max(n_contact, 1)),
        'root_speed_mean': float(np.linalg.norm(np.diff(j[:, 0, [0, 2]], axis=0), axis=1).mean() * FPS),
        'joint_speed_mean': float(vel.mean()),
        'joint_acc_mean': float(acc.mean()),
        'joint_jerk_mean': float(jerk.mean()),
    }


def _stats_file(path):
    return os.path.basename(path)[:-4], motion_stats(np.load(path))


def _render(job):
    from data_loaders.humanml.utils.plot_script import plot_3d_motion
    path, out, title = job
    j = np.load(path)
    clip = plot_3d_motion(out, t2m_kinematic_chain, j, title=title, dataset='humanml', fps=FPS)
    clip.duration = len(j) / FPS
    clip.write_videofile(out, fps=FPS, logger=None)
    return out


def summarize(rows, keys):
    return {k: (np.median([r[k] for r in rows]), np.percentile([r[k] for r in rows], 5),
                np.percentile([r[k] for r in rows], 95)) for k in keys}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='data/kdaee_hml3d')
    p.add_argument('--hml_root', default='dataset/HumanML3D')
    p.add_argument('--hml_n', type=int, default=2000)
    p.add_argument('--render', type=int, default=3, help='windows rendered per emotion (different actors)')
    p.add_argument('--render_ids', nargs='*', default=[], help='extra window ids to render')
    p.add_argument('--workers', type=int, default=12)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    out_dir = os.path.join(args.data, 'validation')
    os.makedirs(os.path.join(out_dir, 'renders'), exist_ok=True)
    meta = {r['id']: r for r in csv.DictReader(open(os.path.join(args.data, 'meta.csv')))}
    rng = random.Random(args.seed)
    hml_files = sorted(f for f in glob.glob(os.path.join(args.hml_root, 'new_joints', '*.npy'))
                       if not os.path.basename(f).startswith('M'))       # skip mirrored copies
    hml_files = rng.sample(hml_files, min(args.hml_n, len(hml_files)))
    kd_files = [os.path.join(args.data, 'new_joints', i + '.npy') for i in meta]

    with Pool(args.workers) as pool:
        kd = dict(pool.map(_stats_file, kd_files, chunksize=8))
        hml = dict(pool.map(_stats_file, hml_files, chunksize=8))

    keys = list(next(iter(kd.values())))
    with open(os.path.join(out_dir, 'window_stats.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['id', 'actor', 'emotion'] + keys)
        w.writeheader()
        for i, s in kd.items():
            w.writerow(dict(id=i, actor=meta[i]['actor'], emotion=meta[i]['emotion'], **s))

    groups = {'HumanML3D': list(hml.values()), 'KDAEE (all)': list(kd.values())}
    for emo in sorted({m['emotion'] for m in meta.values()}):
        groups[f'KDAEE {emo}'] = [kd[i] for i in kd if meta[i]['emotion'] == emo]
    lines = ['| group | n | ' + ' | '.join(keys[1:]) + ' |', '|---' * (len(keys) + 1) + '|']
    for g, rows in groups.items():
        s = summarize(rows, keys[1:])
        lines.append(f'| {g} | {len(rows)} | ' + ' | '.join(f'{v[0]:.3g} [{v[1]:.3g}, {v[2]:.3g}]' for v in s.values()) + ' |')
    table = '\n'.join(lines)
    with open(os.path.join(out_dir, 'summary.md'), 'w') as f:
        f.write('# Physical validation (median [p5, p95] over windows)\n\n' + table + '\n')
    print(table)

    jobs = []
    for emo in sorted({m['emotion'] for m in meta.values()}):
        cands = [i for i, m in meta.items() if m['emotion'] == emo]
        rng.shuffle(cands)
        picked, actors = [], set()
        for i in cands:
            if meta[i]['actor'] not in actors:
                picked.append(i)
                actors.add(meta[i]['actor'])
            if len(picked) == args.render:
                break
        jobs += [(i, f'{emo}_{i}') for i in picked]
    jobs += [(i, f'check_{i}') for i in args.render_ids]
    jobs = [(os.path.join(args.data, 'new_joints', i + '.npy'), os.path.join(out_dir, 'renders', name + '.mp4'),
             f"{i} {meta[i]['emotion']} {meta[i]['scenario_code']}") for i, name in jobs]
    with Pool(args.workers) as pool:
        done = pool.map(_render, jobs)
    print(f'rendered {len(done)} videos to {os.path.join(out_dir, "renders")}')


if __name__ == '__main__':
    main()
