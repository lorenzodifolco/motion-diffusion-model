"""Evaluation of generated motions against real KDAEE windows (sanity fold).

    python -m lora.eval_synth --samples save/lora/sanity_r8_a8_full/samples \
        --extra base_mdm_2.5=save/lora/sanity_r8_a8_full/samples_base_mdm/scale_2.5

Sets: real train / real test windows of the fold, every samples/scale_* dir, plus --extra name=dir.
1. physical metrics (scripts.kdaee.validate.motion_stats), all emotions pooled;
2. kinematic features per emotion + Wasserstein-1 distance to real train (normalised by the real-train std),
   with real test -> real train as the reference scale (natural between-subject variation);
3. memorisation: nearest-neighbour distance synthetic->train vs test->train, in the T2M evaluator embedding
   space and in a raw pose space (heading-normalised local joints resampled to 64 frames);
4. FID / diversity in the T2M evaluator space (evaluator trained on HumanML3D: relative comparison only);
5. renders: the same sample ids at every scale.
Writes <samples>/../eval/{report.md, kinematics.csv, physical.csv, nn.csv} and renders.
"""
import argparse
import csv
import glob
import os
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
import torch
from scipy.stats import spearmanr, wasserstein_distance

from data_loaders.humanml.networks.evaluator_wrapper import build_evaluators
from data_loaders.humanml.utils.metrics import (calculate_activation_statistics, calculate_diversity,
                                                calculate_frechet_distance)
from data_loaders.humanml.utils.word_vectorizer import POS_enumerator
from scripts.kdaee.validate import FPS, _render, motion_stats

EMOTIONS = ['Angry', 'Disgust', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprise']
KIN = ['speed', 'acc', 'jerk', 'root_speed', 'bbox_volume', 'hand_extension']
PHYS = ['bone_len_std_mm', 'lowest_foot_min', 'lowest_foot_med', 'float_frac', 'penetration_frac',
        'skate_speed_mean', 'skate_frac', 'joint_jerk_mean']


def local_joints(feat):
    """Heading-normalised joints from HumanML3D features: root at (0, root_y, 0), ric for the others."""
    ric = feat[:, 4:67].reshape(len(feat), 21, 3)
    root = np.zeros((len(feat), 1, 3))
    root[:, 0, 1] = feat[:, 3]
    return np.concatenate([root, ric], 1)


def kinematics(feat, joints):
    vel = np.linalg.norm(np.diff(joints, axis=0), axis=-1) * FPS
    acc = np.linalg.norm(np.diff(joints, n=2, axis=0), axis=-1) * FPS ** 2
    jerk = np.linalg.norm(np.diff(joints, n=3, axis=0), axis=-1) * FPS ** 3
    loc = local_joints(feat)
    ext = loc.max(1) - loc.min(1)
    return {
        'speed': vel.mean(), 'acc': acc.mean(), 'jerk': jerk.mean(),
        'root_speed': (np.linalg.norm(np.diff(joints[:, 0, [0, 2]], axis=0), axis=1) * FPS).mean(),
        'bbox_volume': np.prod(ext, axis=1).mean(),
        'hand_extension': np.linalg.norm(loc[:, [20, 21]] - loc[:, [0]], axis=-1).mean(),
    }


def raw_descriptor(feat, n=64):
    loc = local_joints(feat)
    t = np.linspace(0, len(loc) - 1, n)
    flat = loc.reshape(len(loc), -1)
    return np.stack([np.interp(t, np.arange(len(loc)), flat[:, k]) for k in range(flat.shape[1])], 1).ravel()


def _window(args):
    wid, vec_path, joint_path = args
    feat, joints = np.load(vec_path), np.load(joint_path)
    return wid, feat, motion_stats(joints), kinematics(feat, joints), raw_descriptor(feat)


def load_set(root, ids, emotion_of, workers):
    jobs = [(i, os.path.join(root, 'new_joint_vecs', i + '.npy'), os.path.join(root, 'new_joints', i + '.npy'))
            for i in ids]
    with Pool(workers) as pool:
        res = pool.map(_window, jobs, chunksize=8)
    return {'ids': [r[0] for r in res], 'emotion': [emotion_of[r[0]] for r in res], 'feat': [r[1] for r in res],
            'phys': [r[2] for r in res], 'kin': [r[3] for r in res], 'raw': np.stack([r[4] for r in res])}


class Evaluator:
    def __init__(self, device):
        opt = {'dataset_name': 't2m', 'device': device, 'dim_word': 300, 'max_motion_length': 196,
               'dim_pos_ohot': len(POS_enumerator), 'dim_motion_hidden': 1024, 'max_text_len': 20,
               'dim_text_hidden': 512, 'dim_coemb_hidden': 512, 'dim_pose': 263,
               'dim_movement_enc_hidden': 512, 'dim_movement_latent': 512, 'checkpoints_dir': '.',
               'unit_length': 4}
        _, self.motion_enc, self.movement_enc = build_evaluators(opt)
        self.motion_enc.to(device).eval()
        self.movement_enc.to(device).eval()
        self.device = device
        self.mean = np.load('dataset/t2m_mean.npy')          # evaluator normalisation (not MDM's Mean/Std)
        self.std = np.load('dataset/t2m_std.npy')

    @torch.no_grad()
    def embed(self, feats, bs=128):
        out = np.zeros((len(feats), 512), np.float32)
        order = np.argsort([-len(f) for f in feats])
        for s in range(0, len(order), bs):
            idx = order[s:s + bs]
            lens = [min(len(feats[i]), 196) for i in idx]
            x = np.zeros((len(idx), 196, 263), np.float32)
            for k, i in enumerate(idx):
                x[k, :lens[k]] = (feats[i][:lens[k]] - self.mean) / self.std
            x = torch.from_numpy(x).to(self.device)
            mov = self.movement_enc(x[..., :-4])
            emb = self.motion_enc(mov, torch.tensor(lens) // 4)
            out[idx] = emb.cpu().numpy()
        return out


def nn_dist(query, ref):
    q, r = torch.from_numpy(query).float(), torch.from_numpy(ref).float()
    return torch.cdist(q, r).min(1).values.numpy()


def med(x):
    return float(np.median(x))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--samples', default='save/lora/sanity_r8_a8_full/samples')
    p.add_argument('--extra', nargs='*', default=[], help='name=dir of additional synthetic sets')
    p.add_argument('--data', default='data/kdaee_hml3d')
    p.add_argument('--fold', default='sanity')
    p.add_argument('--render_per_emotion', type=int, default=2)
    p.add_argument('--workers', type=int, default=16)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()
    out_dir = os.path.join(os.path.dirname(os.path.normpath(args.samples)), 'eval')
    os.makedirs(os.path.join(out_dir, 'renders'), exist_ok=True)

    meta = {m['id']: m for m in csv.DictReader(open(os.path.join(args.data, 'meta.csv')))}
    split = lambda s: [l.strip() for l in open(os.path.join(args.data, 'splits', args.fold, s + '.txt')) if l.strip()]
    sets = {'real_train': load_set(args.data, split('train'), {i: m['emotion'] for i, m in meta.items()}, args.workers),
            'real_test': load_set(args.data, split('test'), {i: m['emotion'] for i, m in meta.items()}, args.workers)}
    synth_dirs = {f"lora_cfg{os.path.basename(d).split('_')[1]}": d
                  for d in sorted(glob.glob(os.path.join(args.samples, 'scale_*')), key=lambda d: float(d.split('_')[-1]))}
    synth_dirs.update(dict(e.split('=', 1) for e in args.extra))
    synth_meta = {}
    for name, d in synth_dirs.items():
        sm = {m['id']: m for m in csv.DictReader(open(os.path.join(d, 'meta.csv')))}
        synth_meta[name] = sm
        sets[name] = load_set(d, list(sm), {i: m['emotion'] for i, m in sm.items()}, args.workers)
    synth_names = list(synth_dirs)

    ev = Evaluator(args.device)
    for s in sets.values():
        s['emb'] = ev.embed(s['feat'])

    lines = [f'# Phase 4 evaluation — fold {args.fold}', '',
             f"Sets: " + ', '.join(f'{k} (n={len(v["ids"])})' for k, v in sets.items()), '']

    # 1. physical
    lines += ['## 1. Physical metrics (median [p5, p95] over windows, all emotions)', '',
              '| set | ' + ' | '.join(PHYS) + ' |', '|---' * (len(PHYS) + 1) + '|']
    phys_rows = []
    for name, s in sets.items():
        cells = []
        for k in PHYS:
            v = np.array([ph[k] for ph in s['phys']])
            cells.append(f'{np.median(v):.3g} [{np.percentile(v, 5):.3g}, {np.percentile(v, 95):.3g}]')
            phys_rows.append(dict(set=name, metric=k, median=np.median(v), p5=np.percentile(v, 5), p95=np.percentile(v, 95)))
        lines.append(f'| {name} | ' + ' | '.join(cells) + ' |')
    with open(os.path.join(out_dir, 'physical.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(phys_rows[0])); w.writeheader(); w.writerows(phys_rows)

    # 2. kinematics per emotion
    def kin_values(s, emo, k):
        return np.array([kn[k] for kn, e in zip(s['kin'], s['emotion']) if e == emo])

    kin_rows, w1 = [], defaultdict(list)
    for emo in EMOTIONS:
        for k in KIN:
            ref = kin_values(sets['real_train'], emo, k)
            sd = ref.std() + 1e-12
            for name in ['real_test'] + synth_names:
                v = kin_values(sets[name], emo, k)
                d = wasserstein_distance(v, ref) / sd
                w1[name].append(d)
                kin_rows.append(dict(emotion=emo, feature=k, set=name, mean=v.mean(), real_train_mean=ref.mean(),
                                     w1_norm=d))
    with open(os.path.join(out_dir, 'kinematics.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(kin_rows[0])); w.writeheader(); w.writerows(kin_rows)

    lines += ['', '## 2. Kinematics per emotion', '',
              'Normalised W1 = Wasserstein-1(set, real train) / std(real train), per emotion x feature; '
              'real_test is the reference (unseen real subjects).', '',
              '| set | mean W1 (42 emotion x feature) | median | max | Spearman of per-emotion means vs real train (avg over features) |',
              '|---|---|---|---|---|']
    for name in ['real_test'] + synth_names:
        rhos = []
        for k in KIN:
            a = [kin_values(sets['real_train'], e, k).mean() for e in EMOTIONS]
            b = [kin_values(sets[name], e, k).mean() for e in EMOTIONS]
            rhos.append(spearmanr(a, b).correlation)
        lines.append(f'| {name} | {np.mean(w1[name]):.2f} | {np.median(w1[name]):.2f} | {np.max(w1[name]):.2f} | {np.mean(rhos):.2f} |')
    for k in KIN:
        lines += ['', f'**{k}** — mean per emotion (normalised W1 to real train in brackets)', '',
                  '| set | ' + ' | '.join(EMOTIONS) + ' |', '|---' * (len(EMOTIONS) + 1) + '|']
        lines.append('| real_train | ' + ' | '.join(f'{kin_values(sets["real_train"], e, k).mean():.3g}' for e in EMOTIONS) + ' |')
        for name in ['real_test'] + synth_names:
            cells = []
            for e in EMOTIONS:
                r = next(r for r in kin_rows if r['emotion'] == e and r['feature'] == k and r['set'] == name)
                cells.append(f"{r['mean']:.3g} ({r['w1_norm']:.2f})")
            lines.append(f'| {name} | ' + ' | '.join(cells) + ' |')

    # 3. memorisation + 4. FID / diversity
    tr = sets['real_train']
    ref_emb = nn_dist(sets['real_test']['emb'], tr['emb'])
    ref_raw = nn_dist(sets['real_test']['raw'], tr['raw'])
    p5_emb, p5_raw = np.percentile(ref_emb, 5), np.percentile(ref_raw, 5)
    mu_tr, cov_tr = calculate_activation_statistics(tr['emb'])
    mu_te, cov_te = calculate_activation_statistics(sets['real_test']['emb'])
    nn_rows = []
    lines += ['', '## 3. Memorisation (nearest neighbour in real train)', '',
              'Distances to the closest real-train window. Reference: real test -> train (unseen subjects). '
              '"< p5(test)" = fraction of windows closer to train than 95% of the real test windows.', '',
              '| set | emb NN median | emb NN p5 | emb < p5(test) | raw NN median (m) | raw NN p5 | raw < p5(test) |',
              '|---|---|---|---|---|---|---|']
    for name in ['real_test'] + synth_names:
        de = ref_emb if name == 'real_test' else nn_dist(sets[name]['emb'], tr['emb'])
        dr = ref_raw if name == 'real_test' else nn_dist(sets[name]['raw'], tr['raw'])
        dr_m = dr / np.sqrt(64 * 22)          # -> RMS per joint-frame, metres
        nn_rows += [dict(set=name, id=i, emb_nn=a, raw_nn_rms=b) for i, a, b in zip(sets[name]['ids'], de, dr_m)]
        lines.append(f'| {name} | {med(de):.2f} | {np.percentile(de, 5):.2f} | {np.mean(de < p5_emb):.3f} | '
                     f'{med(dr_m):.3f} | {np.percentile(dr_m, 5):.3f} | {np.mean(dr < p5_raw):.3f} |')
    with open(os.path.join(out_dir, 'nn.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(nn_rows[0])); w.writeheader(); w.writerows(nn_rows)

    lines += ['', '## 4. FID / diversity (T2M evaluator space, trained on HumanML3D)', '',
              '| set | FID vs real train | FID vs real test | diversity |', '|---|---|---|---|']
    for name, s in sets.items():
        mu, cov = calculate_activation_statistics(s['emb'])
        div = calculate_diversity(s['emb'], min(300, len(s['emb']) - 1))
        f_tr = calculate_frechet_distance(mu, cov, mu_tr, cov_tr)
        f_te = calculate_frechet_distance(mu, cov, mu_te, cov_te)
        lines.append(f'| {name} | {f_tr:.3f} | {f_te:.3f} | {div:.3f} |')

    with open(os.path.join(out_dir, 'report.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))

    # 5. renders: same ids across all synthetic sets
    jobs = []
    for emo_code in ['A', 'D', 'F', 'H', 'N', 'SA', 'SU']:
        for i in range(args.render_per_emotion):
            sid = f'{emo_code}_{i:03d}'
            for name, d in synth_dirs.items():
                m = synth_meta[name][sid]
                jobs.append((os.path.join(d, 'new_joints', sid + '.npy'),
                             os.path.join(out_dir, 'renders', f'{sid}_{name}.mp4'),
                             f"{name}: {m['prompt']}"))
    with Pool(args.workers) as pool:
        pool.map(_render, jobs)
    print(f'rendered {len(jobs)} videos to {out_dir}/renders')


if __name__ == '__main__':
    main()
