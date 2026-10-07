"""Data for the downstream emotion classifier: real KDAEE windows and synthetic samples, both in the converted
HumanML3D format and turned into 22-joint positions with the same function (recover_from_ric).

Also: the simple kinematic augmentation (applied to joint sequences, before feature extraction) and the
anti-leakage assertions of the protocol.
"""
import csv
import json
import os

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d

from data_loaders.humanml.scripts.motion_process import recover_from_ric
from data_loaders.humanml.utils.paramUtil import t2m_kinematic_chain, t2m_raw_offsets

EMOTIONS = ['Angry', 'Disgust', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprise']
LABEL = {e: i for i, e in enumerate(EMOTIONS)}
LR_PAIRS = [(1, 2), (4, 5), (7, 8), (10, 11), (13, 14), (16, 17), (18, 19), (20, 21)]   # HumanML3D left/right
PARENTS = np.full(22, -1)
for _chain in t2m_kinematic_chain:
    for _p, _c in zip(_chain[:-1], _chain[1:]):
        PARENTS[_c] = _p
FEET = [7, 8, 10, 11]


def joints_from_features(feat):
    """The single function used to obtain joint positions for real and synthetic windows."""
    return recover_from_ric(torch.from_numpy(np.asarray(feat, np.float32))[None], 22)[0].numpy()


def _load(root, rows, kind):
    out = []
    for r in rows:
        feat = np.load(os.path.join(root, 'new_joint_vecs', r['id'] + '.npy'))
        out.append(dict(id=r['id'], kind=kind, emotion=r['emotion'], label=LABEL[r['emotion']],
                        actor=r.get('actor', 'synthetic'), source=r.get('source', r['id']),
                        joints=joints_from_features(feat), meta=r))
    return out


def load_fold(data_root, fold):
    meta = {m['id']: m for m in csv.DictReader(open(os.path.join(data_root, 'meta.csv')))}
    spec = json.load(open(os.path.join(data_root, 'splits', 'folds.json')))[fold]
    sets = {}
    for split in ('train', 'val', 'test'):
        ids = open(os.path.join(data_root, 'splits', fold, split + '.txt')).read().split()
        sets[split] = _load(data_root, [meta[i] for i in ids], 'real')
    return spec, sets


def load_synthetic(pool_dir):
    rows = list(csv.DictReader(open(os.path.join(pool_dir, 'meta.csv'))))
    return _load(pool_dir, rows, 'synthetic')


def balanced_subset(pool, n_per_class, rng):
    """n_per_class synthetic windows per emotion, drawn without replacement."""
    out = []
    for lab in range(len(EMOTIONS)):
        cand = [w for w in pool if w['label'] == lab]
        assert len(cand) >= n_per_class, f'pool has {len(cand)} {EMOTIONS[lab]} samples, need {n_per_class}'
        out += [cand[i] for i in rng.choice(len(cand), n_per_class, replace=False)]
    return out


# ---------------------------------------------------------------- anti-leakage

def check_split(spec, sets):
    """Subject-wise split: no actor and no original sequence (all its windows) in two splits."""
    actors = {s: {w['actor'] for w in ws} for s, ws in sets.items()}
    sources = {s: {w['source'] for w in ws} for s, ws in sets.items()}
    for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test')):
        assert not actors[a] & actors[b], f'actor overlap {a}/{b}: {actors[a] & actors[b]}'
        assert not sources[a] & sources[b], f'windows of the same sequence in {a} and {b}'
    for s in ('train', 'val', 'test'):
        assert actors[s] == set(spec[s]), f'{s} actors differ from folds.json'
        assert all(w['kind'] == 'real' for w in sets[s]), f'non-real window in the {s} split'


def check_lora(fold, spec, lora_ckpt):
    """The fold's LoRA was trained only on the fold's training actors (early stopping on its validation actor)."""
    ck = torch.load(lora_ckpt, map_location='cpu')
    assert ck['fold'] == fold, f"LoRA trained on fold {ck['fold']}, used in {fold}"
    assert set(ck['train_actors']) == set(spec['train']), 'LoRA training actors != fold training actors'
    assert set(ck['val_actors']) == set(spec['val']), 'LoRA validation actors != fold validation actors'
    assert not set(ck['train_actors']) & (set(spec['test']) | set(spec['val'])), 'LoRA saw val/test subjects'


def check_pool(name, pool, fold, spec, train_ids, lora_ckpt=None):
    """Synthetic pool of this fold: prompts/lengths from training windows only; 'lora' pools come from this
    fold's LoRA, 'base' pools from the pretrained MDM (lora == 'none')."""
    for w in pool:
        m = w['meta']
        assert w['kind'] == 'synthetic' and m['actor'] == 'synthetic'
        assert m['fold'] == fold, f'{name}: sample generated for fold {m["fold"]}'
        assert m['source_window'] in train_ids, f'{name}: prompt/length taken from a non-train window'
        if name.startswith('lora'):
            assert os.path.abspath(m['lora']) == os.path.abspath(lora_ckpt), f'{name}: sample from another LoRA'
            assert m['lora_fold'] == fold and set(m['lora_train_actors'].split()) == set(spec['train'])
        else:
            assert m['lora'] == 'none', f'{name}: base pool contains LoRA samples'


def check_training_sets(sets, train, val, test):
    """Synthetic / augmented windows only in the training set; val and test are exactly the real split."""
    assert [w['id'] for w in val] == [w['id'] for w in sets['val']]
    assert [w['id'] for w in test] == [w['id'] for w in sets['test']]
    held = {w['id'] for w in sets['val'] + sets['test']}
    held_src = {w['source'] for w in sets['val'] + sets['test']}
    for w in train:
        assert w['kind'] in ('real', 'augmented', 'synthetic')
        assert w['id'] not in held and w['source'] not in held_src, f"{w['id']}: val/test material in train"
        if w['kind'] == 'synthetic':
            assert w['meta']['source_window'] not in held, f"{w['id']}: prompt from a val/test window"


# ---------------------------------------------------------------- augmentation

def _rodrigues(axis_angle):
    """(..., 3) axis-angle vectors -> (..., 3, 3) rotation matrices."""
    th = np.linalg.norm(axis_angle, axis=-1, keepdims=True)
    k = axis_angle / np.maximum(th, 1e-12)
    K = np.zeros(axis_angle.shape[:-1] + (3, 3))
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 2] = -k[..., 2], k[..., 1], -k[..., 0]
    K = K - np.swapaxes(K, -1, -2)
    th = th[..., None]
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def _align(u, d):
    """Minimal rotation taking the unit vector u (3,) onto the unit vectors d (T, 3) -> (T, 3, 3)."""
    axis = np.cross(u, d)
    s = np.linalg.norm(axis, axis=-1, keepdims=True)
    c = (d @ u)[:, None]
    aa = axis / np.maximum(s, 1e-12) * np.arctan2(s, c)
    flip = (s[:, 0] < 1e-8) & (c[:, 0] < 0)                       # u = -d: 180 degrees about any normal axis
    if flip.any():
        n = np.cross(u, [1., 0, 0]) if abs(u[0]) < 0.9 else np.cross(u, [0, 1., 0])
        aa[flip] = np.pi * n / np.linalg.norm(n)
    return _rodrigues(aa)


def fk_augment(joints, rot_noise, bone_scale):
    """Inverse kinematics -> perturbed local joint rotations and bone lengths -> forward kinematics.

    Rotations: global rotation of each bone = minimal rotation from its rest direction (HumanML3D raw offsets)
    onto the observed direction; local = parent^T @ global. The noise multiplies the local rotations, so it
    propagates to the whole subtree. rot_noise: (T, 22, 3) axis-angle (root entry unused); bone_scale: (22,)
    length factor of the bone ending at each joint. Root trajectory unchanged. With zero noise and unit scale
    the input positions are reproduced exactly.
    """
    j = np.asarray(joints, np.float64)
    T = len(j)
    out = np.zeros_like(j)
    out[:, 0] = j[:, 0]
    G, Gn = {0: np.broadcast_to(np.eye(3), (T, 3, 3))}, {0: np.broadcast_to(np.eye(3), (T, 3, 3))}
    for k in range(1, 22):
        p = PARENTS[k]
        v = j[:, k] - j[:, p]
        length = np.linalg.norm(v, axis=-1, keepdims=True)
        u = t2m_raw_offsets[k] / np.linalg.norm(t2m_raw_offsets[k])
        G[k] = _align(u, v / np.maximum(length, 1e-12))
        local = np.swapaxes(G[p], -1, -2) @ G[k]
        Gn[k] = Gn[p] @ local @ _rodrigues(rot_noise[:, k])
        out[:, k] = out[:, p] + (Gn[k] @ u) * length * bone_scale[k]
    return out


def augment(joints, rng, crop=(0.7, 1.0), min_len=40, rot_deg=1.0, rot_smooth=5.0, bone=0.08):
    """Simple augmentation of a (T, 22, 3) sequence (protocol condition 2):
    - random temporal crop (70-100 % of the window, at least `min_len` frames);
    - small Gaussian noise on the local joint rotations: axis-angle noise, low-pass filtered in time
      (gaussian, sigma `rot_smooth` frames, as i.i.d. per-frame noise would dominate the jerk features) and
      rescaled to an RMS angle of `rot_deg` degrees per joint;
    - bone-length variation: one factor U(1 - bone, 1 + bone) per bone, shared by the left/right homologous
      bones; positions recomposed by forward kinematics and re-grounded (same lowest foot height).
    """
    T = len(joints)
    n = max(min(T, min_len), int(round(T * rng.uniform(*crop))))
    s = rng.integers(0, T - n + 1)
    j = np.asarray(joints[s:s + n], np.float64)
    noise = gaussian_filter1d(rng.standard_normal((n, 22, 3)), sigma=rot_smooth, axis=0)
    noise *= np.radians(rot_deg) / (np.sqrt((noise ** 2).sum(-1).mean()) + 1e-12)
    scale = rng.uniform(1 - bone, 1 + bone, 22)
    for a, b in LR_PAIRS:
        scale[b] = scale[a]
    out = fk_augment(j, noise, scale)
    out[..., 1] += j[:, FEET, 1].min() - out[:, FEET, 1].min()
    return out.astype(np.float32)


def augmented_set(windows, n_per_class, rng):
    """n_per_class augmented windows per emotion (class-balanced, 1:1 with the real training set when
    n_per_class = len(real) / 7): source windows drawn without replacement while possible."""
    out = []
    for lab in range(len(EMOTIONS)):
        cand = [w for w in windows if w['label'] == lab]
        idx = np.concatenate([rng.permutation(len(cand)) for _ in range(-(-n_per_class // len(cand)))])[:n_per_class]
        for i, k in enumerate(idx):
            w = cand[k]
            out.append(dict(w, id=f"{w['id']}_aug{i}", kind='augmented', joints=augment(w['joints'], rng)))
    return out
