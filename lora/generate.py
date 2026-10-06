"""Text-to-motion generation with MDM + LoRA adapter (or the plain pretrained MDM).

    python -m lora.generate --lora save/lora/sanity_r8_a8_full/lora_best.pt --scales 1.5 2.5 4 7.5 --n_per_emotion 50
    python -m lora.generate --lora none --scales 2.5 --out save/lora/sanity_r8_a8_full/samples_base   # baseline

Prompts: the generic templates of each emotion (scripts/kdaee/prompts.py), cycled. Lengths: drawn from the
real training windows of the same emotion in the fold. Same seed for every guidance scale, so sample i is
generated from the same noise at every scale (paired comparison).
Sampling as sample.generate: ClassifierFreeSampleModel, p_sample_loop, 1000 DDPM steps.
Output per scale, in the dataset layout (reusable by scripts.kdaee.validate):
    <out>/scale_<s>/{new_joint_vecs,new_joints}/<id>.npy (unnormalised), meta.csv
"""
import argparse
import csv
import os
import random

import numpy as np
import torch

from data_loaders.humanml.scripts.motion_process import recover_from_ric
from data_loaders.tensors import collate
from lora.inject import inject_lora, load_lora_state_dict
from lora.mdm_utils import load_pretrained_mdm
from model.cfg_sampler import ClassifierFreeSampleModel
from scripts.kdaee.prompts import GENERIC

EMOTIONS = {'A': 'Angry', 'D': 'Disgust', 'F': 'Fearful', 'H': 'Happy', 'N': 'Neutral', 'SA': 'Sad', 'SU': 'Surprise'}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model_path', default='save/humanml_trans_enc_512/model000200000.pt')
    p.add_argument('--lora', default='save/lora/sanity_r8_a8_full/lora_best.pt', help="'none' = pretrained MDM")
    p.add_argument('--data', default='data/kdaee_hml3d')
    p.add_argument('--fold', default='sanity')
    p.add_argument('--out', default='', help='default: <lora dir>/samples')
    p.add_argument('--scales', type=float, nargs='+', default=[1.5, 2.5, 4.0, 7.5])
    p.add_argument('--n_per_emotion', type=int, default=50)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--hml_root', default='dataset/HumanML3D')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()


def build_jobs(args):
    meta = list(csv.DictReader(open(os.path.join(args.data, 'meta.csv'))))
    train_ids = set(l.strip() for l in open(os.path.join(args.data, 'splits', args.fold, 'train.txt')) if l.strip())
    rng = random.Random(args.seed)
    jobs = []
    for code, emotion in EMOTIONS.items():
        lengths = [int(m['n_frames']) for m in meta if m['id'] in train_ids and m['emotion'] == emotion]
        for i in range(args.n_per_emotion):
            n = min(196, (rng.choice(lengths) // 4) * 4)     # multiple of 4 as in MDM training crops
            jobs.append(dict(id=f'{code}_{i:03d}', emotion=emotion, emotion_code=code,
                             prompt=GENERIC[code][i % len(GENERIC[code])], n_frames=n))
    return jobs


def main():
    args = parse_args()
    device = torch.device(args.device)
    out_root = args.out or os.path.join(os.path.dirname(args.lora) if args.lora != 'none' else 'save/lora/base_mdm',
                                        'samples')
    model, diffusion, _ = load_pretrained_mdm(args.model_path, device)
    if args.lora != 'none':
        ck = torch.load(args.lora, map_location='cpu')
        c = ck['lora_config']
        inject_lora(model, r=c['r'], alpha=c['alpha'], targets=c['targets'], dropout=c['dropout'], layers=c['layers'])
        load_lora_state_dict(model, ck['lora_state_dict'])
    model.eval()
    sampler = ClassifierFreeSampleModel(model)
    mean = torch.from_numpy(np.load(os.path.join(args.hml_root, 'Mean.npy'))).float()
    std = torch.from_numpy(np.load(os.path.join(args.hml_root, 'Std.npy'))).float()
    jobs = build_jobs(args)

    for scale in args.scales:
        out = os.path.join(out_root, f'scale_{scale:g}')
        os.makedirs(os.path.join(out, 'new_joint_vecs'), exist_ok=True)
        os.makedirs(os.path.join(out, 'new_joints'), exist_ok=True)
        torch.manual_seed(args.seed)                       # same noise for every scale
        for s in range(0, len(jobs), args.batch_size):
            batch = jobs[s:s + args.batch_size]
            _, kw = collate([{'inp': torch.zeros(196), 'tokens': None, 'lengths': j['n_frames'],
                              'text': j['prompt']} for j in batch])
            kw['y'] = {k: v.to(device) if torch.is_tensor(v) else v for k, v in kw['y'].items()}
            kw['y']['scale'] = torch.full((len(batch),), scale, device=device)
            with torch.no_grad():
                kw['y']['text_embed'] = model.encode_text(kw['y']['text'])
                sample = diffusion.p_sample_loop(sampler, (len(batch), model.njoints, model.nfeats, 196),
                                                 clip_denoised=False, model_kwargs=kw, skip_timesteps=0,
                                                 init_image=None, progress=False, dump_steps=None,
                                                 noise=None, const_noise=False)
            feats = sample.cpu()[:, :, 0].permute(0, 2, 1) * std + mean          # [bs, 196, 263], unnormalised
            for j, f in zip(batch, feats):
                f = f[:j['n_frames']]
                joints = recover_from_ric(f[None], 22)[0]
                np.save(os.path.join(out, 'new_joint_vecs', j['id'] + '.npy'), f.numpy().astype(np.float32))
                np.save(os.path.join(out, 'new_joints', j['id'] + '.npy'), joints.numpy().astype(np.float32))
            print(f'scale {scale:g}: {min(s + args.batch_size, len(jobs))}/{len(jobs)}', flush=True)
        with open(os.path.join(out, 'meta.csv'), 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['id', 'actor', 'emotion', 'emotion_code', 'scenario_code', 'prompt',
                                              'n_frames', 'scale', 'lora', 'seed'])
            w.writeheader()
            for j in jobs:
                w.writerow(dict(j, actor='synthetic', scenario_code=f'cfg{scale:g}', scale=scale,
                                lora=args.lora, seed=args.seed))


if __name__ == '__main__':
    main()
