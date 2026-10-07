"""Text-to-motion generation with MDM + LoRA adapter (or the plain pretrained MDM).

    python -m lora.generate --lora save/lora/sanity_r8_a8_full/lora_best.pt --scales 1.5 2.5 4 7.5 --n_per_emotion 50
    python -m lora.generate --lora none --scales 2.5 --out save/lora/sanity_r8_a8_full/samples_base   # baseline

Prompts (--prompt_source):
  - generic (Phase 4 default): the generic templates of each emotion, cycled; lengths drawn from real
    training windows of the same emotion;
  - train_captions: for each sample a random real training window of the same emotion is drawn, and one of
    its captions (generic or scenario) and its length are used -> prompts/lengths follow the training data.
Every emotion has the same number of samples (class-balanced). Draws use one RNG per emotion, so the job
list is a prefix-stable sequence: samples [0, N) are identical whatever --end is. Noise is seeded with
seed + start, so pools generated as [0, N) and [N, 2N) together equal a [0, 2N) pool (nested ratios).
Sampling as sample.generate: ClassifierFreeSampleModel + p_sample_loop; --steps respaces the 1000-step chain.
Speed options (defaults = original behaviour):
  --sampler ddim   DDIM (diffusion.ddim_sample_loop, --eta 0 = deterministic) on the respaced chain;
  --fp16           autocast fp16 for the transformer forward passes only (diffusion arithmetic stays fp32);
  --bucket         jobs sorted by length and each batch generated at its longest length instead of 196 frames.
                   NOT equivalent to generating at 196 and cropping: humanml_trans_enc_512 has mask_frames=False
                   (padded frames are attended to) and was trained on 196-frame padded sequences -> validate.
Output per scale, in the dataset layout (reusable by scripts.kdaee.validate / lora.eval_synth):
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
from lora.mdm_utils import load_pretrained_mdm, respaced_diffusion
from model.cfg_sampler import ClassifierFreeSampleModel
from scripts.kdaee.prompts import GENERIC

EMOTIONS = {'A': 'Angry', 'D': 'Disgust', 'F': 'Fearful', 'H': 'Happy', 'N': 'Neutral', 'SA': 'Sad', 'SU': 'Surprise'}
META_FIELDS = ['id', 'actor', 'emotion', 'emotion_code', 'scenario_code', 'prompt', 'n_frames', 'scale', 'steps',
               'sampler', 'prompt_source', 'source_window', 'fold', 'lora', 'lora_fold', 'lora_train_actors', 'seed']


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model_path', default='save/humanml_trans_enc_512/model000200000.pt')
    p.add_argument('--lora', default='save/lora/sanity_r8_a8_full/lora_best.pt', help="'none' = pretrained MDM")
    p.add_argument('--data', default='data/kdaee_hml3d')
    p.add_argument('--fold', default='sanity')
    p.add_argument('--out', default='', help='default: <lora dir>/samples')
    p.add_argument('--scales', type=float, nargs='+', default=[1.5, 2.5, 4.0, 7.5])
    p.add_argument('--n_per_emotion', type=int, default=50, help='= --end when --end is not given')
    p.add_argument('--start', type=int, default=0, help='first sample index per emotion')
    p.add_argument('--end', type=int, default=None, help='last sample index (exclusive) per emotion')
    p.add_argument('--prompt_source', default='generic', choices=['generic', 'train_captions'])
    p.add_argument('--steps', type=int, default=1000, help='sampling steps (respacing of the 1000-step chain)')
    p.add_argument('--sampler', default='ddpm', choices=['ddpm', 'ddim'])
    p.add_argument('--eta', type=float, default=0.0, help='DDIM eta (0 = deterministic)')
    p.add_argument('--fp16', action='store_true', help='autocast fp16 for the model forward passes')
    p.add_argument('--bucket', action='store_true', help='generate each batch at its longest length')
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--hml_root', default='dataset/HumanML3D')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()


def build_jobs(args, end):
    meta = list(csv.DictReader(open(os.path.join(args.data, 'meta.csv'))))
    train_ids = set(open(os.path.join(args.data, 'splits', args.fold, 'train.txt')).read().split())
    jobs = []
    for k, (code, emotion) in enumerate(EMOTIONS.items()):
        rng = random.Random(args.seed * 1000 + k)            # one stream per emotion -> prefix-stable
        wins = [m for m in meta if m['id'] in train_ids and m['emotion'] == emotion]
        for i in range(end):
            w = rng.choice(wins)
            if args.prompt_source == 'generic':
                prompt = GENERIC[code][i % len(GENERIC[code])]
            else:
                lines = open(os.path.join(args.data, 'texts', w['id'] + '.txt')).read().strip().split('\n')
                prompt = rng.choice(lines).split('#')[0]
            n = min(196, (int(w['n_frames']) // 4) * 4)      # multiple of 4 as in MDM training crops
            jobs.append(dict(id=f'{code}_{i:04d}', emotion=emotion, emotion_code=code, prompt=prompt,
                             n_frames=n, source_window=w['id'], index=i))
    return [j for j in jobs if j['index'] >= args.start]


class Fp16Model(torch.nn.Module):
    """Runs the wrapped (CFG) model under fp16 autocast and returns fp32, so that the diffusion arithmetic is
    unchanged."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, *a, **kw):
        with torch.cuda.amp.autocast():
            return self.model(*a, **kw).float()


def main():
    args = parse_args()
    end = args.end if args.end is not None else args.n_per_emotion
    device = torch.device(args.device)
    out_root = args.out or os.path.join(os.path.dirname(args.lora) if args.lora != 'none' else 'save/lora/base_mdm',
                                        'samples')
    model, _, mdm_args = load_pretrained_mdm(args.model_path, device)
    diffusion = respaced_diffusion(mdm_args, args.steps)
    lora_fold, lora_actors = '', ''
    if args.lora != 'none':
        ck = torch.load(args.lora, map_location='cpu')
        c = ck['lora_config']
        inject_lora(model, r=c['r'], alpha=c['alpha'], targets=c['targets'], dropout=c['dropout'], layers=c['layers'])
        load_lora_state_dict(model, ck['lora_state_dict'])
        lora_fold, lora_actors = ck['fold'], ' '.join(ck.get('train_actors', []))
    model.eval()
    sampler = ClassifierFreeSampleModel(model)
    if args.fp16:
        sampler = Fp16Model(sampler)
    sampler_tag = f'{args.sampler}{args.steps}' + (f'_eta{args.eta:g}' if args.sampler == 'ddim' else '') + \
        ('_fp16' if args.fp16 else '') + ('_bucket' if args.bucket else '')
    mean = torch.from_numpy(np.load(os.path.join(args.hml_root, 'Mean.npy'))).float()
    std = torch.from_numpy(np.load(os.path.join(args.hml_root, 'Std.npy'))).float()
    jobs = build_jobs(args, end)
    if args.bucket:
        jobs = sorted(jobs, key=lambda j: j['n_frames'])     # stable: same order for every scale / rerun

    for scale in args.scales:
        out = os.path.join(out_root, f'scale_{scale:g}')
        os.makedirs(os.path.join(out, 'new_joint_vecs'), exist_ok=True)
        os.makedirs(os.path.join(out, 'new_joints'), exist_ok=True)
        torch.manual_seed(args.seed + args.start)          # same noise for every scale; slices are reproducible
        for s in range(0, len(jobs), args.batch_size):
            batch = jobs[s:s + args.batch_size]
            T = max(j['n_frames'] for j in batch) if args.bucket else 196
            _, kw = collate([{'inp': torch.zeros(T), 'tokens': None, 'lengths': j['n_frames'],
                              'text': j['prompt']} for j in batch])
            kw['y'] = {k: v.to(device) if torch.is_tensor(v) else v for k, v in kw['y'].items()}
            kw['y']['scale'] = torch.full((len(batch),), scale, device=device)
            with torch.no_grad():
                kw['y']['text_embed'] = model.encode_text(kw['y']['text'])
                shape = (len(batch), model.njoints, model.nfeats, T)
                if args.sampler == 'ddim':
                    sample = diffusion.ddim_sample_loop(sampler, shape, clip_denoised=False, model_kwargs=kw,
                                                        skip_timesteps=0, init_image=None, progress=False,
                                                        eta=args.eta)
                else:
                    sample = diffusion.p_sample_loop(sampler, shape, clip_denoised=False, model_kwargs=kw,
                                                     skip_timesteps=0, init_image=None, progress=False,
                                                     dump_steps=None, noise=None, const_noise=False)
            feats = sample.cpu()[:, :, 0].permute(0, 2, 1) * std + mean          # [bs, T, 263], unnormalised
            for j, f in zip(batch, feats):
                f = f[:j['n_frames']]
                joints = recover_from_ric(f[None], 22)[0]
                np.save(os.path.join(out, 'new_joint_vecs', j['id'] + '.npy'), f.numpy().astype(np.float32))
                np.save(os.path.join(out, 'new_joints', j['id'] + '.npy'), joints.numpy().astype(np.float32))
            print(f'scale {scale:g}: {min(s + args.batch_size, len(jobs))}/{len(jobs)}', flush=True)
        meta_path = os.path.join(out, 'meta.csv')
        rows = list(csv.DictReader(open(meta_path))) if os.path.exists(meta_path) else []
        new_ids = {j['id'] for j in jobs}
        rows = [r for r in rows if r['id'] not in new_ids]
        rows += [dict(id=j['id'], actor='synthetic', emotion=j['emotion'], emotion_code=j['emotion_code'],
                      scenario_code=f'cfg{scale:g}', prompt=j['prompt'], n_frames=j['n_frames'], scale=scale,
                      steps=args.steps, sampler=sampler_tag, prompt_source=args.prompt_source, source_window=j['source_window'],
                      fold=args.fold, lora=args.lora, lora_fold=lora_fold, lora_train_actors=lora_actors,
                      seed=args.seed) for j in jobs]
        rows.sort(key=lambda r: r['id'])
        with open(meta_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=META_FIELDS, extrasaction='ignore')
            w.writeheader()
            w.writerows(rows)


if __name__ == '__main__':
    main()
