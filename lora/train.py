"""LoRA fine-tuning of the pretrained MDM on KDAEE (one adapter per fold).

    python -m lora.train --fold sanity [--r 8 --alpha 8 --lr 2e-4 --max_steps 20000 --eval_every 250 --patience 8]

Same objective as MDM pretraining (x0-prediction MSE on HumanML3D-normalised 263-dim features, uniform t,
masked padding, 10% text dropout for classifier-free guidance). Only LoRA params are trained.
Early stopping on a deterministic validation loss (validation subjects, fixed crops/captions/t/noise).
Saves only the LoRA weights: <save_dir>/lora_best.pt, plus log.csv, config.json, loss_curve.png.
"""
import argparse
import csv
import json
import os
import random
import time

import numpy as np
import torch

from lora.inject import count_parameters, inject_lora, lora_parameters, lora_state_dict
from lora.mdm_utils import kdaee_dataset, kdaee_loader, load_pretrained_mdm
from data_loaders.tensors import t2m_collate


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model_path', default='save/humanml_trans_enc_512/model000200000.pt')
    p.add_argument('--data', default='data/kdaee_hml3d')
    p.add_argument('--fold', default='sanity')
    p.add_argument('--save_dir', default='', help='default: save/lora/<fold>_r<r>_a<alpha>')
    p.add_argument('--r', type=int, default=8)
    p.add_argument('--alpha', type=float, default=8)
    p.add_argument('--targets', nargs='+', default=['q', 'k', 'v', 'o', 'ffn1', 'ffn2'])
    p.add_argument('--lora_dropout', type=float, default=0.0)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--weight_decay', type=float, default=0.0)
    p.add_argument('--warmup_steps', type=int, default=200)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--max_steps', type=int, default=20000)
    p.add_argument('--eval_every', type=int, default=250)
    p.add_argument('--patience', type=int, default=8, help='evaluations without improvement before stopping')
    p.add_argument('--val_draws', type=int, default=8, help='(crop, caption, t, noise) draws per val window')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class TextEmbedCache:
    """CLIP is frozen and KDAEE has ~100 distinct captions: encode each once."""

    def __init__(self, model):
        self.model, self.cache = model, {}

    def __call__(self, texts):
        new = [t for t in set(texts) if t not in self.cache]
        if new:
            with torch.no_grad():
                emb = self.model.encode_text(new)            # [1, n, 512]
            for i, t in enumerate(new):
                self.cache[t] = emb[:, i]
        return torch.stack([self.cache[t] for t in texts], 1)  # [1, bs, 512]


def to_device(motion, cond, device, text_cache):
    cond['y'] = {k: v.to(device) if torch.is_tensor(v) else v for k, v in cond['y'].items()}
    cond['y']['text_embed'] = text_cache(cond['y']['text'])
    return motion.to(device), cond


def build_val_set(data, split_file, draws, seed, num_timesteps):
    """Fixed validation batches: `draws` random crops/captions per window, fixed t (stratified) and noise."""
    ds = kdaee_dataset(data, split_file)
    rng_state = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    seed_all(seed)
    items = [ds[i] for _ in range(draws) for i in range(len(ds))]
    n = len(items)
    t = (torch.arange(n) * num_timesteps // n)[torch.randperm(n)]               # stratified over [0, T)
    batches = []
    for s in range(0, n, 64):
        motion, cond = t2m_collate(items[s:s + 64], target_batch_size=len(items[s:s + 64]))
        batches.append((motion, cond, t[s:s + 64], torch.randn_like(motion)))
    random.setstate(rng_state[0]); np.random.set_state(rng_state[1]); torch.set_rng_state(rng_state[2])
    return batches, len(ds)


@torch.no_grad()
def evaluate(model, diffusion, val_batches, device, text_cache):
    model.eval()
    tot, cnt = 0.0, 0
    for motion, cond, t, noise in val_batches:
        cond = {'y': dict(cond['y'])}
        motion, cond = to_device(motion, cond, device, text_cache)
        losses = diffusion.training_losses(model, motion, t.to(device), model_kwargs=cond, noise=noise.to(device))
        tot += losses['loss'].sum().item()
        cnt += len(t)
    model.train()
    return tot / cnt


def plot_curve(log_rows, path, base_val):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    steps = [r['step'] for r in log_rows]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(steps, [r['train_loss'] for r in log_rows], label='train (mean over interval)')
    ax.plot(steps, [r['val_loss'] for r in log_rows], label='val (fixed draws)')
    ax.axhline(base_val, ls='--', c='gray', label='val, pretrained MDM (LoRA = 0)')
    ax.set_xlabel('step')
    ax.set_ylabel('diffusion loss (x0 MSE)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    args = parse_args()
    seed_all(args.seed)
    device = torch.device(args.device)
    save_dir = args.save_dir or os.path.join('save', 'lora', f'{args.fold}_r{args.r}_a{args.alpha:g}')
    os.makedirs(save_dir, exist_ok=True)
    split_dir = os.path.join(args.data, 'splits', args.fold)

    model, diffusion, mdm_args = load_pretrained_mdm(args.model_path, device)
    lora_cfg = inject_lora(model, r=args.r, alpha=args.alpha, targets=args.targets, dropout=args.lora_dropout)
    counts = count_parameters(model)
    print('parameters:', counts)
    text_cache = TextEmbedCache(model)

    train_loader = kdaee_loader(args.data, os.path.join(split_dir, 'train.txt'), args.batch_size)
    # recorded in the checkpoint so downstream code can assert the adapter never saw validation/test subjects
    meta = {m['id']: m for m in csv.DictReader(open(os.path.join(args.data, 'meta.csv')))}
    split_actors = {s: sorted({meta[i]['actor'] for i in open(os.path.join(split_dir, s + '.txt')).read().split()})
                    for s in ('train', 'val', 'test')}
    assert not set(split_actors['train']) & (set(split_actors['val']) | set(split_actors['test']))
    val_batches, n_val = build_val_set(args.data, os.path.join(split_dir, 'val.txt'), args.val_draws,
                                       args.seed + 1, diffusion.num_timesteps)
    print(f'train windows: {len(train_loader.dataset)}, val windows: {n_val} x {args.val_draws} draws')

    opt = torch.optim.AdamW(lora_parameters(model), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, args.warmup_steps)))

    base_val = evaluate(model, diffusion, val_batches, device, text_cache)
    print(f'step 0: val loss (pretrained MDM, LoRA = 0) {base_val:.5f}')
    config = dict(vars(args), lora=lora_cfg, params=counts, base_val_loss=base_val, save_dir=save_dir,
                  split_actors=split_actors,
                  mdm_args={k: v for k, v in vars(mdm_args).items() if isinstance(v, (int, float, str, bool))})
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=1)

    log_rows, best, bad_evals, step, t0 = [], float('inf'), 0, 0, time.time()
    run_loss, run_n = 0.0, 0
    model.train()
    done = False
    while not done:
        for motion, cond in train_loader:
            motion, cond = to_device(motion, cond, device, text_cache)
            t = torch.randint(0, diffusion.num_timesteps, (motion.shape[0],), device=device)
            loss = diffusion.training_losses(model, motion, t, model_kwargs=cond)['loss'].mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            run_loss += loss.item()
            run_n += 1

            if step % args.eval_every == 0:
                val = evaluate(model, diffusion, val_batches, device, text_cache)
                row = dict(step=step, train_loss=run_loss / run_n, val_loss=val, lr=sched.get_last_lr()[0],
                           minutes=(time.time() - t0) / 60)
                log_rows.append(row)
                run_loss, run_n = 0.0, 0
                improved = val < best
                if improved:
                    best, bad_evals = val, 0
                    torch.save({'lora_state_dict': lora_state_dict(model), 'lora_config': lora_cfg,
                                'base_model_path': args.model_path, 'fold': args.fold, 'step': step,
                                'val_loss': val, 'train_actors': split_actors['train'],
                                'val_actors': split_actors['val']}, os.path.join(save_dir, 'lora_best.pt'))
                else:
                    bad_evals += 1
                print(f"step {step}: train {row['train_loss']:.5f} val {val:.5f} "
                      f"{'*' if improved else f'({bad_evals}/{args.patience})'} [{row['minutes']:.1f} min]")
                with open(os.path.join(save_dir, 'log.csv'), 'w', newline='') as f:
                    w = csv.DictWriter(f, fieldnames=list(row))
                    w.writeheader()
                    w.writerows(log_rows)
                plot_curve(log_rows, os.path.join(save_dir, 'loss_curve.png'), base_val)
                if bad_evals >= args.patience:
                    print('early stopping')
                    done = True
            if step >= args.max_steps:
                done = True
            if done:
                break

    print(f'best val loss {best:.5f} (pretrained {base_val:.5f}); LoRA weights in {save_dir}/lora_best.pt')


if __name__ == '__main__':
    main()
