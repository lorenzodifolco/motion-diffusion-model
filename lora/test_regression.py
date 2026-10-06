"""Regression tests for LoRA in MDM.

    python -m lora.test_regression [--model_path save/humanml_trans_enc_512/model000200000.pt] [--device cuda]

1. zero LoRA (B = 0, as at init) -> output bit-identical to the original MDM (eval mode, fixed inputs);
2. random LoRA weights: LoRA modules == plain torch modules with merged weights W0 + scale * B A;
3. one optimisation step: gradients only on LoRA params, all original weights (CLIP included) unchanged;
4. parameter counts.
"""
import argparse
import copy

import torch
import torch.nn as nn

from lora.inject import count_parameters, inject_lora, lora_parameters
from lora.layers import LoRALinear, LoRAMultiheadAttention
from lora.mdm_utils import load_pretrained_mdm

TEXTS = ['a person moves angrily.', 'a person walks forward and sits down.',
         'a sad person shows their sadness with their whole body.', '']


def make_batch(model, device, bs=4, n_frames=196, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(bs, model.njoints, model.nfeats, n_frames, generator=g).to(device)
    t = torch.randint(0, 1000, (bs,), generator=g).to(device)
    lengths = torch.tensor([196, 120, 60, 196])[:bs].to(device)
    mask = (torch.arange(n_frames, device=device)[None] < lengths[:, None])[:, None, None]
    y = {'mask': mask, 'lengths': lengths, 'text': TEXTS[:bs]}
    with torch.no_grad():
        y['text_embed'] = model.encode_text(y['text'])
    return x, t, y


def check(name, ok, detail=''):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model_path', default='save/humanml_trans_enc_512/model000200000.pt')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--r', type=int, default=8)
    p.add_argument('--alpha', type=float, default=8)
    args = p.parse_args()
    dev = torch.device(args.device)
    results = []

    model, _, _ = load_pretrained_mdm(args.model_path, dev)
    model.eval()
    x, t, y = make_batch(model, dev)
    with torch.no_grad():
        out_ref = model(x, t, y)
        out_ref_uncond = model(x, t, dict(y, uncond=True))
    base_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    # --- 1. zero LoRA == original
    cfg = inject_lora(model, r=args.r, alpha=args.alpha)
    model.eval()
    with torch.no_grad():
        out_lora = model(x, t, y)
        out_lora_uncond = model(x, t, dict(y, uncond=True))
    results.append(check('zero-LoRA output identical (cond)', torch.equal(out_ref, out_lora),
                         f'max|diff|={(out_ref - out_lora).abs().max().item():.3e}'))
    results.append(check('zero-LoRA output identical (uncond / CFG branch)', torch.equal(out_ref_uncond, out_lora_uncond),
                         f'max|diff|={(out_ref_uncond - out_lora_uncond).abs().max().item():.3e}'))

    # --- 2. random LoRA == merged plain modules (TF32 off: on Ampere+ torch>=1.7 runs fp32 matmuls in TF32,
    #        ~1e-3 relative precision, and the merged/unmerged paths do different matmuls)
    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(1)
    for prm in lora_parameters(model):
        prm.data.normal_(0, 0.02)
    layer = model.seqTransEncoder.layers[0]
    seq = torch.randn(197, 4, model.latent_dim, device=dev)
    kpm = torch.zeros(4, 197, dtype=torch.bool, device=dev)
    kpm[2, 150:] = True
    with torch.no_grad():
        att = layer.self_attn
        ref_att = copy.deepcopy(att.base)
        w_in, w_out = att.merged_weights()
        ref_att.in_proj_weight.data.copy_(w_in)
        ref_att.out_proj.weight.data.copy_(w_out)
        a1, _ = att(seq, seq, seq, key_padding_mask=kpm)
        a2, _ = ref_att(seq, seq, seq, key_padding_mask=kpm)
        lin = layer.linear1
        ref_lin = nn.Linear(lin.base.in_features, lin.base.out_features).to(dev)
        ref_lin.weight.data.copy_(lin.base.weight + lin.adapter.delta())
        ref_lin.bias.data.copy_(lin.base.bias)
        l1, l2 = lin(seq), ref_lin(seq)
        out_rand = model(x, t, y)
    torch.backends.cuda.matmul.allow_tf32 = tf32
    results.append(check('random-LoRA attention == MHA with merged weights', torch.allclose(a1, a2, atol=1e-5),
                         f'max|diff|={(a1 - a2).abs().max().item():.3e}'))
    results.append(check('random-LoRA FFN linear == Linear with merged weights', torch.allclose(l1, l2, atol=1e-5),
                         f'max|diff|={(l1 - l2).abs().max().item():.3e}'))
    results.append(check('random-LoRA changes the model output', not torch.allclose(out_rand, out_ref),
                         f'mean|diff|={(out_rand - out_ref).abs().mean().item():.3e}'))

    # --- 3. only LoRA params train; base weights untouched by an optimiser step
    model.train()
    opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=1e-2, weight_decay=0.1)
    loss = model(x, t, y).pow(2).mean()
    loss.backward()
    with_grad = {n for n, q in model.named_parameters() if q.grad is not None and q.grad.abs().sum() > 0}
    lora_names = {n for n, _ in model.named_parameters() if '.lora_' in n}
    results.append(check('gradients only on LoRA params', with_grad <= lora_names and len(with_grad) > 0,
                         f'{len(with_grad)} tensors with grad, {len(lora_names)} LoRA tensors'))
    opt.step()
    new_state = model.state_dict()
    changed = [k for k, v in base_state.items()
               if not torch.equal(v, new_state[k.replace('self_attn.', 'self_attn.base.')
                                  .replace('linear1.', 'linear1.base.').replace('linear2.', 'linear2.base.')
                                  if k.startswith('seqTransEncoder') and ('self_attn' in k or 'linear' in k) else k])]
    results.append(check('original weights (incl. CLIP) unchanged after step', not changed, f'changed={changed[:3]}'))

    # --- 4. counts
    c = count_parameters(model)
    per_layer = c['lora'] // len(cfg['layers'])
    print(f"\nLoRA config: {cfg}")
    print(f"MDM params (w/o CLIP, incl. pos-enc buffers excluded): {c['mdm_wo_clip']:,}  | CLIP: {c['clip']:,}")
    print(f"LoRA params: {c['lora']:,} ({per_layer:,}/layer) = {100 * c['lora'] / c['mdm_wo_clip']:.2f}% of MDM w/o CLIP"
          f" | trainable: {c['trainable']:,}")
    results.append(check('trainable == LoRA params', c['trainable'] == c['lora']))
    print('\nALL PASSED' if all(results) else '\nSOME TESTS FAILED')


if __name__ == '__main__':
    main()
