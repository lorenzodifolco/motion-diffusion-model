"""Inject / save / load LoRA adapters in MDM's transformer encoder."""
import torch.nn as nn

from lora.layers import LoRALinear, LoRAMultiheadAttention

DEFAULT_TARGETS = ('q', 'k', 'v', 'o', 'ffn1', 'ffn2')


def inject_lora(mdm, r=8, alpha=8, targets=DEFAULT_TARGETS, dropout=0.0, layers=None):
    """Wrap self-attention (q/k/v/o) and FFN (linear1/linear2) of mdm.seqTransEncoder layers in place,
    then freeze everything except the LoRA parameters (CLIP included). Returns the LoRA config dict."""
    assert mdm.arch == 'trans_enc', 'LoRA injection implemented for the trans_enc architecture'
    enc_layers = mdm.seqTransEncoder.layers
    layer_ids = list(range(len(enc_layers))) if layers is None else list(layers)
    attn_targets = tuple(t for t in targets if t in 'qkvo' and len(t) == 1)
    for i in layer_ids:
        layer = enc_layers[i]
        if attn_targets:
            layer.self_attn = LoRAMultiheadAttention(layer.self_attn, r, alpha, attn_targets)
        if 'ffn1' in targets:
            layer.linear1 = LoRALinear(layer.linear1, r, alpha, dropout)
        if 'ffn2' in targets:
            layer.linear2 = LoRALinear(layer.linear2, r, alpha, dropout)
    for name, p in mdm.named_parameters():
        p.requires_grad = is_lora_param(name)
    return {'r': r, 'alpha': alpha, 'targets': list(targets), 'dropout': dropout, 'layers': layer_ids}


def is_lora_param(name):
    return '.lora_A' in name or '.lora_B' in name


def lora_parameters(model):
    return [p for n, p in model.named_parameters() if is_lora_param(n)]


def lora_state_dict(model):
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters() if is_lora_param(n)}


def load_lora_state_dict(model, state_dict):
    own = dict(model.named_parameters())
    missing = [n for n in own if is_lora_param(n) and n not in state_dict]
    unexpected = [n for n in state_dict if n not in own]
    assert not missing and not unexpected, (missing, unexpected)
    for n, v in state_dict.items():
        own[n].data.copy_(v.to(own[n].device))


def count_parameters(model):
    total = sum(p.numel() for n, p in model.named_parameters() if not n.startswith('clip_model.'))
    clip = sum(p.numel() for n, p in model.named_parameters() if n.startswith('clip_model.'))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lora = sum(p.numel() for n, p in model.named_parameters() if is_lora_param(n))
    return {'mdm_wo_clip': total - lora, 'clip': clip, 'lora': lora, 'trainable': trainable}
