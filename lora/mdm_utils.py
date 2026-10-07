"""Load the pretrained MDM and build KDAEE data loaders with MDM's own HumanML3D pipeline."""
import json
import os
from argparse import ArgumentParser
from functools import partial
from os.path import join as pjoin
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

from data_loaders.humanml.data.dataset import Text2MotionDatasetV2
from data_loaders.humanml.utils.get_opt import get_opt
from data_loaders.humanml.utils.word_vectorizer import WordVectorizer
from data_loaders.tensors import t2m_collate
from utils import parser_util
from utils.model_util import create_model_and_diffusion, load_model_wo_clip


def load_mdm_args(model_path):
    """Same resolution as sample.generate: parser defaults overwritten by the model's args.json."""
    parser = ArgumentParser()
    parser_util.add_base_options(parser)
    parser_util.add_data_options(parser)
    parser_util.add_model_options(parser)
    parser_util.add_diffusion_options(parser)
    args = parser.parse_args([])
    with open(pjoin(os.path.dirname(model_path), 'args.json')) as f:
        model_args = json.load(f)
    for group in ('dataset', 'model', 'diffusion'):
        for name in parser_util.get_args_per_group_name(parser, args, group):
            if name in model_args:
                setattr(args, name, model_args[name])
            elif 'cond_mode' in model_args:
                args.unconstrained = model_args['cond_mode'] == 'no_cond'
    args.model_path = model_path
    return parser_util.apply_rules(args)


def load_pretrained_mdm(model_path, device='cpu'):
    args = load_mdm_args(model_path)
    dummy_data = SimpleNamespace(dataset=SimpleNamespace())   # only used for num_actions (text model -> 1)
    model, diffusion = create_model_and_diffusion(args, dummy_data)
    load_model_wo_clip(model, torch.load(model_path, map_location='cpu'))
    model.to(device)   # MDM._apply does not return self, so model.to() returns None
    return model, diffusion, args


def kdaee_dataset(data_root, split_file, hml_root='dataset/HumanML3D', device=None):
    """Text2MotionDatasetV2 (same cropping / normalisation / padding as MDM pretraining) on KDAEE windows,
    normalised with the pretrained model's HumanML3D Mean/Std (not recomputed on KDAEE)."""
    opt = get_opt('dataset/humanml_opt.txt', device)
    opt.data_root = data_root
    opt.motion_dir = pjoin(data_root, 'new_joint_vecs')
    opt.text_dir = pjoin(data_root, 'texts')
    # Text2MotionDatasetV2 always writes <cache_dir>/dataset/t2m_<split>.npy: keep it next to the split
    # files, never in ./dataset/ where it would shadow the HumanML3D cache.
    opt.cache_dir = os.path.dirname(os.path.abspath(split_file))
    os.makedirs(pjoin(opt.cache_dir, 'dataset'), exist_ok=True)
    opt.use_cache = False
    opt.fixed_len = 0
    opt.disable_offset_aug = False
    mean = np.load(pjoin(hml_root, 'Mean.npy'))
    std = np.load(pjoin(hml_root, 'Std.npy'))
    w_vectorizer = WordVectorizer('glove', 'our_vab')
    return Text2MotionDatasetV2(opt, mean, std, split_file, w_vectorizer)


def _seed_numpy_worker(worker_id):
    # torch 1.7 reseeds `random` and torch in each worker but not numpy (used for the crop length coin)
    np.random.seed(torch.initial_seed() % 2 ** 32)


def kdaee_loader(data_root, split_file, batch_size, shuffle=True, drop_last=True, num_workers=4):
    ds = kdaee_dataset(data_root, split_file)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      drop_last=drop_last, collate_fn=partial(t2m_collate, target_batch_size=batch_size),
                      worker_init_fn=_seed_numpy_worker)


def respaced_diffusion(args, steps):
    """Same as utils.model_util.create_gaussian_diffusion (x0 prediction, MSE, fixed small sigma) but with the
    sampling chain respaced to `steps` evenly spaced timesteps of the original `args.diffusion_steps`."""
    from diffusion import gaussian_diffusion as gd
    from diffusion.respace import SpacedDiffusion, space_timesteps
    return SpacedDiffusion(
        use_timesteps=space_timesteps(args.diffusion_steps, [steps]),
        betas=gd.get_named_beta_schedule(args.noise_schedule, args.diffusion_steps, 1.),
        model_mean_type=gd.ModelMeanType.START_X,
        model_var_type=gd.ModelVarType.FIXED_SMALL if args.sigma_small else gd.ModelVarType.FIXED_LARGE,
        loss_type=gd.LossType.MSE,
        rescale_timesteps=False,
        lambda_vel=args.lambda_vel, lambda_rcxyz=args.lambda_rcxyz, lambda_fc=args.lambda_fc,
        lambda_target_loc=getattr(args, 'lambda_target_loc', 0.),
    )
