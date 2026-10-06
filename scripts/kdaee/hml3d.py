"""KDAEE skeleton -> HumanML3D 22-joint positions -> 263-dim HumanML3D features.

Wraps data_loaders/humanml/scripts/motion_process.py without modifying it: process_file()
reads module-level globals that upstream only defines in a commented-out __main__, so we set
them here with the t2m values.
"""
import os

import numpy as np
import torch

import data_loaders.humanml.scripts.motion_process as mp
from data_loaders.humanml.common.skeleton import Skeleton
from data_loaders.humanml.utils.paramUtil import t2m_raw_offsets, t2m_kinematic_chain

HML_JOINTS = ['pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee', 'spine2',
              'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot', 'neck',
              'left_collar', 'right_collar', 'head', 'left_shoulder', 'right_shoulder',
              'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist']

# KDAEE joint for each HumanML3D joint. 'X:end' = End Site of joint X.
_COMMON = {
    'pelvis': 'Hips', 'left_hip': 'LeftUpLeg', 'right_hip': 'RightUpLeg',
    'left_knee': 'LeftLeg', 'right_knee': 'RightLeg',
    'left_ankle': 'LeftFoot', 'right_ankle': 'RightFoot',
    'left_foot': 'LeftFoot:end', 'right_foot': 'RightFoot:end',
    'neck': 'Neck', 'head': 'Head',
    'left_collar': 'LeftShoulder', 'right_collar': 'RightShoulder',
    'left_shoulder': 'LeftArm', 'right_shoulder': 'RightArm',
    'left_elbow': 'LeftForeArm', 'right_elbow': 'RightForeArm',
    'left_wrist': 'LeftHand', 'right_wrist': 'RightHand',
}
JOINT_MAPS = {
    # A: matches cumulative heights above the pelvis (Spine 14/Spine1 24/Spine2 34/Neck 54 cm
    #    vs t2m spine1 13/spine2 28/spine3 33/neck 55 cm)
    'A': dict(_COMMON, spine1='Spine', spine2='Spine1', spine3='Spine2'),
    # B: spine3 at the real branching point of neck and shoulders in KDAEE
    'B': dict(_COMMON, spine1='Spine', spine2='Spine2', spine3='Spine3'),
}

FEET_THRE = 0.002
T2M_TGT_SKEL_ID = '000021'
SMPL_PATH = 'body_models/smpl/SMPL_NEUTRAL.pkl'
HML_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]

# Variant R: KDAEE segment whose global rotation drives the SMPL bone parent(i) -> i.
# Heights above the pelvis: SMPL spine1 11 / spine2 24 / spine3 30 cm, KDAEE Spine 14 / Spine1 24 / Spine2 34 cm.
RETARGET_SEGMENT = {
    'left_hip': 'Hips', 'right_hip': 'Hips', 'spine1': 'Hips',
    'left_knee': 'LeftUpLeg', 'right_knee': 'RightUpLeg',
    'left_ankle': 'LeftLeg', 'right_ankle': 'RightLeg',
    'left_foot': 'LeftFoot', 'right_foot': 'RightFoot',
    'spine2': 'Spine', 'spine3': 'Spine1',
    'neck': 'Spine3', 'left_collar': 'Spine3', 'right_collar': 'Spine3',
    'head': 'Neck',
    'left_shoulder': 'LeftShoulder', 'right_shoulder': 'RightShoulder',
    'left_elbow': 'LeftArm', 'right_elbow': 'RightArm',
    'left_wrist': 'LeftForeArm', 'right_wrist': 'RightForeArm',
}
# Bones whose rest direction is taken from the KDAEE OFFSET (with SMPL length) instead of SMPL:
# SMPL's collar->shoulder rest vector is tilted ~18 deg upwards, KDAEE's is horizontal; with the SMPL
# direction the mean collar->shoulder direction deviates 15 deg from HumanML3D, with KDAEE's 4 deg.
REST_DIRECTION_FROM_KDAEE = {'left_shoulder': 'LeftArm', 'right_shoulder': 'RightArm'}
_smpl_rest_offsets = None


def smpl_rest_offsets():
    """SMPL neutral rest-pose bone vectors parent->child (22, 3), metres. Same axes as KDAEE (Y up, +Z fwd, +X left)."""
    global _smpl_rest_offsets
    if _smpl_rest_offsets is None:
        import pickle
        import scipy.sparse
        with open(SMPL_PATH, 'rb') as f:
            smpl = pickle.load(f, encoding='latin1')
        jr = smpl['J_regressor']
        jr = jr.toarray() if scipy.sparse.issparse(jr) else np.asarray(jr)
        joints = (jr @ np.asarray(smpl['v_template']))[:22]
        off = np.zeros((22, 3))
        for i, p in enumerate(HML_PARENTS):
            if p >= 0:
                off[i] = joints[i] - joints[p]
        _smpl_rest_offsets = off
    return _smpl_rest_offsets


def kdaee_retarget_smpl(bvh, global_pos, global_rot):
    """Variant R: transfer KDAEE global segment rotations onto the SMPL rest skeleton.

    Both rest poses are T-poses with the same axes, so a zero KDAEE rotation maps to the SMPL rest pose;
    static anatomical offsets (hips below the pelvis, head in front of the neck, ...) come from SMPL.
    Root translation = KDAEE Hips (pelvis height 93.6 cm vs SMPL 93.9 cm, no rescaling).
    Returns (T, 22, 3) metres in HumanML3D joint order.
    """
    off = smpl_rest_offsets().copy()
    for name, kd_joint in REST_DIRECTION_FROM_KDAEE.items():
        i, d = HML_JOINTS.index(name), bvh.offsets[bvh.index(kd_joint)]
        off[i] = d / np.linalg.norm(d) * np.linalg.norm(off[i])
    out = np.zeros((len(global_pos), 22, 3))
    out[:, 0] = global_pos[:, bvh.index('Hips')] / 100.0
    for i in range(1, 22):
        rot = global_rot[:, bvh.index(RETARGET_SEGMENT[HML_JOINTS[i]])]
        out[:, i] = out[:, HML_PARENTS[i]] + rot @ off[i]
    return out


def kdaee_to_hml_joints(bvh, global_pos, end_sites, variant='A'):
    """(T, 59, 3) cm -> (T, 22, 3) metres in HumanML3D joint order (position-copy variants A/B)."""
    jmap = JOINT_MAPS[variant]
    out = []
    for name in HML_JOINTS:
        src = jmap[name]
        if src.endswith(':end'):
            out.append(end_sites[bvh.index(src[:-4])])
        else:
            out.append(global_pos[:, bvh.index(src)])
    return np.stack(out, 1) / 100.0


def init_motion_process(hml_root):
    """Set the globals that motion_process.process_file expects (t2m configuration)."""
    tgt = np.load(os.path.join(hml_root, 'new_joints', T2M_TGT_SKEL_ID + '.npy'))
    mp.n_raw_offsets = torch.from_numpy(t2m_raw_offsets)
    mp.kinematic_chain = t2m_kinematic_chain
    mp.l_idx1, mp.l_idx2 = 5, 8
    mp.fid_r, mp.fid_l = [8, 11], [7, 10]
    mp.face_joint_indx = [2, 1, 17, 16]
    mp.r_hip, mp.l_hip = 2, 1
    mp.joints_num = 22
    skel = Skeleton(mp.n_raw_offsets, mp.kinematic_chain, 'cpu')
    mp.tgt_offsets = skel.get_offsets_joints(torch.from_numpy(tgt[0]).float())
    return mp.tgt_offsets


def joints_to_features(joints):
    """(T, 22, 3) metres, Y-up, 20 fps -> features (T-1, 263) and recovered joints (T-1, 22, 3).

    Same steps HumanML3D used to produce new_joint_vecs/new_joints (unnormalised).
    """
    data, _, _, _ = mp.process_file(joints.astype(np.float64), FEET_THRE)
    rec = mp.recover_from_ric(torch.from_numpy(data).unsqueeze(0).float(), 22).squeeze(0).numpy()
    return data.astype(np.float32), rec.astype(np.float32)
