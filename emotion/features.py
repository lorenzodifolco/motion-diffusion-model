"""Hand-crafted kinematic features per window, from 22-joint positions (m, Y-up, 20 fps, HumanML3D joint order).

Same function for real, augmented and synthetic windows. Dynamics (speed / acceleration / jerk) use global
positions (magnitudes, rotation invariant); posture features use a per-frame heading-normalised body frame
(root at the origin in XZ, body facing +Z), so they do not depend on the facing direction. Window duration is
deliberately NOT a feature (it reflects recording/windowing, not emotion).

Features (35):
  {head, trunk, arms, legs}_{speed, acc, jerk}_{mean, std}   24  per-frame mean joint magnitude of the group,
                                                                  mean / std over time
  bbox_volume_{mean, std}                                     2  body-frame bounding-box volume (m^3)
  expansion_{mean, std}                                       2  mean distance of head, hands, feet from the pelvis
  head_incl_{mean, std}                                       2  angle neck->head vs vertical (deg)
  head_pitch_mean                                             1  signed sagittal tilt of neck->head (deg, + forward)
  shoulder_elev_{mean, std}                                   2  shoulder height above spine3 (m)
  arm_speed_asym_mean                                         1  |v_L - v_R| / (v_L + v_R), elbows and wrists
  arm_pos_asym_mean                                           1  distance between left and mirrored right
                                                                  elbow/wrist positions in the body frame (m)
"""
import numpy as np

FPS = 20
GROUPS = {'head': [12, 15], 'trunk': [0, 3, 6, 9, 13, 14], 'arms': [16, 17, 18, 19, 20, 21],
          'legs': [1, 2, 4, 5, 7, 8, 10, 11]}
PELVIS, SPINE3, NECK, HEAD = 0, 9, 12, 15
L_HIP, R_HIP, L_SHO, R_SHO = 1, 2, 16, 17
EXTREMITIES = [15, 20, 21, 10, 11]
ARM_PAIRS = [(18, 19), (20, 21)]                                   # elbows, wrists (left, right)


def body_frame(j):
    """Per-frame heading-normalised joints: subtract root XZ, rotate so the body faces +Z."""
    across = (j[:, R_HIP] - j[:, L_HIP]) + (j[:, R_SHO] - j[:, L_SHO])
    fwd = np.cross(np.array([0., 1., 0.]), across)
    ang = np.arctan2(fwd[:, 0], fwd[:, 2])                         # heading angle w.r.t. +Z
    c, s = np.cos(-ang), np.sin(-ang)
    loc = j.copy()
    loc[..., 0] -= j[:, :1, 0]
    loc[..., 2] -= j[:, :1, 2]
    x, z = loc[..., 0].copy(), loc[..., 2].copy()
    loc[..., 0] = c[:, None] * x + s[:, None] * z
    loc[..., 2] = -s[:, None] * x + c[:, None] * z
    return loc


def window_features(j):
    j = np.asarray(j, np.float64)
    f = {}
    vel = np.diff(j, axis=0) * FPS
    acc = np.diff(j, n=2, axis=0) * FPS ** 2
    jerk = np.diff(j, n=3, axis=0) * FPS ** 3
    for name, idx in GROUPS.items():
        for qname, q in (('speed', vel), ('acc', acc), ('jerk', jerk)):
            m = np.linalg.norm(q[:, idx], axis=-1).mean(1)
            f[f'{name}_{qname}_mean'] = m.mean()
            f[f'{name}_{qname}_std'] = m.std()

    loc = body_frame(j)
    vol = np.prod(loc.max(1) - loc.min(1), axis=1)
    f['bbox_volume_mean'], f['bbox_volume_std'] = vol.mean(), vol.std()
    ext = np.linalg.norm(loc[:, EXTREMITIES] - loc[:, [PELVIS]], axis=-1).mean(1)
    f['expansion_mean'], f['expansion_std'] = ext.mean(), ext.std()

    head = loc[:, HEAD] - loc[:, NECK]
    incl = np.degrees(np.arccos(np.clip(head[:, 1] / (np.linalg.norm(head, axis=1) + 1e-9), -1, 1)))
    f['head_incl_mean'], f['head_incl_std'] = incl.mean(), incl.std()
    f['head_pitch_mean'] = np.degrees(np.arctan2(head[:, 2], head[:, 1])).mean()
    sho = loc[:, [L_SHO, R_SHO], 1].mean(1) - loc[:, SPINE3, 1]
    f['shoulder_elev_mean'], f['shoulder_elev_std'] = sho.mean(), sho.std()

    sp, pos = [], []
    for a, b in ARM_PAIRS:
        sa, sb = np.linalg.norm(vel[:, a], axis=-1), np.linalg.norm(vel[:, b], axis=-1)
        sp.append(np.abs(sa - sb) / (sa + sb + 1e-6))
        pos.append(np.linalg.norm(loc[:, a] - loc[:, b] * np.array([-1, 1, 1]), axis=-1))
    f['arm_speed_asym_mean'] = np.mean(sp)
    f['arm_pos_asym_mean'] = np.mean(pos)
    return f


FEATURE_NAMES = list(window_features(np.random.default_rng(0).standard_normal((40, 22, 3))))
