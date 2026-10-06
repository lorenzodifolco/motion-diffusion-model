"""Repair Perception Neuron root artifacts in KDAEE (joints in metres, Y-up, native 125 Hz).

1. Root "snaps": Axis Neuron lets the body drift (typically sinking below the floor during/after
   jumps) and then resets it in a single frame (10-40 cm steps in 8 ms). Steps larger than
   SNAP_STEP are replaced by the velocity interpolated from the neighbouring frames.
2. Re-grounding: on foot-contact frames (lowest foot joint nearly static) the vertical offset of
   the lowest foot w.r.t. its height in the first second (neutral stance right after calibration)
   is measured, interpolated over non-contact frames, low-passed and subtracted from all joints.
"""
import numpy as np
from scipy.signal import butter, filtfilt

FOOT_JOINTS = [7, 8, 10, 11]   # HumanML3D: l/r ankle, l/r foot
SNAP_STEP = 0.03               # m per frame at 125 Hz (3.75 m/s)
CONTACT_SPEED = 0.25           # m/s, lowest foot joint speed below which it is considered in contact
OFFSET_LOWPASS_HZ = 1.0
BASELINE_S = 1.0


def remove_root_snaps(joints, step=SNAP_STEP):
    """Returns repaired joints and number of repaired frames."""
    root = joints[:, 0]
    d = np.diff(root, axis=0)
    bad = (np.abs(d[:, 1]) > step) | (np.linalg.norm(d[:, [0, 2]], axis=1) > step)
    if not bad.any():
        return joints, 0
    good_idx = np.where(~bad)[0]
    d_fixed = d.copy()
    for c in range(3):
        d_fixed[bad, c] = np.interp(np.where(bad)[0], good_idx, d[good_idx, c])
    new_root = np.concatenate([root[:1], root[:1] + np.cumsum(d_fixed, axis=0)])
    return joints + (new_root - root)[:, None, :], int(bad.sum())


def reground(joints, fps):
    """Returns re-grounded joints and the subtracted vertical offset (T,)."""
    feet = joints[:, FOOT_JOINTS]
    lowest_j = feet[:, :, 1].argmin(1)
    lowest_y = feet[np.arange(len(feet)), lowest_j, 1]
    vel = np.zeros(len(feet))
    step = np.linalg.norm(feet[1:] - feet[:-1], axis=2)          # (T-1, 4)
    vel[1:] = step[np.arange(len(step)), lowest_j[1:]] * fps
    vel = filtfilt(*butter(2, 5.0 / (fps / 2)), vel)
    contact = vel < CONTACT_SPEED
    n0 = int(BASELINE_S * fps)
    base_frames = contact[:n0] if contact[:n0].any() else np.ones(n0, bool)
    baseline = np.median(lowest_y[:n0][base_frames])
    if contact.sum() < 2:
        return joints, np.zeros(len(joints))
    idx = np.where(contact)[0]
    offset = np.interp(np.arange(len(joints)), idx, lowest_y[idx] - baseline)
    offset = filtfilt(*butter(2, OFFSET_LOWPASS_HZ / (fps / 2)), offset)
    out = joints.copy()
    out[:, :, 1] -= offset[:, None]
    return out, offset


def repair(joints, fps):
    joints, n_snaps = remove_root_snaps(joints)
    joints, offset = reground(joints, fps)
    return joints, {'n_snap_frames': n_snaps,
                    'reground_offset_absmax': float(np.abs(offset).max()),
                    'reground_offset_absmed': float(np.median(np.abs(offset)))}
