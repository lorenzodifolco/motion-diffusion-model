"""Minimal BVH reader + forward kinematics for KDAEE (Perception Neuron / Axis Neuron export).

Verified on KDAEE v2.1.0 (see NOTES_kdaee.md):
  - single hierarchy, 59 joints, every joint has 6 channels
    (Xposition Yposition Zposition Xrotation Yrotation Zrotation), 125 Hz, cm, Y-up.
  - non-root position channels = OFFSET + Perception Neuron "displacement" jitter;
    we ignore them and use the fixed OFFSETs so bone lengths are constant.
"""
import numpy as np


class BVH:
    def __init__(self, names, parents, offsets, end_sites, channels, frame_time, motion):
        self.names = names              # list[str], joints only (no End Sites)
        self.parents = parents          # np.int array, -1 for root
        self.offsets = offsets          # (J, 3) rest offsets (cm)
        self.end_sites = end_sites      # dict joint_idx -> (3,) offset of its End Site
        self.channels = channels        # list[list[str]] per joint
        self.frame_time = frame_time
        self.motion = motion            # (T, sum(channels))

    @property
    def fps(self):
        return 1.0 / self.frame_time

    def index(self, name):
        return self.names.index(name)


def read_bvh(path):
    with open(path) as f:
        text = f.read()
    head, body = text.split('MOTION', 1)

    names, parents, offsets, channels, end_sites = [], [], [], [], {}
    stack, cur, in_end = [], None, False
    for line in head.splitlines():
        t = line.strip()
        if t.startswith('ROOT') or t.startswith('JOINT'):
            names.append(t.split()[1])
            parents.append(stack[-1] if stack else -1)
            cur = len(names) - 1
            in_end = False
        elif t.startswith('End Site'):
            in_end = True
        elif t == '{':
            if not in_end:
                stack.append(cur)
        elif t == '}':
            if in_end:
                in_end = False
            else:
                stack.pop()
        elif t.startswith('OFFSET'):
            o = np.array([float(v) for v in t.split()[1:4]])
            if in_end:
                end_sites[cur] = o
            else:
                offsets.append(o)
        elif t.startswith('CHANNELS'):
            channels.append(t.split()[2:])

    lines = body.strip().split('\n', 2)
    n_frames = int(lines[0].split(':')[1])
    frame_time = float(lines[1].split(':')[1])
    motion = np.array(lines[2].split(), dtype=np.float64)
    n_ch = sum(len(c) for c in channels)
    assert motion.size == n_frames * n_ch, f'{path}: {motion.size} values, expected {n_frames}x{n_ch}'
    motion = motion.reshape(n_frames, n_ch)

    for c in channels:
        assert c == ['Xposition', 'Yposition', 'Zposition', 'Xrotation', 'Yrotation', 'Zrotation'], c
    return BVH(names, np.array(parents), np.stack(offsets), end_sites, channels, frame_time, motion)


def _euler_xyz_to_mat(deg):
    """Intrinsic X-Y-Z Euler (BVH channel order) -> rotation matrices. deg: (..., 3)."""
    a, b, c = np.deg2rad(deg[..., 0]), np.deg2rad(deg[..., 1]), np.deg2rad(deg[..., 2])
    ca, sa, cb, sb, cc, sc = np.cos(a), np.sin(a), np.cos(b), np.sin(b), np.cos(c), np.sin(c)
    one, zero = np.ones_like(a), np.zeros_like(a)
    rx = np.stack([one, zero, zero, zero, ca, -sa, zero, sa, ca], -1).reshape(a.shape + (3, 3))
    ry = np.stack([cb, zero, sb, zero, one, zero, -sb, zero, cb], -1).reshape(a.shape + (3, 3))
    rz = np.stack([cc, -sc, zero, sc, cc, zero, zero, zero, one], -1).reshape(a.shape + (3, 3))
    return rx @ ry @ rz


def forward_kinematics(bvh, use_displacement=False, return_rotations=False):
    """Global joint positions (T, J, 3) and End Site positions {joint_idx: (T, 3)}, in BVH units (cm).

    With return_rotations=True also returns the global rotation matrices (T, J, 3, 3).
    """
    T, J = len(bvh.motion), len(bvh.names)
    m = bvh.motion.reshape(T, J, 6)
    local_rot = _euler_xyz_to_mat(m[:, :, 3:])
    glob_rot = np.empty_like(local_rot)
    pos = np.empty((T, J, 3))
    for j in range(J):
        p = bvh.parents[j]
        if p < 0:
            pos[:, j] = m[:, j, :3]
            glob_rot[:, j] = local_rot[:, j]
            continue
        off = m[:, j, :3] if use_displacement else np.broadcast_to(bvh.offsets[j], (T, 3))
        pos[:, j] = pos[:, p] + np.einsum('tij,tj->ti', glob_rot[:, p], off)
        glob_rot[:, j] = glob_rot[:, p] @ local_rot[:, j]
    ends = {j: pos[:, j] + glob_rot[:, j] @ o for j, o in bvh.end_sites.items()}
    if return_rotations:
        return pos, ends, glob_rot
    return pos, ends
