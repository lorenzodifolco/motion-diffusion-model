"""Convert KDAEE BVH files to HumanML3D format (22 joints, 20 fps, 263-dim features).

Usage (from repo root):
    python -m scripts.kdaee.convert --out data/kdaee_hml3d [--variant A] [--limit N] [--workers 12]

Outputs (unnormalised, like HumanML3D; MDM normalises on load with dataset/HumanML3D/Mean.npy, Std.npy):
    <out>/new_joint_vecs/<id>.npy   (n_frames, 263) float32
    <out>/new_joints/<id>.npy       (n_frames, 22, 3) float32, recovered from features as in HumanML3D
    <out>/meta.csv                  one row per window
    <out>/clips_qc.csv              one row per source BVH (QC metrics, kept/excluded + reason)
"""
import argparse
import csv
import os
from multiprocessing import Pool

import numpy as np
from scipy.signal import butter, filtfilt

from scripts.kdaee import bvh as bvhlib
from scripts.kdaee import cleanup
from scripts.kdaee import hml3d

KDAEE_ROOT = 'dataset/KDAEE'
HML_ROOT = 'dataset/HumanML3D'
TARGET_FPS = 20
WINDOW = 197          # position frames -> 196 feature frames (MDM max_motion_length)
STRIDE = 98           # ~50% overlap
EMOTION_CODE = {'Angry': 'A', 'Disgust': 'D', 'Fearful': 'F', 'Happy': 'H',
                'Neutral': 'N', 'Sad': 'SA', 'Surprise': 'SU'}

# Excluded before conversion, with reason (see NOTES_kdaee.md).
MANUAL_EXCLUDE = {
    'M01N0V1': 'protocol: free performance (scenario 0) is defined only for emotional recordings',
}


def lowpass_resample(joints, fps_in, fps_out, cutoff):
    """Zero-phase Butterworth low-pass (anti-aliasing) then linear resampling. joints: (T, J, 3)."""
    if cutoff:
        b, a = butter(4, cutoff / (fps_in / 2.0))
        joints = filtfilt(b, a, joints, axis=0)
    t_in = np.arange(len(joints)) / fps_in
    t_out = np.arange(0.0, t_in[-1] + 1e-9, 1.0 / fps_out)
    flat = joints.reshape(len(joints), -1)
    out = np.stack([np.interp(t_out, t_in, flat[:, k]) for k in range(flat.shape[1])], 1)
    return out.reshape(len(t_out), *joints.shape[1:])


def window_starts(n_frames, window=WINDOW, stride=STRIDE):
    """Whole clip if it fits; otherwise evenly spaced windows with ~stride spacing covering the clip."""
    if n_frames <= window:
        return [0]
    n = int(round((n_frames - window) / stride)) + 1
    return [int(round(s)) for s in np.linspace(0, n_frames - window, max(n, 2))]


def clip_qc(j20):
    """Per-clip quality metrics on 20 fps joints (metres, before HumanML3D processing)."""
    feet = j20[:, [7, 8, 10, 11], 1]
    lowest = feet.min(1)
    root_v = np.linalg.norm(np.diff(j20[:, 0], axis=0), axis=1) * TARGET_FPS
    return {
        'has_nan': bool(np.isnan(j20).any()),
        'root_speed_max': float(root_v.max()),
        'lowest_foot_p1': float(np.percentile(lowest, 1)),
        'lowest_foot_p50': float(np.median(lowest)),
        'lowest_foot_p99': float(np.percentile(lowest, 99)),
        'root_height_p50': float(np.median(j20[:, 0, 1])),
    }


# Automatic exclusion rules, applied after repair (thresholds chosen on the QC distributions and
# checked visually, see NOTES_kdaee.md). A large re-grounding offset alone is NOT a failure: it measures
# how much the raw data drifted, and visually repaired clips (e.g. M01SU0V1) look correct.
QC_RULES = [
    ('n_snap_frames', lambda v: v > 20, 'root snaps: >20 frames with >3 cm single-frame root steps'),
    ('lowest_foot_p99', lambda v: v > 0.5, 'feet >50 cm above the floor for >1% of frames'),
]


def qc_exclusion(qc):
    return '; '.join(msg for key, bad, msg in QC_RULES if key in qc and bad(qc[key]))


def _init_worker():
    hml3d.init_motion_process(HML_ROOT)


def convert_one(job):
    row, args = job
    name = row['filename']
    path = os.path.join(KDAEE_ROOT, 'BVH', row['actor_ID'], name + '.bvh')
    b = bvhlib.read_bvh(path)
    pos, ends, rots = bvhlib.forward_kinematics(b, return_rotations=True)
    if args.variant == 'R':
        j = hml3d.kdaee_retarget_smpl(b, pos, rots)
    else:
        j = hml3d.kdaee_to_hml_joints(b, pos, ends, args.variant)
    repair_info = {}
    if not args.no_repair:
        j, repair_info = cleanup.repair(j, b.fps)
    j20 = lowpass_resample(j, b.fps, TARGET_FPS, args.cutoff)
    qc = dict(filename=name, emotion=row['emotion'], n_frames_125=len(b.motion), n_frames_20=len(j20),
              **repair_info, **clip_qc(j20))
    reason = qc_exclusion(qc)
    qc['qc_excluded'] = reason
    if reason and not args.keep_qc_failures:
        return qc, []

    windows = []
    starts = window_starts(len(j20))
    for k, s in enumerate(starts):
        seg = j20[s:s + WINDOW]
        feats, rec = hml3d.joints_to_features(seg)
        wid = f'{name}_w{k}'
        np.save(os.path.join(args.out, 'new_joint_vecs', wid + '.npy'), feats)
        np.save(os.path.join(args.out, 'new_joints', wid + '.npy'), rec)
        windows.append(dict(
            id=wid, source=name, actor=row['actor_ID'], gender=row['actor_gender'],
            emotion=row['emotion'], emotion_code=EMOTION_CODE[row['emotion']],
            scenario_id=int(row['scenario_ID']),
            scenario_code=f"{EMOTION_CODE[row['emotion']]}{row['scenario_ID']}",
            version=int(row['version']), window_idx=k, n_windows=len(starts),
            start_frame=s, end_frame=s + len(seg), n_frames=len(feats),
            start_s=round(s / TARGET_FPS, 3), duration_s=round(len(feats) / TARGET_FPS, 3),
            variant=args.variant, cutoff_hz=args.cutoff))
    return qc, windows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', default='data/kdaee_hml3d')
    p.add_argument('--variant', default='R', choices=sorted(hml3d.JOINT_MAPS) + ['R'],
                   help='A/B: copy joint positions (spine3=Spine2/Spine3); R: rotations onto SMPL rest skeleton')
    p.add_argument('--cutoff', type=float, default=8.0, help='low-pass cutoff in Hz at 125 Hz (0 = off)')
    p.add_argument('--limit', type=int, default=0, help='convert only the first N files (debug)')
    p.add_argument('--files', nargs='*', help='convert only these file names (no extension)')
    p.add_argument('--workers', type=int, default=12)
    p.add_argument('--no_repair', action='store_true', help='skip root snap removal / re-grounding')
    p.add_argument('--keep_qc_failures', action='store_true', help='convert clips failing QC_RULES too')
    args = p.parse_args()

    os.makedirs(os.path.join(args.out, 'new_joint_vecs'), exist_ok=True)
    os.makedirs(os.path.join(args.out, 'new_joints'), exist_ok=True)

    rows = list(csv.DictReader(open(os.path.join(KDAEE_ROOT, 'file-info.csv'))))
    excluded = [dict(filename=r['filename'], emotion=r['emotion'], reason=MANUAL_EXCLUDE[r['filename']])
                for r in rows if r['filename'] in MANUAL_EXCLUDE]
    rows = [r for r in rows if r['filename'] not in MANUAL_EXCLUDE]
    if args.files:
        rows = [r for r in rows if r['filename'] in set(args.files)]
    if args.limit:
        rows = rows[:args.limit]

    with Pool(args.workers, initializer=_init_worker) as pool:
        results = pool.map(convert_one, [(r, args) for r in rows], chunksize=4)

    qcs = [q for q, _ in results]
    windows = [w for _, ws in results for w in ws]
    excluded += [dict(filename=q['filename'], emotion=q['emotion'], reason=q['qc_excluded'])
                 for q in qcs if q['qc_excluded'] and not args.keep_qc_failures]
    with open(os.path.join(args.out, 'meta.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(windows[0]))
        w.writeheader()
        w.writerows(windows)
    with open(os.path.join(args.out, 'clips_qc.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(qcs[0]))
        w.writeheader()
        w.writerows(qcs)
    with open(os.path.join(args.out, 'excluded.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['filename', 'emotion', 'reason'])
        w.writeheader()
        w.writerows(excluded)
    print(f'{len(rows)} clips -> {len(windows)} windows, '
          f'{sum(x["n_frames"] for x in windows) / TARGET_FPS / 3600:.2f} h; excluded {len(excluded)}')


if __name__ == '__main__':
    main()
