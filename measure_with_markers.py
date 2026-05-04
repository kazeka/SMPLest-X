"""
Body measurements (mm) from SMPLest-X results, with anatomical slice heights
determined by ArUco markers on a calibration garment instead of derived from
SMPL-X spine joints.

Slice heights come from markers when detected reliably (>=2 of 4 per band),
and fall back to joint-derived heights on a per-band basis. Both are always
printed for comparison, enabling a direct marker vs. markerless evaluation.

Requires inference.py to have been run with --calibration_npz so that the
per-frame .npz files contain non-zero camera_K fields.

Usage:
    python measure_with_markers.py demo/results/myvideo/ \\
        --garment_config garment_v1.json

    python measure_with_markers.py demo/results/myvideo/ \\
        --garment_config garment_v1.json \\
        --calibration_npz p50pro_main_12MP.npz \\
        --ground_truth 1815,1070,990,1040,470,1400
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np
from tqdm import tqdm

from measure_bodies import (
    L_HIP, L_SHOULDER, L_WRIST, R_HIP, R_SHOULDER, R_WRIST,
    SPINE_1, SPINE_2, SPINE_3,
    cross_section_perimeter,
    load_smplx,
    reject_betas_outliers,
    tpose_mesh,
)


# ---------------------------------------------------------------------------
# ArUco marker 3D estimation
# ---------------------------------------------------------------------------

def _marker_object_points(size_mm: float) -> np.ndarray:
    """4 object-space corners of a square marker in marker-local coords (metres).

    OpenCV ArUco corner order: top-left, top-right, bottom-right, bottom-left.
    Marker frame: origin at centre, +X right, +Y down, +Z out of marker plane.
    """
    half = size_mm / 2000.0
    return np.array([
        [-half, -half, 0.0],
        [+half, -half, 0.0],
        [+half, +half, 0.0],
        [-half, +half, 0.0],
    ], dtype=np.float32)


def marker_centre_camera_frame(
    corners_2d: np.ndarray,
    marker_size_mm: float,
    K: np.ndarray,
    dist: np.ndarray,
) -> np.ndarray | None:
    """Return marker centre 3D position in camera frame (metres), or None on failure."""
    obj_pts = _marker_object_points(marker_size_mm)
    success, _, tvec = cv2.solvePnP(
        obj_pts, corners_2d.astype(np.float32), K, dist,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    if not success:
        return None
    return tvec.flatten()  # marker origin is its centre


def transform_camera_to_smplx_local(
    point_cam: np.ndarray,
    smplx_global_orient: np.ndarray,
    smplx_cam_trans: np.ndarray,
) -> np.ndarray:
    """Convert a 3D point from camera frame to SMPL-X body-local frame.

    Body-to-camera: x_cam = R_global @ x_body + cam_trans
    Camera-to-body: x_body = R_global.T @ (x_cam - cam_trans)
    """
    R_global, _ = cv2.Rodrigues(np.array(smplx_global_orient, dtype=np.float64).reshape(3, 1))
    return R_global.T @ (point_cam.astype(np.float64) - smplx_cam_trans.astype(np.float64))


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_markers_in_body_frame(
    per_frame_data: list[dict],
    garment_config: dict,
) -> tuple[dict[int, np.ndarray], dict[int, dict]]:
    """Aggregate each marker's body-local 3D position across frames.

    Returns:
        aggregated: marker_id -> median position (3,) — only reliably detected markers
        stats:      marker_id -> {n_frames, std_mm, rejected}
    """
    positions_per_id: dict[int, list[np.ndarray]] = {}

    for frame_data in per_frame_data:
        ids      = frame_data['markers_ids']
        corners  = frame_data['markers_corners']
        K        = frame_data['camera_K']
        dist     = frame_data['camera_dist']
        cam_trans       = frame_data['cam_trans']
        global_orient   = frame_data['global_orient']

        if np.all(K == 0):
            continue  # no calibration for this frame

        for idx in range(len(ids)):
            mid = int(ids[idx])
            if str(mid) not in garment_config['markers']:
                continue

            size_mm = garment_config['markers'][str(mid)]['size_mm']
            centre_cam = marker_centre_camera_frame(corners[idx], size_mm, K, dist)
            if centre_cam is None:
                continue

            centre_body = transform_camera_to_smplx_local(centre_cam, global_orient, cam_trans)
            positions_per_id.setdefault(mid, []).append(centre_body)

    aggregated: dict[int, np.ndarray] = {}
    stats: dict[int, dict] = {}
    _STD_REJECT_M = 0.05  # 50 mm — reject markers that don't cluster

    for mid, positions in positions_per_id.items():
        arr = np.array(positions)  # (N, 3)
        median = np.median(arr, axis=0)
        std = np.std(arr, axis=0)
        std_mm = float(np.linalg.norm(std) * 1000)
        rejected = std_mm > (_STD_REJECT_M * 1000)
        stats[mid] = {'n_frames': len(arr), 'std_mm': std_mm, 'rejected': rejected}
        if rejected:
            print(f"  warning: marker {mid} std={std_mm:.1f} mm > {_STD_REJECT_M * 1000:.0f} mm threshold; rejected")
        else:
            aggregated[mid] = median

    return aggregated, stats


def slice_heights_from_markers(
    aggregated: dict[int, np.ndarray],
    garment_config: dict,
) -> dict[str, float | None]:
    """For each band, compute the slice Y-height as mean Y of its detected markers.

    Returns None for bands with fewer than 2 detected markers.
    """
    band_y: dict[str, float | None] = {}
    for band in ('chest', 'waist', 'hips'):
        ys = []
        for mid_str, m_info in garment_config['markers'].items():
            if m_info['band'] != band:
                continue
            mid = int(mid_str)
            if mid in aggregated:
                ys.append(float(aggregated[mid][1]))  # SMPL-X Y-axis is up
        if len(ys) >= 2:
            band_y[band] = float(np.mean(ys))
        else:
            band_y[band] = None
            print(f"  warning: only {len(ys)} marker(s) found for '{band}' band — falling back to joints")
    return band_y


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_frame_data(path: str, K_override: np.ndarray | None, dist_override: np.ndarray | None) -> dict:
    data = dict(np.load(path, allow_pickle=False))
    K    = K_override    if K_override    is not None else data.get('camera_K',    np.zeros((3, 3)))
    dist = dist_override if dist_override is not None else data.get('camera_dist', np.zeros(5))
    return {
        'betas':          data['betas'].reshape(10).astype(np.float32),
        'cam_trans':      data.get('cam_trans',       np.zeros(3)).astype(np.float64),
        'global_orient':  data.get('global_orient',   np.zeros(3)).astype(np.float64),
        'markers_ids':    data.get('markers_ids',     np.zeros((0,), dtype=np.int32)),
        'markers_corners':data.get('markers_corners', np.zeros((0, 4, 2), dtype=np.float32)),
        'camera_K':       K.astype(np.float64),
        'camera_dist':    dist.astype(np.float64),
    }


def frame_index(path: str) -> int:
    return int(os.path.basename(path).split('_')[0])


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _fmt_val(val_m: float | None, gt_mm: float | None) -> str:
    if val_m is None:
        return 'n/a'
    s = f'{val_m * 1000:.1f} mm'
    if gt_mm is not None:
        err = val_m * 1000 - gt_mm
        s += f'  | GT error: {err:+.1f} mm ({err / gt_mm * 100:+.1f}%)'
    return s


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Body measurements from SMPLest-X results using ArUco marker slice heights'
    )
    parser.add_argument('results_dir', help='Path to demo/results/{VIDEO_NAME}/')
    parser.add_argument('--garment_config', required=True,
                        help='Path to garment layout JSON (e.g. garment_v1.json)')
    parser.add_argument('--calibration_npz', default=None,
                        help='Camera calibration .npz with keys K, dist. Override for per-npz camera_K.')
    _repo_root = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument('--model_path',
                        default=os.path.join(_repo_root, 'human_models', 'human_model_files'))
    parser.add_argument('--gender', default='neutral', choices=['neutral', 'male', 'female'])
    parser.add_argument('--sigma_threshold', type=float, default=2.0,
                        help='Betas outlier rejection Z-score threshold (default 2.0)')
    parser.add_argument('--ground_truth', default=None, metavar='H,C,W,HP,SH,AR',
                        help='Ground truth in mm: height,chest,waist,hips,shoulders,arm_span')
    args = parser.parse_args()

    # ground truth
    _GT_KEYS = ('height', 'chest', 'waist', 'hips', 'shoulder_width', 'arm_span')
    ground_truth: dict[str, float] | None = None
    if args.ground_truth is not None:
        try:
            vals = [float(x) for x in args.ground_truth.split(',')]
            if len(vals) != len(_GT_KEYS):
                raise ValueError
        except ValueError:
            sys.exit(f'--ground_truth must be exactly {len(_GT_KEYS)} comma-separated numbers')
        ground_truth = dict(zip(_GT_KEYS, vals))

    # garment config
    with open(args.garment_config) as f:
        garment_config = json.load(f)

    # calibration override
    K_override = dist_override = None
    if args.calibration_npz:
        _calib = np.load(args.calibration_npz)
        K_override    = _calib['K'].astype(np.float64)
        dist_override = np.zeros(5, dtype=np.float64)  # assume pre-undistorted

    # discover .npz files
    smplx_dir = os.path.join(args.results_dir, 'smplx')
    npz_files = sorted(glob.glob(os.path.join(smplx_dir, '*.npz')))
    if not npz_files:
        sys.exit(
            f'No .npz files found under {smplx_dir}\n'
            'Run inference first: sh scripts/inference.sh smplest_x_h myvideo.mp4 30'
        )
    print(f'Loading {len(npz_files)} detections from {smplx_dir}')

    # load model
    model = load_smplx(args.model_path, args.gender)
    model.eval()
    faces = model.faces

    # load all per-frame data
    per_frame_data: list[dict] = []
    all_betas: list[np.ndarray] = []
    for path in tqdm(npz_files, desc='Loading .npz files'):
        fd = load_frame_data(path, K_override, dist_override)
        per_frame_data.append(fd)
        all_betas.append(fd['betas'])
    all_betas_arr = np.array(all_betas)

    # check calibration availability
    has_calibration = np.any(per_frame_data[0]['camera_K'] != 0)
    if not has_calibration:
        print('\nWARNING: No camera calibration found in .npz files and no --calibration_npz provided.')
        print('  Marker 3D estimation requires metric camera intrinsics.')
        print('  Re-run inference with --calibration_npz, or pass --calibration_npz here.')
        print('  All bands will fall back to joint-derived slice heights.\n')

    # aggregate marker positions in body-local frame
    aggregated_markers: dict[int, np.ndarray] = {}
    marker_stats: dict[int, dict] = {}
    if has_calibration:
        print('Aggregating marker positions in body-local frame...')
        aggregated_markers, marker_stats = aggregate_markers_in_body_frame(
            per_frame_data, garment_config
        )
        print(f'  {len(aggregated_markers)} unique markers aggregated successfully.')

    # marker-derived slice heights (None per band where insufficient detections)
    band_y_markers = slice_heights_from_markers(aggregated_markers, garment_config)

    # canonical betas (outlier-rejected median)
    cleaned, mask = reject_betas_outliers(all_betas_arr, args.sigma_threshold)
    n_rejected = int((~mask).sum())
    if len(cleaned) == 0:
        print('WARNING: All betas rejected by outlier filter — using full set.')
        cleaned = all_betas_arr
    agg_betas = np.median(cleaned, axis=0)

    # T-pose mesh from canonical betas
    verts, joints = tpose_mesh(model, agg_betas)

    # joint-derived slice heights (used for fallback and comparison)
    y_chest_j = float((joints[SPINE_2, 1] + joints[SPINE_3, 1]) / 2)
    y_waist_j = float(joints[SPINE_1, 1])
    y_hips_j  = float((joints[L_HIP, 1] + joints[R_HIP, 1]) / 2)
    joint_heights = {'chest': y_chest_j, 'waist': y_waist_j, 'hips': y_hips_j}

    # resolve final slice heights: markers where reliable, joints as fallback
    final_heights: dict[str, float] = {}
    height_sources: dict[str, str] = {}
    for band in ('chest', 'waist', 'hips'):
        marker_y = band_y_markers.get(band)
        if marker_y is not None:
            final_heights[band] = marker_y
            height_sources[band] = 'markers'
        else:
            final_heights[band] = joint_heights[band]
            height_sources[band] = 'joints'

    # sanity check: chest > waist > hips in Y (SMPL-X Y-up)
    if not (final_heights['chest'] > final_heights['waist'] > final_heights['hips']):
        print('ERROR: band ordering violation — chest_y must be > waist_y > hips_y')
        print(f"  chest={final_heights['chest']:.4f}  waist={final_heights['waist']:.4f}  hips={final_heights['hips']:.4f}")
        print('  Check garment orientation and marker ID mapping.')
        sys.exit(1)

    # compute measurements
    height_m       = float(verts[:, 1].max() - verts[:, 1].min())
    chest_m        = cross_section_perimeter(verts, faces, final_heights['chest'])
    waist_m        = cross_section_perimeter(verts, faces, final_heights['waist'])
    hips_m         = cross_section_perimeter(verts, faces, final_heights['hips'])
    shoulder_m     = float(abs(joints[L_SHOULDER, 0] - joints[R_SHOULDER, 0]))
    arm_span_m     = float(abs(joints[L_WRIST, 0] - joints[R_WRIST, 0]))

    # joint-only versions (for comparison column)
    chest_j_m      = cross_section_perimeter(verts, faces, y_chest_j)
    waist_j_m      = cross_section_perimeter(verts, faces, y_waist_j)
    hips_j_m       = cross_section_perimeter(verts, faces, y_hips_j)

    # ---------------------------------------------------------------------------
    # Output
    # ---------------------------------------------------------------------------
    video_name = os.path.basename(os.path.normpath(args.results_dir))
    lines: list[str] = []

    def emit(s: str = '') -> None:
        print(s)
        lines.append(s)

    emit(f'\n=== SMPLest-X + ArUco Marker Measurements  [{video_name}  gender={args.gender}] ===')
    emit(f'    Betas: {n_rejected}/{len(all_betas_arr)} detections rejected at σ>{args.sigma_threshold}')

    # --- slice height comparison table ---
    emit('\n--- Slice heights (body-local Y, metres) ---')
    emit(f"  {'band':<8}  {'marker':<12}  {'joints':<12}  source")
    emit(f"  {'-'*8}  {'-'*12}  {'-'*12}  {'-'*6}")
    for band in ('chest', 'waist', 'hips'):
        m_val = f"{band_y_markers[band]:.4f}" if band_y_markers[band] is not None else 'n/a'
        j_val = f"{joint_heights[band]:.4f}"
        src   = f"[{height_sources[band]}]"
        emit(f"  {band:<8}  {m_val:<12}  {j_val:<12}  {src}")

    # --- marker reliability table ---
    emit('\n--- Marker reliability ---')
    emit(f"  {'id':<6}  {'band':<8}  {'position':<14}  {'n_frames':<10}  {'std_mm':<10}  status")
    emit(f"  {'-'*6}  {'-'*8}  {'-'*14}  {'-'*10}  {'-'*10}  {'-'*8}")
    for mid_str, m_info in sorted(garment_config['markers'].items(), key=lambda x: int(x[0])):
        mid = int(mid_str)
        if mid in marker_stats:
            s = marker_stats[mid]
            status = 'REJECTED' if s['rejected'] else 'ok'
            emit(f"  {mid:<6}  {m_info['band']:<8}  {m_info['position']:<14}  "
                 f"{s['n_frames']:<10}  {s['std_mm']:<10.1f}  {status}")
        else:
            emit(f"  {mid:<6}  {m_info['band']:<8}  {m_info['position']:<14}  "
                 f"{'0':<10}  {'n/a':<10}  not detected")

    # --- final measurements ---
    emit('\n--- Canonical measurements ---')
    emit(f"  {'measurement':<22}  {'marker-path':<30}  {'joints-path':<30}")
    emit(f"  {'-'*22}  {'-'*30}  {'-'*30}")

    gt = ground_truth or {}

    def _row(name: str, marker_val: float | None, joints_val: float | None) -> None:
        src_tag = f' [{height_sources[name]}]' if name in height_sources else ''
        label = name + src_tag
        emit(f"  {label:<22}  {_fmt_val(marker_val, gt.get(name)):<30}  "
             f"{_fmt_val(joints_val, gt.get(name))}")

    _row('height',         height_m,   height_m)     # same — height doesn't depend on slice
    _row('chest',          chest_m,    chest_j_m)
    _row('waist',          waist_m,    waist_j_m)
    _row('hips',           hips_m,     hips_j_m)
    _row('shoulder_width', shoulder_m, shoulder_m)   # same — joint position, not slice
    _row('arm_span',       arm_span_m, arm_span_m)   # same

    # save results
    txt_path = os.path.join(args.results_dir, f'{video_name}_markers.txt')
    with open(txt_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'\nResults saved to {txt_path}')


if __name__ == '__main__':
    main()
