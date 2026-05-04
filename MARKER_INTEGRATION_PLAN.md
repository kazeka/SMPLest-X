# ArUco Markers + Camera Calibration Integration — Planning Doc

> **Audience:** Claude Code, iterating with the engineer.
> **Status:** Sketch / planning — verify all assumptions against the actual SMPLest-X codebase before implementing.
> **Goal:** Add ArUco-marker-based scale anchoring and anatomical-plane fixing to the body measurement pipeline, alongside the SfM track.

---

## 1. Why this exists

Our current pipeline (SMPLest-X → β → SMPL-X mesh in T-pose → horizontal-plane slicing) hits two structural error sources:

1. **Slice-height drift.** Chest/waist/hip slice heights are derived from SMPL-X spine joints, which are skinning anchors, not anatomical landmarks. They drift with body proportions by ±2–3% of stature.
2. **Soft-tissue under-estimation.** SMPL-X mesh vertices smooth over real-body contour detail, especially around the chest and abdomen.

ArUco markers on a calibration garment address (1) directly: if the markers are at known anatomical levels, slicing at the marker height — not the joint height — eliminates the drift entirely. Markers also provide metric scale anchoring independent of the network's learned scale prior.

Markers do **not** address (2) on their own. For that, we need to either fit the SMPL-X mesh more aggressively (Path B below) or move to the SfM track (separate workstream).

---

## 2. Two paths, in priority order

### Path A — Marker-direct slicing (do this first)

Use markers to **define measurement planes**, slice the existing SMPLest-X mesh at those planes. No optimization, no mesh refitting. Fast, low risk, captures the bigger of the two error sources.

Engineering cost: ~1 week.
Expected improvement: chest/waist/hip absolute errors should drop from ~30–80 mm to ~20–40 mm.

### Path B — Optimization-based refinement with marker constraints (second iteration)

Treat SMPLest-X output as initialization. Run L-BFGS on a multi-term loss (keypoint reprojection + marker reprojection + silhouette IoU + priors) to refine β, θ, t per-frame, with β shared across frames of one capture. Adds the full classic SMPLify recipe.

Engineering cost: ~3–4 weeks.
Expected improvement: marginal beyond Path A unless silhouette + keypoint terms are well-tuned. Probably not worth pursuing if SfM track delivers in the same timeframe.

> **Recommendation: build Path A, evaluate, then decide whether Path B is worth pursuing or whether SfM supersedes it.**

---

## 3. Calibration garment design (assumed — verify with the engineer)

The pipeline requires the engineer to define the marker layout up-front. Until that's confirmed, build against this **assumed design** and parameterize it so it can change:

- Tight-fitting top + leggings
- ArUco markers (DICT_4X4_50, 30mm × 30mm) printed on adhesive patches
- Three horizontal bands of markers at known body levels:
  - **Chest band:** 4 markers at the nipple line / Spine_3 anatomical level
  - **Waist band:** 4 markers at the natural waist / iliac crest level
  - **Hip band:** 4 markers at the maximum buttock circumference / greater trochanter level
- Markers within a band are evenly spaced (90° apart, front / back / each side)
- Each marker has a unique ID; IDs are mapped to (band, position) in a JSON config

**Required for the pipeline:** known **3D positions of all markers in body-local SMPL-X coordinates** when the subject is in T-pose. This is the calibration data the engineer needs to provide. For now, treat it as a JSON file:

```json
{
  "garment_id": "v1",
  "markers": {
    "0":  {"band": "chest", "position": "front",       "size_mm": 30},
    "1":  {"band": "chest", "position": "right_side",  "size_mm": 30},
    "2":  {"band": "chest", "position": "back",        "size_mm": 30},
    "3":  {"band": "chest", "position": "left_side",   "size_mm": 30},
    "4":  {"band": "waist", "position": "front",       "size_mm": 30},
    "...": "..."
  },
  "bands": {
    "chest": {"smplx_y_offset_metres": null, "comment": "Determined per-subject from first calibration frame"},
    "waist": {"smplx_y_offset_metres": null},
    "hips":  {"smplx_y_offset_metres": null}
  }
}
```

> **TODO for engineer:** confirm marker count, layout, IDs, and whether band Y-offsets are known up-front (e.g. measured in metres above the floor when subject stands) or determined per-subject from the data.

---

## 4. Modifications to `inference.py`

These changes are **shared between Path A and Path B** — both paths need the same enriched per-frame data saved to disk. Make these changes first; they are non-disruptive (existing downstream consumers keep working, new fields are additive).

### 4.1 Save extra fields in the per-frame `.npz`

The current `np.savez` call saves three keys: `betas`, `cam_trans`, `vertices`. We need more.

```python
# CURRENT
np.savez(
    osp.join(smplx_dir, f'{int(frame):06d}_{bbox_id}.npz'),
    betas=out['smplx_shape'].detach().cpu().numpy()[0],
    cam_trans=out['cam_trans'].detach().cpu().numpy()[0],
    vertices=mesh,
)

# NEW
np.savez(
    osp.join(smplx_dir, f'{int(frame):06d}_{bbox_id}.npz'),
    # --- existing ---
    betas=out['smplx_shape'].detach().cpu().numpy()[0],
    cam_trans=out['cam_trans'].detach().cpu().numpy()[0],
    vertices=mesh,
    # --- pose params, needed for Path B (optimizer initialization) ---
    body_pose=out['smplx_body_pose'].detach().cpu().numpy()[0],          # TODO: verify key
    global_orient=out['smplx_root_pose'].detach().cpu().numpy()[0],      # TODO: verify key
    # --- camera, needed for both paths (back-projection of markers) ---
    bbox=bbox,                                                           # processed bbox xywh
    bbox_xyxy=yolo_bbox[bbox_id],                                        # original detection
    focal=np.array(focal),                                               # already computed
    princpt=np.array(princpt),                                           # already computed
    # --- camera calibration (see §4.3) ---
    camera_K=K if K is not None else np.zeros((3, 3)),
    camera_dist=dist if dist is not None else np.zeros(5),
    # --- markers (see §4.2) ---
    markers_ids=marker_ids,           # int array, shape (N,)
    markers_corners=marker_corners,   # float array, shape (N, 4, 2) — pixel coords
    # --- provenance ---
    img_path=img_path,
    img_shape=np.array(original_img.shape[:2]),  # (H, W)
    frame_idx=frame,
)
```

> **TODO for Claude Code:** verify the exact key names that SMPLest-X's model output produces. Run `print(out.keys())` in a one-off inference call before committing. Do not assume — the keys above (`smplx_body_pose`, `smplx_root_pose`) are educated guesses based on SMPLer-X conventions but may differ in SMPLest-X.

### 4.2 ArUco detection per-frame

Add ArUco detection alongside YOLO person detection. Both run on the same `original_img`.

```python
# Near top of main(), after detector init:
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()
aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

# Inside the per-frame loop, after load_img(img_path):
gray = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)
corners_raw, ids_raw, _ = aruco_detector.detectMarkers(gray)

if ids_raw is not None:
    marker_ids = ids_raw.flatten().astype(np.int32)               # (N,)
    marker_corners = np.stack([c.squeeze() for c in corners_raw]) # (N, 4, 2)
else:
    marker_ids = np.zeros((0,), dtype=np.int32)
    marker_corners = np.zeros((0, 4, 2), dtype=np.float32)
```

This is cheap (~5–10 ms / frame on CPU) and runs once per frame regardless of how many people are detected. Markers are saved to every per-person `.npz` for that frame; downstream code filters by which markers fall inside which person's bbox.

### 4.3 Camera calibration loading

We already have `undistort_images.py` and per-device calibration `.npz` files. The current setup undistorts as a separate ETL step before frames arrive in `input_frames/`. **Don't undistort twice** — pick one location:

**Option 1 (recommended):** keep undistortion as a pre-step (current state). Just load the calibration `.npz` in `inference.py` to know `K` and `dist` (with `dist` being all zeros since frames are pre-undistorted). Save them in each frame's npz so downstream code has metric camera intrinsics.

**Option 2:** move undistortion into `inference.py`. Saves a pipeline stage but couples concerns.

Going with Option 1:

```python
# Add CLI arg:
parser.add_argument('--calibration_npz', type=str, default=None,
                    help='Path to camera calibration .npz with keys K, dist, image_size.')

# Near top of main(), after parsing:
if args.calibration_npz:
    calib = np.load(args.calibration_npz)
    K = calib['K']
    # dist is zeros if frames have been pre-undistorted
    dist = np.zeros(5) if args.assume_undistorted else calib['dist']
else:
    # Fall back to the rendered camera intrinsics (per-bbox), not metric.
    K = None
    dist = None
```

> **Note:** the `focal`/`princpt` already computed in the existing code are SMPLest-X's *rendered* camera (a virtual camera matched to the bbox crop), not the physical camera that captured the frame. Both are needed: rendered camera for back-projection of mesh, physical camera for back-projection of markers. Save both, label clearly.

---

## 5. Path A — Marker-direct slicing

A new script, `measure_with_markers.py`. Sits alongside the existing `measure_bodies.py` and shares its model-loading and aggregation code.

### 5.1 High-level flow

For each capture (folder of per-frame `.npz` files):

1. **Load all per-frame data** including markers and camera intrinsics.
2. **Estimate marker 3D positions** in camera coordinates by treating each marker as a planar square of known physical size (30mm). OpenCV's `cv2.solvePnP` with the marker corners and a known 4-point template gives R, t per marker.
3. **Aggregate marker positions across frames:** for a stationary subject, each marker ID has a 3D position that should be approximately constant in body-local coordinates. Use the mean SMPLest-X cam_trans and global rotation to bring marker camera-coords into body-local coords, then take the median across frames per marker ID.
4. **Project marker positions onto SMPL-X T-pose mesh:** use the median betas, run the mesh in T-pose, transform marker positions through the same T-pose-equivalent transform. Now markers are in T-pose mesh coordinates.
5. **Determine slice heights from markers:** each band's Y-coordinate is the mean Y of its 4 markers in T-pose mesh coords. Replaces joint-derived slice heights from `measure_bodies.py`.
6. **Slice the mesh and compute perimeters** as before, but at marker-determined Y values instead of joint-derived ones.

### 5.2 Skeleton

```python
"""
measure_with_markers.py

Body measurements (mm) from SMPLest-X results, with anatomical slice heights
determined by ArUco markers on a calibration garment instead of derived from
SMPL-X spine joints.

Usage:
    python measure_with_markers.py demo/results/myvideo/ \
        --garment_config garment_v1.json \
        --calibration_npz p50pro_main_12MP.npz
"""

import json, glob, os, sys
import numpy as np
import cv2
import torch
from tqdm import tqdm

from measure_bodies import (
    load_smplx, tpose_mesh, cross_section_perimeter,
    reject_betas_outliers, _contour_perimeters,
)


def estimate_marker_pose_camera_frame(corners_2d, marker_size_mm, K, dist):
    """
    Given 4 image-pixel corners of a single ArUco marker and known marker
    physical size, return (R, t) of the marker frame in camera coordinates.

    Marker coordinate convention: origin at marker centre, +X right, +Y down,
    +Z out of the marker plane (right-hand rule).
    """
    half = marker_size_mm / 2000.0  # mm → metres, half side length
    # Marker corner order from cv2.aruco: top-left, top-right, bottom-right, bottom-left
    object_points = np.array([
        [-half, -half, 0],
        [+half, -half, 0],
        [+half, +half, 0],
        [-half, +half, 0],
    ], dtype=np.float32)
    success, rvec, tvec = cv2.solvePnP(
        object_points, corners_2d.astype(np.float32), K, dist,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    if not success:
        return None, None
    R, _ = cv2.Rodrigues(rvec)
    return R, tvec.flatten()  # (3, 3), (3,)


def marker_centre_camera_frame(corners_2d, marker_size_mm, K, dist):
    """Return marker centre 3D position in camera frame, or None on failure."""
    R, t = estimate_marker_pose_camera_frame(corners_2d, marker_size_mm, K, dist)
    return t  # marker origin is its centre, so t IS the centre


def transform_camera_to_smplx_local(point_cam, smplx_global_orient, smplx_cam_trans):
    """
    Convert a 3D point from camera coordinates to SMPL-X body-local coordinates
    (the canonical T-pose frame of the regressed body for this frame).

    SMPLest-X outputs `cam_trans` (translation of pelvis in camera frame) and
    `global_orient` (rotation of pelvis in camera frame, axis-angle). Inverting
    that transform takes camera-frame points into body-local frame.
    """
    R_global, _ = cv2.Rodrigues(np.array(smplx_global_orient).reshape(3, 1))
    # Body-to-camera: x_cam = R_global @ x_body + cam_trans
    # Camera-to-body: x_body = R_global.T @ (x_cam - cam_trans)
    return R_global.T @ (point_cam - smplx_cam_trans)


def aggregate_markers_in_body_frame(per_frame_data, garment_config, K, dist):
    """
    For each marker ID, collect its body-local 3D position across all frames
    and return the median. Returns dict: marker_id -> position (3,).
    """
    marker_positions_per_id = {}  # int → list of (3,) arrays

    marker_size_mm = 30.0  # TODO: read from garment_config

    for frame_data in per_frame_data:
        ids = frame_data['markers_ids']
        corners = frame_data['markers_corners']
        cam_trans = frame_data['cam_trans']
        global_orient = frame_data['global_orient']

        for marker_idx in range(len(ids)):
            mid = int(ids[marker_idx])
            if str(mid) not in garment_config['markers']:
                continue  # marker not part of garment

            centre_cam = marker_centre_camera_frame(
                corners[marker_idx], marker_size_mm, K, dist
            )
            if centre_cam is None:
                continue

            centre_body = transform_camera_to_smplx_local(
                centre_cam, global_orient, cam_trans
            )
            marker_positions_per_id.setdefault(mid, []).append(centre_body)

    # Take median per marker, with sanity-check on cluster tightness
    aggregated = {}
    for mid, positions in marker_positions_per_id.items():
        positions = np.array(positions)  # (N, 3)
        median = np.median(positions, axis=0)
        std = np.std(positions, axis=0)
        # Reject markers whose per-frame positions don't cluster (likely
        # detection drift or wrong-body-frame transform on bad frames).
        if np.linalg.norm(std) > 0.05:  # 5cm std is the rejection threshold
            print(f"  warning: marker {mid} has high cross-frame std ({std}); rejecting")
            continue
        aggregated[mid] = median

    return aggregated


def slice_heights_from_markers(aggregated_markers, garment_config):
    """
    For each band (chest, waist, hips), compute the slice Y-height as the
    mean Y of the band's markers in body-local coordinates.

    Falls back to None for bands with insufficient markers; caller decides
    whether to fall back to joint-derived heights.
    """
    band_y = {}
    for band_name in ('chest', 'waist', 'hips'):
        ys = []
        for mid_str, m_info in garment_config['markers'].items():
            if m_info['band'] != band_name:
                continue
            mid = int(mid_str)
            if mid in aggregated_markers:
                ys.append(aggregated_markers[mid][1])  # Y component, SMPL-X up-axis
        if len(ys) >= 2:  # Need at least 2 of the 4 to estimate the band
            band_y[band_name] = float(np.mean(ys))
        else:
            band_y[band_name] = None
            print(f"  warning: only {len(ys)} markers found for {band_name} band; "
                  "falling back to joint-derived height")
    return band_y


def main():
    # parse args (same as measure_bodies.py + new --garment_config and --calibration_npz)
    # load garment config JSON
    # load camera calibration
    # load all per-frame .npz files
    # extract markers, betas, cam_trans, global_orient per frame
    # aggregate markers in body-local frame across frames
    # determine slice heights from markers (with joint fallback)
    # take median betas across frames (same as measure_bodies.py)
    # run T-pose mesh from median betas
    # slice mesh at marker-derived Y values
    # report measurements
    ...


if __name__ == "__main__":
    main()
```

### 5.3 Sanity checks Path A should run

These should be implemented as part of the script, not optional:

- **Marker count per band:** if a band has fewer than 2 detected markers across the capture, fall back to joint-derived height for that band and warn loudly.
- **Marker cluster tightness:** the std of a marker's position across frames should be <50mm in body-local frame. If higher, reject and report.
- **Band ordering:** chest_y > waist_y > hips_y in SMPL-X (Y-up) coords. If the order is wrong, the calibration garment is upside down or marker IDs are mis-mapped — abort and report.
- **Sanity vs. joint-derived:** print both the marker-derived and the joint-derived slice heights side by side for each band so the engineer can spot drift quickly.

### 5.4 Expected output

Same six measurements as `measure_bodies.py`, but with:
- A "marker-derived slice heights" section in the output showing chest/waist/hips Y values from markers vs. from joints.
- A per-marker reliability table: marker ID, band, n_frames detected, position std.
- A flag column on each measurement: `[markers]` if it used marker-derived height, `[joints]` if it fell back.

---

## 6. Path B — Optimization-based refinement (deferred)

If Path A delivers the expected ~30-40% improvement and SfM is delayed, Path B becomes the next investment. **Don't build this until Path A is in production.**

### 6.1 Architecture

A new script `fit_with_markers.py`. Consumes the same per-frame `.npz` files. For each capture:

1. **Initialize** β, θ, t from SMPLest-X output (already saved in npz).
2. **Define optimizable parameters:** β (10 dims, shared across frames), per-frame θ (~66 dims), per-frame t (3 dims). For a 1000-frame capture this is ~70k parameters — manageable for L-BFGS but slow.
3. **Run optimization** with multi-term loss:
   - `L_marker`: reproject SMPL-X mesh points (from known marker-to-vertex correspondences) into image, MSE against detected ArUco corners. **Highest weight**: this is what markers buy us.
   - `L_keypoint`: reproject SMPL-X joints, MSE against 2D keypoint detections (need to add a keypoint detector — MMPose or MediaPipe).
   - `L_shape_prior`: ‖β‖² regularization, low weight.
   - `L_pose_prior`: ‖θ‖² or VPoser, prevents impossible poses.
   - `L_temporal`: ‖θ_t − θ_{t-1}‖, smoothness across frames.
4. **Coarse-to-fine:** first stage fits θ + t with β fixed at SMPLest-X output (pose alignment). Second stage unlocks β with marker + keypoint + silhouette. Final stage refines all jointly with reduced LR.

### 6.2 Marker-to-vertex correspondences

This is the missing piece. Each ArUco marker on the garment overlies a specific region of the SMPL-X mesh. We need a JSON mapping:

```json
{
  "marker_to_vertex": {
    "0": [4521, 4522, 4523, 4524],  // 4 nearest SMPL-X vertex IDs
    "1": [4598, 4599, 4600, 4601],
    "...": "..."
  }
}
```

This is determined **once, manually** by:
1. Capturing one calibration session in known T-pose with the garment on
2. Running SMPLest-X to get the mesh
3. For each marker's detected 3D position in camera frame, finding the nearest mesh vertices
4. Recording the correspondences

It's a one-time cost per garment design.

### 6.3 Why Path B may not be worth it

- **Marker reprojection alone may be redundant** with Path A's slice-height fix — once slice heights are right, the remaining error is soft-tissue, which optimization on β can only partially address.
- **Adding a keypoint detector** is its own engineering project (MMPose installation, version compatibility with SMPLest-X's mmcv).
- **Silhouette term needs a differentiable renderer** (PyTorch3D or nvdiffrast). More dependencies, more dragons.
- **Per-frame optimization** is slow: ~5–30 seconds/frame on GPU. For 1000-frame captures this turns a 5-minute pipeline into a 1–4 hour pipeline.

The honest assessment: Path B has good ROI only if (a) Path A leaves a meaningful gap to AOF tolerance AND (b) SfM track is delayed/cancelled. If SfM ships, it dominates Path B on accuracy.

---

## 7. Coordinate frames cheat sheet

This is where bugs hide. Be explicit about which frame each quantity is in.

| Frame | Origin | Axes | Used for |
|---|---|---|---|
| **Image** | Top-left pixel | (u, v), pixels | YOLO bbox, ArUco corners |
| **Camera** | Optical centre | +Z forward (into scene), +X right, +Y down (OpenCV convention) | `cv2.solvePnP` output, `cam_trans` |
| **SMPL-X body-local** | Pelvis joint | +Y up, +X left (subject's left), +Z forward (out of subject's chest) | T-pose mesh vertices, joint positions |
| **SMPL-X world (posed)** | Same as body-local but rotated by `global_orient` and translated by `cam_trans` | Per-frame, follows camera | Posed mesh in inference.py |

Going from camera to body-local: `x_body = R_global.T @ (x_cam - cam_trans)`.
Going from body-local to camera: `x_cam = R_global @ x_body + cam_trans`.

Marker-on-garment positions are stable in **body-local** frame for a given subject (same shirt, same fit). They are NOT stable in camera frame because the subject moves. Always aggregate markers in body-local.

---

## 8. Open questions for the engineer

Things the engineer needs to decide / provide before either path can ship:

1. **Garment design:** marker count, layout, IDs, physical size, band positions. Sketch of garment with marker placements is needed.
2. **Garment-to-vertex correspondences (Path B only):** one-time calibration session + manual mapping.
3. **Calibration trust:** is the camera calibration (`p50pro_main_12MP.npz`) trusted enough that we should rely on its `K` for `cv2.solvePnP`, or should we re-solve the camera per-capture?
4. **Frame undistortion location:** keep the current `undistort_images.py` pre-step, or move into `inference.py`? (Recommend keeping pre-step.)
5. **SMPLest-X output keys:** verify `smplx_body_pose` and `smplx_root_pose` are the actual key names, or what they're called instead.
6. **Subject calibration video:** do we want a separate "calibration capture" in T-pose where the subject stands still for 5 seconds with the garment on, before the rotation capture? This would be the cleanest source of ground-truth marker positions.

---

## 9. Suggested implementation order

1. **Modify `inference.py` per §4** — additive only, ships independently. Validate by re-running existing captures and confirming new fields appear in `.npz` files. **Half day.**
2. **Define garment config JSON** with placeholder values, so downstream code has a schema to load against. **1 hour.**
3. **Implement `measure_with_markers.py` per §5** without the marker math first — get the script structure, arg parsing, and joint-fallback path working using existing joint-derived heights. **1 day.**
4. **Add ArUco math** (`solvePnP`, body-frame transformation, aggregation). Test on one capture with synthetic markers (manually annotated 4-corner positions in code) before requiring a real calibration garment. **2 days.**
5. **Capture real calibration garment data** with the actual marker layout. Iterate on the script with real detections. **2 days, blocked on garment.**
6. **Integrate with the report pipeline** so marker-derived runs appear alongside joint-derived runs in the comparison tables. **Half day.**
7. **Path B if needed** — separate sprint, not blocked on Path A's completion but won't be prioritized unless Path A leaves a gap.

Path A end-to-end: ~1 week of engineering once the garment is available.

---

## 10. Out of scope for this doc

- **Phase 2 mobile app:** the marker garment is a Phase 1 R&D tool. The production app uses a different (likely markerless or different-marker) approach. Don't conflate.
- **SfM track:** parallel workstream. The marker work here informs but doesn't depend on SfM, and vice versa. SfM benefits from the same camera calibration but uses dense reconstruction rather than parametric body fitting.
- **Multi-subject support:** all current scripts assume one subject per capture. Not a near-term blocker.
