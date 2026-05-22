# Marker-Only Body Measurement Pipeline — Planning Doc

> **Audience:** Claude Code, iterating with the engineer.
> **Status:** Draft — significant departure from the existing SMPLer-X / SMPLest-X pipeline. Verify capture-protocol assumptions with the engineer before implementing.
> **Goal:** Body measurements (chest / waist / hip circumferences) computed entirely from ArUco markers on a calibration garment, with no parametric body model in the pipeline.

---

## 1. Why this exists, vs. the SMPL+marker hybrid

The existing tracks all keep SMPL-X in the pipeline as the source of body geometry:

- `measure_bodies.py` — joint-derived slice heights, SMPL-X mesh circumference
- `measure_with_markers.py` (Path A in `MARKER_INTEGRATION_PLAN.md`) — marker-anchored slice heights, **SMPL-X mesh** circumference
- `fit_with_markers.py` (Path B) — markers as one term in an optimization that refines the SMPL-X mesh

In all three, the circumference comes from slicing a SMPL-X mesh. SMPL-X has known structural limitations:

- Smooth skin assumption that under-resolves chest, abdomen, breast tissue
- Soft-tissue under-estimation systematically biases circumference low (~5% on chest in our data)
- Shape regression from monocular video has a hard accuracy floor (~60–80 mm per-vertex error)

A marker-only pipeline removes SMPL-X from the measurement path entirely. The measurements come from the **physical positions of markers on the garment**, recovered geometrically via solvePnP on each detected marker. No neural network, no body model, no betas.

### Trade-offs vs. SMPL-anchored approaches

**Wins:**
- No parametric model bias on circumference (you measure where the markers are, not where SMPL-X thinks the body is)
- No GPU inference — entire pipeline runs on CPU
- No model dependency, no checkpoint management, no version drift
- Failure modes are interpretable: missing marker = missing data, not silently wrong number
- Per-subject calibration is unnecessary (the metric scale comes from marker physical size)

**Losses:**
- **Camera calibration is mandatory.** No rendered-camera fallback because there's no SMPL-X to provide one.
- **Only the measurements you place markers for.** No height, no arm length, no anything outside the garment. To match the AOF six-measurement set you need markers for each of: chest, waist, hips, shoulders (biacromial), arm length per side. Height and arm span need separate measurement.
- **Marker density determines accuracy.** 4 markers per band is the minimum; 8+ per band gives meaningfully better circumference reconstruction.
- **Pose constraints.** The pipeline assumes a recoverable rigid body frame (see §3 below) — subject can rotate but should not bend torso, lift arms unevenly, etc. Looser than T-pose, stricter than "free movement."

### When to pick this over Path A

Pick marker-only if:
- Path A has been tried and the residual error is dominated by SMPL-X shape bias rather than slice placement
- You need a fallback that doesn't depend on the SMPLest-X stack (license, GPU availability, model maintenance)
- You're willing to invest in a denser marker garment (8+ per band) and per-camera calibration

Pick Path A (SMPL+marker) if:
- You want measurements outside the bands (height, arm length, shoulder breadth) without adding more markers
- The garment design is constrained to ~4 markers per band
- You need a quick win and are happy with ~30–40 mm circumference error

The two are not exclusive — the same garment can feed both pipelines and produce a per-measurement comparison for the Sprint 3 tech-selection report.

---

## 2. Pipeline overview

```
   video frames
        │
        ▼
   undistort (cv2.undistort with calibrated K, dist)
        │
        ▼
   per-frame ArUco detection (DICT_4X4_50)
        │
        ▼
   per-marker solvePnP → 3D pose in physical camera frame
        │
        ▼
   identify reference triad markers (always the same IDs)
        │
        ▼
   build body-local frame from triad → transform all markers into it
        │
        ▼
   aggregate per-marker positions across frames (median + outlier rejection)
        │
        ▼
   per-band: collect markers in band → ellipse fit → circumference
        │
        ▼
   apply per-measurement calibration correction (vs. ground-truth tape)
        │
        ▼
   report: chest / waist / hip circumferences (mm)
```

No SMPLest-X anywhere. No betas, no rendered camera, no `cam_trans`. The only ML in the pipeline is whatever produces the calibration `.npz` (one-time checkerboard calibration).

---

## 3. The reference-triad design

This is the central design choice that makes the pipeline work. It's worth spending time on.

### 3.1 The problem

Without SMPL-X providing a body-local frame (`cam_trans` + `global_orient`), we have no way to transform per-frame marker positions into a stable reference frame. As the subject rotates, each marker's position in camera frame traces an arc, and naively averaging across frames produces a meaningless cylindrical smear.

### 3.2 The solution

Designate **3 markers placed on a rigid body region** as a reference triad. They define a body-local coordinate frame at every frame. Other markers are transformed into that frame and aggregated there.

The triad must be:

1. **On a rigid region of the body** — somewhere that doesn't deform with breathing, posture shifts, or arm movement. Best candidates: upper chest (above the breast tissue, near the sternal manubrium), lower back (above the lumbar curve), lateral upper hip.
2. **Non-collinear and non-coplanar with respect to the body's symmetry plane** — three markers in a row don't define a unique frame. Triangle, with one marker offset front-back from the other two.
3. **Uniquely identified** — fixed ArUco IDs reserved for the triad, never reused for measurement bands.

Recommended placement for a v1 garment (open to discussion):

- **Triad-A** — upper sternum, ~3 cm below clavicle notch, slightly off-centre to clear the bone ridge
- **Triad-B** — left scapula, ~5 cm lateral to spine, between scapula and rib
- **Triad-C** — right scapula, mirror of Triad-B

This gives a triangle in the upper torso that's stable through arm motion, breathing, and rotation. The "front" marker is Triad-A, the back two are Triad-B and Triad-C. The body-local frame is constructed:

```
origin = centroid of (A, B, C)
y_axis = "up" — defined by an external up reference (gravity from accelerometer
         metadata if available, or a 4th calibration marker on a separate band
         placed directly above the triad)
x_axis = (B - C) normalised  (lateral, subject's right to left)
z_axis = y_axis × x_axis     (forward, into chest)
```

> **Open question:** if no external up reference exists, the y-axis must be derived. One option is to require the subject to start every capture in a known T-pose for ~2 seconds, average the triad orientation in camera frame over that interval, and use it as the body-up reference. Then maintain the orientation across frames by tracking the triad's relative rotation. Worth discussing with the engineer.

### 3.3 Triad robustness

In any frame, all 3 triad markers must be detected to construct the frame. Frames missing one or more triad markers are dropped (not used for aggregation). With 3 well-placed markers on the upper torso, detection rate during a 360° rotation is typically >70%; if it drops below that, add a 4th triad marker on the front-lower torso as a fallback.

The triad markers must be at **larger physical size** than band markers — recommend 50mm × 50mm vs. 30mm for band markers — to reduce solvePnP noise on the frame definition. Frame noise propagates into every other marker's body-local position.

---

## 4. Capture protocol

The protocol is **stricter** than for SMPL-based pipelines because we cannot rely on a body model to absorb pose noise. The protocol must produce:

1. **Static initial T-pose / A-pose calibration segment (5 seconds):** subject stands facing camera, arms slightly out, no movement. Used to establish initial triad orientation and as a quality-control reference for triad noise.
2. **Slow 360° rotation (30–40 seconds):** subject rotates clockwise on the spot, ~10°/sec, maintaining torso uprightness. Arms can drop to A-pose during rotation but should not swing.
3. **Static end T-pose (3 seconds):** repeat of (1), used for end-to-end consistency check (initial vs. final triad orientation should match).

Distance, lighting, framing: same as SMPL pipelines. Camera distance 2.5–3.5 m, even soft lighting, full body in frame, fixed tripod.

### 4.1 What this protocol does NOT support

- Subject walking or moving translationally (triad orientation tracking assumes stationary rotation)
- Multiple simultaneous subjects (triad ID space is per-subject)
- Loose clothing over the calibration garment (markers must be visible)
- Sitting, kneeling, or other non-upright poses (band Y-heights are defined relative to body up-axis)

For the AOF use case these constraints are fine. Document them as protocol requirements in the user-facing app.

---

## 5. Measurement bands

### 5.1 Marker count per band

For circumference reconstruction, marker count is the dominant quality lever.

- **4 markers per band** — minimum viable, requires ellipse-fit assumption (see §6.2). Expected error: 5–10% on chest, dominated by chest cross-section being non-elliptical.
- **8 markers per band** — recommended for v1. Allows spline reconstruction (see §6.3). Expected error: 2–4% on chest.
- **12+ markers per band** — diminishing returns; sub-2% accuracy possible but garment manufacturing complexity rises sharply.

Recommend 8 markers per band as the design target.

### 5.2 Band placement

Same anatomical rules as in `MARKER_INTEGRATION_PLAN.md` §3.1:

- Same Y-height within a band (±10 mm tolerance)
- Distributed around the body (~45° apart for 8 markers)
- On flat torso panels, not bony ridges or crease lines
- Front/back labels are advisory, not enforced geometric constraints

### 5.3 Garment config schema

```json
{
  "garment_id": "marker_only_v1",
  "reference_triad": {
    "marker_ids": [100, 101, 102],
    "size_mm": 50,
    "comment": "Upper torso rigid region: sternum + L/R scapulae"
  },
  "bands": {
    "chest": {
      "marker_ids": [10, 11, 12, 13, 14, 15, 16, 17],
      "marker_size_mm": 30,
      "anatomical_level": "nipple line / 4th intercostal space"
    },
    "waist": {
      "marker_ids": [20, 21, 22, 23, 24, 25, 26, 27],
      "marker_size_mm": 30,
      "anatomical_level": "natural waist / iliac crest"
    },
    "hips": {
      "marker_ids": [30, 31, 32, 33, 34, 35, 36, 37],
      "marker_size_mm": 30,
      "anatomical_level": "max gluteal circumference"
    }
  }
}
```

ID ranges are by-convention (100s = triad, 10s = chest, 20s = waist, 30s = hips). Easy to memorise, easy to validate.

---

## 6. Circumference reconstruction

### 6.1 The problem

After aggregation we have 8 (or 4) 3D points per band, all at approximately the same Y-height in body-local frame. We need a single number: the perimeter of the body cross-section at that height. The 8 points lie *on* the garment surface (approximately on the body surface), so we want to fit a closed curve through them and take its perimeter.

### 6.2 Ellipse fit (4–6 markers per band)

Fit an ellipse to the (X, Z) coordinates of the band markers using least-squares:

```python
# After aggregating markers in body-local frame:
#   band_points: (N, 3) array of marker positions, N >= 4
# Y is body-up; ellipse lives in the X-Z plane.
xz = band_points[:, [0, 2]]
ellipse = cv2.fitEllipse(xz.astype(np.float32))
(cx, cy), (a, b), angle = ellipse  # major/minor axes are 'a' and 'b' (full lengths)

# Ramanujan's approximation for ellipse circumference:
h = ((a - b) ** 2) / ((a + b) ** 2)
circumference = math.pi * (a + b) * (1 + (3 * h) / (10 + math.sqrt(4 - 3 * h)))
```

Notes:
- `cv2.fitEllipse` returns axes as **full lengths**, not semi-axes. Confirm before using in formulas.
- Body cross-sections are roughly elliptical at waist and hip but distinctly non-elliptical at male chest (more rectangular due to pectorals) and female chest (asymmetric due to bust). Expect systematic error here.
- This approach has been validated in similar phone-based body scanning literature; expected accuracy is ~5% on real subjects.

### 6.3 Spline reconstruction (8+ markers per band)

With 8+ markers per band, a closed periodic cubic spline through the points is more accurate:

```python
from scipy.interpolate import CubicSpline

# Order points by angle around the centroid (necessary for spline closure)
xz = band_points[:, [0, 2]]
centroid = xz.mean(axis=0)
angles = np.arctan2(xz[:, 1] - centroid[1], xz[:, 0] - centroid[0])
order = np.argsort(angles)
xz_ordered = xz[order]

# Close the loop by appending the first point at the end
xz_closed = np.vstack([xz_ordered, xz_ordered[:1]])
t = np.linspace(0, 1, len(xz_closed))

# Periodic cubic spline through the points
cs_x = CubicSpline(t, xz_closed[:, 0], bc_type='periodic')
cs_z = CubicSpline(t, xz_closed[:, 1], bc_type='periodic')

# Numerical perimeter via dense sampling
ts = np.linspace(0, 1, 1000)
xs = cs_x(ts)
zs = cs_z(ts)
perimeter = np.sum(np.sqrt(np.diff(xs)**2 + np.diff(zs)**2))
```

Notes:
- Periodic boundary condition ensures the spline is smooth at the closure point.
- 1000 sample points is enough for sub-mm numerical accuracy on a ~1m perimeter.
- The fit can over-shoot between markers if they're unevenly spaced — checks for self-intersection of the spline curve are advisable.

### 6.4 Convex hull as a sanity baseline

Always also compute the convex hull perimeter as a baseline. The hull will systematically *under-read* (it's the smallest convex envelope of the points), but it's robust and provides a lower bound for sanity-checking the ellipse / spline result.

```python
from scipy.spatial import ConvexHull
hull = ConvexHull(xz)
hull_perimeter = sum(
    np.linalg.norm(xz[hull.vertices[i]] - xz[hull.vertices[(i+1) % len(hull.vertices)]])
    for i in range(len(hull.vertices))
)
```

Report all three (hull, ellipse, spline) in the output and let the engineer compare. The hull-vs-spline ratio tells you how convex the cross-section is at this band; large gaps indicate non-elliptical shape (e.g., male pectorals).

---

## 7. Code skeleton

```python
"""
measure_markers_only.py

Body circumference measurements (mm) from ArUco markers on a calibration
garment. No SMPL / SMPL-X / neural network involved. Camera calibration
required.

Usage:
    python measure_markers_only.py path/to/video.mp4 \\
        --garment_config marker_only_v1.json \\
        --calibration_npz p50pro_main_12MP.npz \\
        --output report.json
"""

import argparse
import json
from pathlib import Path
import math

import cv2
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial import ConvexHull
from tqdm import tqdm


# ============================================================================
# Per-frame: detect markers and recover their 3D pose in camera frame
# ============================================================================

def detect_markers(frame_bgr, aruco_detector):
    """Return dict: marker_id -> 4x2 corner array (pixels)."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = aruco_detector.detectMarkers(gray)
    if ids is None:
        return {}
    return {int(ids[i, 0]): corners[i].squeeze() for i in range(len(ids))}


def estimate_marker_pose(corners_2d, marker_size_m, K, dist):
    """Return (R, t) of marker frame in camera frame, or (None, None) on failure."""
    half = marker_size_m / 2.0
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
    return R, tvec.flatten()


# ============================================================================
# Reference triad: build a body-local frame from 3 designated markers
# ============================================================================

def build_body_frame_from_triad(triad_positions_cam):
    """
    Given the 3D camera-frame positions of the three triad markers
    (dict: triad_id -> (3,) position), return a 4x4 transform from
    camera frame to body-local frame. Returns None if any triad marker
    is missing.

    Convention:
      origin = centroid of the three triad markers
      x_axis = normalised (B - C)  — lateral, scapula-to-scapula
      "tentative_up" = perpendicular to triangle plane, sign chosen so
                       body-Y points roughly up in camera frame
      y_axis = derived from tentative_up (refined later if external up exists)
      z_axis = y_axis × x_axis

    NOTE: This is a v1 implementation. The tentative_up direction is
    ambiguous — the triangle plane normal can point forward or backward.
    We resolve it by requiring the body-Y axis to have a positive dot
    product with the global -Y of the camera (i.e., body up = camera down).
    This works for standard upright capture; it breaks for tilted cameras
    or non-upright subjects. See §3.2 of the planning doc for alternatives.
    """
    A_id, B_id, C_id = 100, 101, 102  # TODO: read from garment_config
    if not all(k in triad_positions_cam for k in (A_id, B_id, C_id)):
        return None

    A = triad_positions_cam[A_id]
    B = triad_positions_cam[B_id]
    C = triad_positions_cam[C_id]
    origin = (A + B + C) / 3.0

    x_axis = (B - C) / np.linalg.norm(B - C)
    plane_normal = np.cross(B - A, C - A)
    plane_normal /= np.linalg.norm(plane_normal)

    # Resolve sign of plane normal. In OpenCV camera frame, +Y is DOWN, so
    # body up corresponds to camera -Y. We want body_y to dot positively
    # with camera -Y axis = (0, -1, 0).
    if plane_normal[1] > 0:
        plane_normal = -plane_normal
    y_axis = plane_normal

    # Re-orthogonalise (Gram–Schmidt) so x_axis stays unchanged
    y_axis = y_axis - np.dot(y_axis, x_axis) * x_axis
    y_axis /= np.linalg.norm(y_axis)
    z_axis = np.cross(x_axis, y_axis)

    R_cam_to_body = np.stack([x_axis, y_axis, z_axis], axis=0)  # rows are body axes in camera coords
    T = np.eye(4)
    T[:3, :3] = R_cam_to_body
    T[:3, 3] = -R_cam_to_body @ origin
    return T


def transform_point_cam_to_body(point_cam, T_cam_to_body):
    """Apply 4x4 transform to a 3-vector."""
    p4 = np.array([point_cam[0], point_cam[1], point_cam[2], 1.0])
    return (T_cam_to_body @ p4)[:3]


# ============================================================================
# Aggregation: collect each marker's body-local position across frames
# ============================================================================

def aggregate_markers(per_frame_markers, garment_config, calibration):
    """
    For each non-triad marker ID, return its median body-local position
    across all frames where the triad was successfully reconstructed.
    """
    K = calibration['K']
    dist = calibration['dist']
    triad_size = garment_config['reference_triad']['size_mm'] / 1000.0
    triad_ids = set(garment_config['reference_triad']['marker_ids'])

    # Per-band marker size lookup
    band_size_m = {}
    for band_name, band_cfg in garment_config['bands'].items():
        for mid in band_cfg['marker_ids']:
            band_size_m[mid] = band_cfg['marker_size_mm'] / 1000.0

    body_positions = {}  # marker_id -> list of (3,) body-frame positions

    for frame_data in per_frame_markers:
        # frame_data: dict marker_id -> 4x2 corners (pixels)

        # 1. Estimate triad poses
        triad_positions_cam = {}
        for tid in triad_ids:
            if tid not in frame_data:
                continue
            _, t = estimate_marker_pose(frame_data[tid], triad_size, K, dist)
            if t is not None:
                triad_positions_cam[tid] = t

        # 2. Build body frame; skip frame if triad incomplete
        T_cam_to_body = build_body_frame_from_triad(triad_positions_cam)
        if T_cam_to_body is None:
            continue

        # 3. Transform every band marker's centre into body frame
        for mid, corners in frame_data.items():
            if mid in triad_ids or mid not in band_size_m:
                continue
            _, t_cam = estimate_marker_pose(corners, band_size_m[mid], K, dist)
            if t_cam is None:
                continue
            t_body = transform_point_cam_to_body(t_cam, T_cam_to_body)
            body_positions.setdefault(mid, []).append(t_body)

    # 4. Median + outlier rejection per marker
    aggregated = {}
    for mid, positions in body_positions.items():
        arr = np.array(positions)  # (n_frames, 3)
        median = np.median(arr, axis=0)
        # Reject markers whose per-frame positions are too noisy
        std = np.std(arr, axis=0)
        if np.linalg.norm(std) > 0.04:  # 4cm — slightly tighter than Path A
            print(f"  warning: marker {mid} has high cross-frame std "
                  f"(|std|={np.linalg.norm(std)*1000:.1f}mm); rejecting")
            continue
        aggregated[mid] = median
    return aggregated


# ============================================================================
# Per-band: ellipse / spline / convex-hull circumference
# ============================================================================

def circumference_ellipse(xz_points):
    """Ramanujan-approximation circumference of fitted ellipse."""
    ellipse = cv2.fitEllipse(xz_points.astype(np.float32))
    (_, _), (a_full, b_full), _ = ellipse
    a, b = a_full / 2.0, b_full / 2.0  # semi-axes
    h = ((a - b) ** 2) / ((a + b) ** 2)
    return math.pi * (a + b) * (1 + (3 * h) / (10 + math.sqrt(4 - 3 * h)))


def circumference_spline(xz_points):
    """Closed periodic cubic spline perimeter."""
    centroid = xz_points.mean(axis=0)
    angles = np.arctan2(xz_points[:, 1] - centroid[1], xz_points[:, 0] - centroid[0])
    order = np.argsort(angles)
    pts = xz_points[order]
    pts_closed = np.vstack([pts, pts[:1]])
    t = np.linspace(0, 1, len(pts_closed))
    cs_x = CubicSpline(t, pts_closed[:, 0], bc_type='periodic')
    cs_y = CubicSpline(t, pts_closed[:, 1], bc_type='periodic')
    ts_dense = np.linspace(0, 1, 1000)
    xs = cs_x(ts_dense)
    ys = cs_y(ts_dense)
    return float(np.sum(np.sqrt(np.diff(xs)**2 + np.diff(ys)**2)))


def circumference_hull(xz_points):
    """Convex-hull perimeter (always under-reads; sanity baseline)."""
    hull = ConvexHull(xz_points)
    verts = hull.vertices
    return float(sum(
        np.linalg.norm(xz_points[verts[i]] - xz_points[verts[(i+1) % len(verts)]])
        for i in range(len(verts))
    ))


def measure_band(band_marker_ids, aggregated, n_min=4):
    """
    Compute circumference for one band given the band's marker IDs and the
    aggregated body-frame positions. Returns dict with hull/ellipse/spline
    estimates in metres, plus diagnostic info.
    """
    pts = np.array([
        aggregated[mid] for mid in band_marker_ids
        if mid in aggregated
    ])
    if len(pts) < n_min:
        return {'error': f'only {len(pts)}/{len(band_marker_ids)} markers detected'}

    # All band markers should be at approximately the same Y. Diagnostic:
    y_std = float(np.std(pts[:, 1]))
    xz = pts[:, [0, 2]]

    result = {
        'n_markers_used': len(pts),
        'y_mean_m': float(np.mean(pts[:, 1])),
        'y_std_mm': y_std * 1000.0,
        'hull_perimeter_mm': circumference_hull(xz) * 1000.0,
        'ellipse_perimeter_mm': circumference_ellipse(xz) * 1000.0,
    }
    if len(pts) >= 6:
        result['spline_perimeter_mm'] = circumference_spline(xz) * 1000.0
    return result


# ============================================================================
# Main pipeline
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('video', type=Path)
    parser.add_argument('--garment_config', type=Path, required=True)
    parser.add_argument('--calibration_npz', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('report.json'))
    parser.add_argument('--frame_step', type=int, default=1,
                        help='Process every Nth frame')
    args = parser.parse_args()

    # Load configs
    with open(args.garment_config) as f:
        garment_config = json.load(f)
    calib = np.load(args.calibration_npz)
    calibration = {'K': calib['K'].astype(np.float64),
                   'dist': calib['dist'].astype(np.float64)}

    # ArUco detector setup
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters()
    aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

    # Per-frame ArUco detection (with optional undistortion)
    cap = cv2.VideoCapture(str(args.video))
    n_frames_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    per_frame_markers = []
    frame_idx = 0

    pbar = tqdm(total=n_frames_total // args.frame_step, desc='Detecting markers')
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % args.frame_step == 0:
            # Undistort
            frame_u = cv2.undistort(frame, calibration['K'], calibration['dist'])
            per_frame_markers.append(detect_markers(frame_u, aruco_detector))
            pbar.update(1)
        frame_idx += 1
    pbar.close()
    cap.release()

    print(f'Processed {len(per_frame_markers)} frames')

    # Aggregate to body frame
    aggregated = aggregate_markers(per_frame_markers, garment_config, calibration)
    print(f'Aggregated {len(aggregated)} unique markers in body frame')

    # Per-band measurements
    report = {
        'video': str(args.video),
        'garment': garment_config['garment_id'],
        'n_frames_processed': len(per_frame_markers),
        'n_markers_aggregated': len(aggregated),
        'measurements': {},
    }
    for band_name, band_cfg in garment_config['bands'].items():
        report['measurements'][band_name] = measure_band(
            band_cfg['marker_ids'], aggregated
        )

    with open(args.output, 'w') as f:
        json.dump(report, f, indent=2)
    print(f'Report saved to {args.output}')

    # Print summary
    for band_name, result in report['measurements'].items():
        if 'error' in result:
            print(f'{band_name}: {result["error"]}')
            continue
        ellipse_mm = result.get('ellipse_perimeter_mm', 'n/a')
        spline_mm = result.get('spline_perimeter_mm', 'n/a')
        print(f'{band_name}: ellipse={ellipse_mm:.1f}mm, spline={spline_mm}, '
              f'y_std={result["y_std_mm"]:.1f}mm, n={result["n_markers_used"]}')


if __name__ == '__main__':
    main()
```

---

## 8. Sanity checks the script must implement

Before publishing any measurement number, the script must verify:

1. **Triad detection rate.** If the triad was reconstructed in <50% of frames, abort with a clear error — capture is unusable. Log the rate.
2. **Triad orientation drift.** Compare initial triad frame (first 5 sec) with final (last 3 sec). If the body x-axis or y-axis differs by more than 5° between start and end, capture had subject movement that broke the rigid-body assumption — flag prominently.
3. **Per-marker cross-frame std.** Already in the skeleton: reject any marker whose body-frame position varies by more than 40 mm across detections.
4. **Band Y-coplanarity.** All markers in a band should be at the same Y. Log the std; if >15 mm, the band placement is non-horizontal on the garment and circumference will be biased.
5. **Hull-vs-spline ratio.** If hull perimeter > 95% of spline perimeter, the cross-section is essentially convex (ellipse / spline are likely correct). If hull < 85% of spline, the spline is fitting major concavities — cross-check with raw marker positions before trusting the number.
6. **Marker count per band.** Below 4 markers = abort that band. 4–5 markers = ellipse only, no spline. 6+ markers = both. Be explicit in the output which method produced each number.

---

## 9. Calibration & validation

### 9.1 Per-camera calibration

A camera calibration `.npz` (from the existing `undistort_images.py` workflow or a separate `calibrate_camera.py`) is mandatory. The pipeline aborts if not provided.

Recommended calibration parameters: at least 20 checkerboard images covering the field of view, reprojection error <1 px. Verify by re-running the calibration on a known marker grid and confirming recovered marker positions match measured positions to <1 mm.

### 9.2 Per-subject ground-truth comparison

Every captured subject should have tape-measure ground truth for chest, waist, hips. The pipeline output includes the measurements at three levels (hull, ellipse, spline); choose the one closest to ground truth across the calibration cohort and apply a per-measurement linear correction:

```
corrected = a × predicted + b
```

Fit `a, b` per measurement (chest, waist, hips) on a 5–8 subject cohort spanning the AOF size range. Apply at inference time. Same recipe as proposed in `MARKER_INTEGRATION_PLAN.md` §9.

### 9.3 Cross-validation against SMPL pipelines

For each subject, run all three:

- `measure_bodies.py` (joint-derived slice heights, SMPL mesh)
- `measure_with_markers.py` (marker-derived heights, SMPL mesh)
- `measure_markers_only.py` (this pipeline)

Report all three side-by-side in the Sprint 3 tech-selection report. Per-measurement winners are likely to vary by band; the goal is to identify which approach delivers AOF tolerance most reliably across subjects, not to declare one universal winner.

---

## 10. Open questions for the engineer

1. **Triad placement.** Is the upper-sternum + L/R scapula triad acceptable, or does the garment design constrain placement? Alternative: anterior triad (sternum + L/R lateral chest) is easier to manufacture but more vulnerable to breathing artifact.
2. **Up-axis reference.** Is gravity/IMU data available from the capture device, or must the body-up axis be derived from a calibration T-pose at the start of each capture?
3. **Marker count per band.** 4 (minimum), 8 (recommended), or 12+ (best)? Drives garment manufacturing cost and accuracy ceiling.
4. **Variable marker sizes.** Triad markers should be larger (50mm) than band markers (30mm) for solvePnP precision. Is this acceptable from a manufacturing standpoint?
5. **Linear measurements (height, arm length, span).** Out of scope for this pipeline as written. Should these be measured by adding more markers (e.g., a head-top marker, wrist markers), or measured separately by the SMPL pipelines and combined into a hybrid report?
6. **Capture protocol enforcement.** The strict static / rotation / static segmentation requires the user-facing app to guide the subject through the protocol with timed prompts. Acceptable for Phase 1 R&D; may need to soften for Phase 2 production.

---

## 11. Suggested implementation order

1. **Camera calibration utility verified end-to-end.** Re-use `undistort_images.py` plus a new `calibrate_camera.py` if not already present. Validate by undistorting a checkerboard and confirming straight lines stay straight. **Half day.**
2. **ArUco detection harness.** Standalone script that reads a video, runs detection per frame, and saves per-frame marker dicts to disk. No body math yet. **1 day.**
3. **Triad-frame builder.** Implement `build_body_frame_from_triad`, test on a static T-pose video where the triad position is hand-verified. **1 day.**
4. **Single-band aggregation + ellipse.** Pick the chest band, aggregate, fit ellipse, compare to tape. Iterate until error <10%. **2 days.**
5. **All bands + spline + sanity checks.** Extend to all three bands; add the spline path; implement the §8 sanity checks. **2 days.**
6. **Linear bias correction over a 5-subject cohort.** Capture, measure, fit `a, b` per measurement, evaluate residual error. **3 days, blocked on subject availability.**
7. **Cross-pipeline comparison report.** Integrate output with the existing report pipeline so marker-only runs appear alongside SMPL-based runs in the comparison table. **Half day.**

End-to-end: ~2 weeks of engineering once the garment is available, plus subject capture time.

---

## 12. Out of scope for this doc

- **Mobile / production deployment.** This pipeline is Phase 1 R&D, designed for desktop iteration. Phase 2 mobile app is a separate concern with different constraints (real-time, lower-resolution capture, no checkerboard calibration available to end users).
- **Garment manufacturing.** Marker placement, fabric choice, sewing tolerances, durability — all engineer-side decisions. This doc assumes the garment exists.
- **Multi-subject capture.** All scripts assume one subject per capture.
- **Comparison to SfM track.** The marker-only pipeline produces sparse data (8 points per band); SfM produces dense data (~10⁶ points across the body). They are complementary, not competing. SfM evaluation is a separate workstream.
