"""
Compute body measurements (mm) from SMPLest-X inference results.

No camera calibration required for the measurement step: SMPL-X shape `betas`
encode body proportions in absolute metric units (metres) on a canonical mesh.
We run the model in T-pose (all joint rotations zeroed) so measurements
reflect body shape alone, independent of the captured pose. The saved `transl`
values are intentionally ignored.

The pipeline aggregates `betas` across detections rather than aggregating
per-frame measurements: for a single subject, `betas` should be constant
across frames, and inter-frame variability is regression noise. Median-betas
aggregation with outlier rejection is more robust than averaging per-frame
measurements because per-frame errors are correlated (one bad frame produces
bad measurements for everything together) and surface as outliers in
shape-parameter space where they can be detected and rejected.

See MEASUREMENTS.md for the full methodology, anatomical assumptions, and
sourcing.

Input .npz files are produced by running inference first:
    sh scripts/inference.sh smplest_x_h myvideo.mp4 30
Results are written to demo/results/{VIDEO_NAME}/smplx/*.npz

Usage:
    python measure_bodies.py demo/results/myvideo/
    python measure_bodies.py demo/results/myvideo/ --model_path ./human_models/human_model_files
    python measure_bodies.py demo/results/myvideo/ --aggregation median_betas
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

# ---------------------------------------------------------------------------
# SMPL-X body joint indices (first 22 entries of the joints output array).
#
# Canonical reference: vchoutas/smplx joint_names.py
#     https://github.com/vchoutas/smplx/blob/main/smplx/joint_names.py
# Cross-referenced with Meshcapade SMPL wiki:
#     https://github.com/Meshcapade/wiki/blob/main/wiki/SMPL.md
#
# Indexing is identical between SMPL, SMPL+H and SMPL-X for the first 22
# body joints. SMPL-X with use_face_contour=True returns 144 joints total
# (127 default + 17 face contour landmarks); only the first 22 are used here.
#
#   0  pelvis        1  left_hip      2  right_hip
#   3  spine1        4  left_knee     5  right_knee
#   6  spine2        7  left_ankle    8  right_ankle
#   9  spine3       10  left_foot    11  right_foot
#  12  neck         13  left_collar  14  right_collar
#  15  head         16  left_shoulder 17  right_shoulder
#  18  left_elbow   19  right_elbow  20  left_wrist  21  right_wrist
#
# Anatomical note: spine1/spine2/spine3 are linear-blend-skinning anchors
# fit to bulk torso deformation during model training. They are NOT
# anatomical landmarks and do NOT correspond to specific vertebrae. Their
# heights drift with body proportions. See MEASUREMENTS.md §4.
# ---------------------------------------------------------------------------
PELVIS      = 0
L_HIP, R_HIP = 1, 2
SPINE_1     = 3   # ~58–62% of stature; lower lumbar / iliac crest area
SPINE_2     = 6   # ~65–70% of stature; lower thoracic / floating-rib level
SPINE_3     = 9   # ~76–80% of stature; upper thoracic / scapular level
NECK        = 12
L_SHOULDER, R_SHOULDER = 16, 17
L_WRIST, R_WRIST       = 20, 21


def load_smplx(model_path: str, gender: str = "neutral"):
    """Load the SMPL-X body model from disk.

    use_face_contour=True expands the joints output from 127 to 144 by
    appending 17 dynamic face contour landmarks. We don't use these for
    measurement, but the inference pipeline upstream may rely on them.
    """
    try:
        import smplx
    except ImportError:
        sys.exit("smplx not installed. Run: pip install smplx")

    # Use SMPLX directly rather than smplx.create(), which infers model type
    # from the path string and misidentifies 'human_model_files' as 'human'.
    smplx_dir = os.path.join(model_path, "smplx")
    return smplx.SMPLX(
        smplx_dir,
        gender=gender,
        num_betas=10,
        use_face_contour=True,
        use_pca=False,
        flat_hand_mean=False,
        batch_size=1,
    )


def tpose_mesh(model, betas: np.ndarray):
    """Forward pass with zero pose, returning (vertices, joints) in metres.

    All joint rotations, hand/jaw/eye poses, and expression parameters are
    left at zero — only `betas` drives the output. This produces the
    canonical T-pose mesh whose geometry depends solely on body shape.
    """
    b = torch.tensor(betas.reshape(1, 10), dtype=torch.float32)
    with torch.no_grad():
        out = model(betas=b, return_verts=True)
    return out.vertices.squeeze().numpy(), out.joints.squeeze().numpy()


def _contour_perimeters(vertices: np.ndarray, faces: np.ndarray, y: float) -> list[float]:
    """
    Find all closed contours formed by intersecting horizontal plane Y=y with
    the mesh, and return their individual perimeters.

    Each triangle that straddles the plane contributes a single line segment
    (entry crossing → exit crossing). Stitching these segments end-to-end via
    shared crossing points reconstructs each closed contour around the body
    cross-section.

    Why we need this: at certain Y values the plane intersects the mesh in
    multiple disjoint loops (e.g. just below the armpits, where one loop goes
    around the torso and two more loops go around each upper arm). Naively
    summing all segment lengths conflates these into one bogus perimeter.
    Returning the *list* of contour perimeters lets the caller pick the right
    one (typically the largest = torso).

    The "0 or 2 crossings per triangle" invariant comes from the topology of
    a plane cutting a triangle: it must enter through one edge and exit
    through another, or graze a vertex (treated as no crossing).
    """
    # Collect line segments contributed by each straddling triangle.
    # Each segment is a pair of 3D points (entry, exit) on the slice plane.
    segments: list[tuple[np.ndarray, np.ndarray]] = []

    for tri_idx in faces:
        tri = vertices[tri_idx]                # (3, 3)
        above = tri[:, 1] > y
        # Skip degenerate cases: all-above or all-below means no crossing.
        if above.all() or not above.any():
            continue

        crossings: list[np.ndarray] = []
        for i in range(3):
            j = (i + 1) % 3
            if above[i] != above[j]:
                # Linear interpolation along the edge:
                # solve y = y_i + t * (y_j - y_i) for t, then interpolate the
                # full 3D position at that fraction along the edge.
                t = (y - tri[i, 1]) / (tri[j, 1] - tri[i, 1])
                crossings.append(tri[i] + t * (tri[j] - tri[i]))

        # Topological invariant: a triangle straddling a plane must produce
        # exactly 2 crossings. Anything else means the input mesh is degenerate
        # (zero-area triangle, vertex exactly on the plane, etc.).
        assert len(crossings) == 2, (
            f"Triangle has {len(crossings)} crossings at y={y:.4f}; expected 2. "
            "This indicates a degenerate mesh face or a vertex landing exactly "
            "on the slice plane."
        )
        segments.append((crossings[0], crossings[1]))

    if not segments:
        return []

    # Stitch segments into closed contours by matching endpoints.
    # Two segments belong to the same contour if they share a crossing point;
    # each crossing point is the meeting of exactly two adjacent triangles.
    # We use spatial hashing rather than exact equality because floating-point
    # interpolation can produce points that are equal in principle but differ
    # by ~1e-9 metres in practice.
    SCALE = 1e6  # quantise to micrometres — well below mesh resolution
    def key(p: np.ndarray) -> tuple[int, int, int]:
        return (int(round(p[0] * SCALE)), int(round(p[1] * SCALE)), int(round(p[2] * SCALE)))

    # Build adjacency: each crossing-point key maps to the set of segments that touch it.
    point_to_segments: dict[tuple[int, int, int], list[int]] = {}
    for seg_idx, (a, b) in enumerate(segments):
        for p in (a, b):
            point_to_segments.setdefault(key(p), []).append(seg_idx)

    # Walk segments to assemble contours via shared endpoints.
    visited = [False] * len(segments)
    contour_perimeters: list[float] = []

    for start in range(len(segments)):
        if visited[start]:
            continue
        # Walk a contour starting from segment `start`.
        current = start
        visited[current] = True
        a, b = segments[current]
        perimeter = float(np.linalg.norm(b - a))
        last_endpoint = b

        while True:
            neighbours = point_to_segments.get(key(last_endpoint), [])
            next_seg = None
            for n in neighbours:
                if n != current and not visited[n]:
                    next_seg = n
                    break
            if next_seg is None:
                break
            visited[next_seg] = True
            a, b = segments[next_seg]
            # The shared point may be at either end of the next segment;
            # advance to whichever endpoint is *not* the shared one.
            if key(a) == key(last_endpoint):
                perimeter += float(np.linalg.norm(b - a))
                last_endpoint = b
            else:
                perimeter += float(np.linalg.norm(a - b))
                last_endpoint = a
            current = next_seg

        contour_perimeters.append(perimeter)

    return contour_perimeters


def cross_section_perimeter(vertices: np.ndarray, faces: np.ndarray, y: float) -> float | None:
    """Perimeter of the largest closed contour at horizontal slice Y=y.

    Returns the largest contour because for horizontal slices through the
    torso (chest/waist/hip range), the largest contour is always the body
    trunk. Smaller contours at the same height — e.g. through grazed upper
    arms when the slice drifts high — are correctly excluded rather than
    summed into the result.

    Returns None if the plane misses the mesh entirely.
    """
    perimeters = _contour_perimeters(vertices, faces, y)
    if not perimeters:
        return None
    return max(perimeters)


def measure(vertices: np.ndarray, joints: np.ndarray, faces: np.ndarray) -> dict:
    """
    Returns measurements in metres.

    Slice heights are derived from joint positions in T-pose. See MEASUREMENTS.md
    §4 for the anatomical caveats — these are SMPL-X skinning anchors, not
    anatomical landmarks, and approximate (not specify) the named regions.

    Slice heights:
      chest  → midpoint between spine2 and spine3  (~71–74% of stature)
      waist  → spine1                              (~58–62% of stature)
      hips   → mean of L_Hip / R_Hip               (~50–53% of stature)

    These slice positions were corrected from an earlier version that placed
    chest at (spine3+neck)/2 (~83% of stature, clavicle level) and waist at
    spine2 (~67% of stature, lower thorax). The clavicle level is the widest
    upper-body cross-section in males due to trapezius and shoulder-blade
    geometry, producing systematic +50–100 mm errors on chest. Both heights
    have been corrected.

    Linear measurements (shoulder width, arm length, arm span) use joint
    positions directly. NOTE that shoulder_width here is glenohumeral-to-
    glenohumeral distance, NOT biacromial breadth — it under-reads true
    shoulder width by 60–100 mm. See MEASUREMENTS.md §5.
    """
    # Stature: vertical extent from highest vertex (cranium) to lowest (heel).
    # SMPL-X has no hair geometry, so the top vertex matches stadiometer height.
    height = float(vertices[:, 1].max() - vertices[:, 1].min())

    # Circumference slice heights in metres (SMPL-X Y-axis is up).
    y_chest = float((joints[SPINE_2, 1] + joints[SPINE_3, 1]) / 2)
    y_waist = float(joints[SPINE_1, 1])
    y_hips  = float((joints[L_HIP, 1] + joints[R_HIP, 1]) / 2)

    # Shoulder width: lateral (X-axis) distance between glenohumeral joint
    # centres. Under-reads biacromial breadth — apply +60 to +90 mm offset
    # for downstream garment-sizing if biacromial is needed.
    shoulder_width = float(abs(joints[L_SHOULDER, 0] - joints[R_SHOULDER, 0]))

    # Arm span: wrist-to-wrist X-axis distance. T-pose places arms horizontal
    # and lateral, so X distance equals straight-line wrist separation.
    # NOTE: This is wrist-to-wrist span, NOT fingertip-to-fingertip arm span
    # as commonly defined in anthropometry.
    arm_span = float(abs(joints[L_WRIST, 0] - joints[R_WRIST, 0]))

    return {
        "height":         height,
        "chest":          cross_section_perimeter(vertices, faces, y_chest),
        "waist":          cross_section_perimeter(vertices, faces, y_waist),
        "hips":           cross_section_perimeter(vertices, faces, y_hips),
        "shoulder_width": shoulder_width,
        "arm_span":       arm_span,
    }


def reject_betas_outliers(
    all_betas: np.ndarray, sigma_threshold: float = 2.0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Reject detections whose `betas` deviate from the median by more than
    `sigma_threshold` standard deviations in any dimension.

    Why median+std rather than mean+std: a single bad detection (wrong
    bounding box, failed regression) can produce extreme `betas` that would
    inflate the mean and the standard deviation, masking itself as inlier.
    Median is robust; using it as the centre and population-std as the scale
    is a standard robust-rejection recipe.

    Returns:
        cleaned_betas: (N_kept, 10) array of accepted detections
        accepted_mask: (N_total,) boolean mask, True for kept detections
    """
    median = np.median(all_betas, axis=0)
    std = np.std(all_betas, axis=0)
    # Avoid division by zero on dimensions where all detections agree exactly
    # (rare but possible for synthetic test inputs).
    std_safe = np.where(std > 1e-9, std, 1e-9)
    z = np.abs(all_betas - median) / std_safe
    # Accept only detections within threshold on EVERY betas dimension —
    # one bad dimension is enough to suspect the whole detection.
    accepted = (z < sigma_threshold).all(axis=1)
    return all_betas[accepted], accepted


def load_betas(npz_path: str) -> np.ndarray:
    data = dict(np.load(npz_path, allow_pickle=True))
    return data["betas"].reshape(10).astype(np.float32)


def frame_index(npz_path: str) -> int:
    """Extract frame number from filename, e.g. '000042_1.npz' → 42."""
    return int(os.path.basename(npz_path).split("_")[0])


def _scatter_rolling(ax, frames, vals_mm, window, scatter_label, color="steelblue"):
    ax.scatter(frames, vals_mm, s=4, alpha=0.35, color=color, label=scatter_label)
    order = np.argsort(frames)
    f_s, v_s = frames[order], vals_mm[order]
    if len(v_s) >= window:
        rolling = np.convolve(v_s, np.ones(window) / window, mode="valid")
        ax.plot(f_s[window - 1:], rolling, color="crimson", linewidth=1.5,
                label=f"rolling mean (w={window})")


def _finish_subplots(axes, n, ncols, nrows):
    for ax in axes[n:]:
        ax.set_visible(False)
    for ax in axes[(nrows - 1) * ncols: n]:
        ax.set_xlabel("frame index")


def plot_measurements(
    frame_indices: list[int],
    per_detection: dict[str, list],
    save_path: str,
    window: int = 30,
    ground_truth: dict[str, float] | None = None,
) -> None:
    keys = list(per_detection.keys())
    n = len(keys)
    ncols = 2
    nrows = (n + 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 3), sharex=True)
    axes = axes.flatten()
    frames = np.array(frame_indices)

    for ax, key in zip(axes, keys):
        raw = per_detection[key]
        valid = np.array([v is not None for v in raw])
        vals_mm = np.array([v * 1000 if v is not None else np.nan for v in raw], dtype=float)
        _scatter_rolling(ax, frames[valid], vals_mm[valid], window, "per detection")
        if ground_truth and key in ground_truth:
            ax.axhline(ground_truth[key], color="limegreen", linewidth=1.5,
                       linestyle="--", label="ground truth")
        ax.set_title(key.replace("_", " ").title())
        ax.set_ylabel("mm")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, linewidth=0.4, alpha=0.5)

    _finish_subplots(axes, n, ncols, nrows)
    fig.suptitle("Body measurements vs frame index", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {save_path}")

    if not (ground_truth and "height" in ground_truth):
        return

    # Height-adjusted figure: scale each detection's measurements by gt_height / measured_height
    gt_h = ground_truth["height"]
    heights_mm = np.array(
        [v * 1000 if v is not None else np.nan for v in per_detection["height"]], dtype=float
    )
    k_arr = gt_h / heights_mm  # per-detection coefficient

    adj_keys = [k for k in keys if k != "height"]
    n_adj = len(adj_keys)
    nrows_adj = (n_adj + 1) // ncols
    fig2, axes2 = plt.subplots(nrows_adj, ncols, figsize=(14, nrows_adj * 3), sharex=True)
    axes2 = axes2.flatten()

    for ax, key in zip(axes2, adj_keys):
        raw = per_detection[key]
        valid = np.array([v is not None for v in raw])
        vals_mm = np.array([v * 1000 if v is not None else np.nan for v in raw], dtype=float)
        adj_mm = vals_mm * k_arr
        _scatter_rolling(ax, frames[valid], adj_mm[valid], window,
                         "height-adjusted", color="mediumpurple")
        if ground_truth and key in ground_truth:
            ax.axhline(ground_truth[key], color="limegreen", linewidth=1.5,
                       linestyle="--", label="ground truth")
        ax.set_title(f"{key.replace('_', ' ').title()} (height-adjusted)")
        ax.set_ylabel("mm")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, linewidth=0.4, alpha=0.5)

    _finish_subplots(axes2, n_adj, ncols, nrows_adj)
    fig2.suptitle("Height-adjusted body measurements vs frame index", fontsize=13, y=1.01)
    fig2.tight_layout()
    adj_path = save_path.replace(".png", "_height_adjusted.png")
    fig2.savefig(adj_path, dpi=120, bbox_inches="tight")
    plt.close(fig2)
    print(f"Plot saved to {adj_path}")


def format_stats(label: str, values_m: list, gt_mm: float | None = None) -> str:
    values_m = [v for v in values_m if v is not None]
    if not values_m:
        return f"  {label:<18}: no data"
    arr = np.array(values_m) * 1000
    line = (f"  {label:<18}: {arr.mean():.1f} mm  ±{arr.std():.1f}  "
            f"[{arr.min():.1f} – {arr.max():.1f}]  n={len(arr)}")
    if gt_mm is not None:
        err = arr.mean() - gt_mm
        line += f"  |  error vs GT: {err:+.1f} mm ({err / gt_mm * 100:+.1f}%)"
    return line


def print_stats(label: str, values_m: list, gt_mm: float | None = None) -> str:
    line = format_stats(label, values_m, gt_mm)
    print(line)
    return line


def main():
    parser = argparse.ArgumentParser(
        description="Body measurement averages from SMPLest-X inference results"
    )
    parser.add_argument("results_dir", help="Path to demo/results/{VIDEO_NAME}/")
    _repo_root = os.path.dirname(os.path.abspath(__file__))
    _default_model_path = os.path.join(_repo_root, "human_models", "human_model_files")
    parser.add_argument("--model_path", default=_default_model_path,
                        help="Path to SMPL-X model files (default: ./human_models/human_model_files)")
    parser.add_argument("--gender", default="neutral", choices=["neutral", "male", "female"],
                        help="Body model gender (default: neutral)")
    parser.add_argument(
        "--aggregation",
        default="median_betas",
        choices=["median_betas", "mean_betas", "per_frame"],
        help=(
            "How to produce the canonical per-subject measurement. "
            "'median_betas' (default, recommended): take median of `betas` "
            "across detections after 2-sigma outlier rejection, then measure "
            "once. Robust to bad frames. "
            "'mean_betas': arithmetic mean of `betas`, no rejection. Less "
            "robust but matches the original behaviour. "
            "'per_frame': measure each detection independently and average "
            "the measurements. Discouraged — hides correlated per-frame errors."
        ),
    )
    parser.add_argument(
        "--sigma-threshold",
        type=float,
        default=2.0,
        help=(
            "Z-score threshold for `betas` outlier rejection in median_betas "
            "mode. Detections with any betas dimension more than this many "
            "standard deviations from the median are rejected. Default: 2.0."
        ),
    )
    parser.add_argument("--no-plot", dest="no_plot", action="store_true",
                        help="Skip saving the measurements plot")
    parser.add_argument("--rolling-window", type=int, default=30, metavar="N",
                        help="Rolling mean window size for the plot (default: 30)")
    parser.add_argument(
        "--ground-truth",
        default=None,
        metavar="H,C,W,HP,SH,AR",
        help=(
            "Optional ground truth in mm: height,chest,waist,hips,shoulders,arm_span "
            "(e.g. --ground-truth 1815,1070,990,1040,470,1400)"
        ),
    )
    args = parser.parse_args()

    _gt_keys = ("height", "chest", "waist", "hips", "shoulder_width", "arm_span")
    ground_truth: dict[str, float] | None = None
    if args.ground_truth is not None:
        try:
            _gt_vals = [float(x) for x in args.ground_truth.split(",")]
            if len(_gt_vals) != len(_gt_keys):
                raise ValueError
        except ValueError:
            sys.exit(f"--ground-truth must be exactly {len(_gt_keys)} comma-separated numbers")
        ground_truth = dict(zip(_gt_keys, _gt_vals))

    # ------------------------------------------------------------------
    # Discover .npz detection files.
    # ------------------------------------------------------------------
    smplx_dir = os.path.join(args.results_dir, "smplx")
    npz_files = sorted(glob.glob(os.path.join(smplx_dir, "*.npz")))
    if not npz_files:
        sys.exit(
            f"No .npz files found under {smplx_dir}\n"
            "Run inference first: sh scripts/inference.sh smplest_x_h myvideo.mp4 30"
        )

    print(f"Loading {len(npz_files)} detections from {smplx_dir}")

    model = load_smplx(args.model_path, gender=args.gender)
    model.eval()
    faces = model.faces  # (N_faces, 3) int array

    # ------------------------------------------------------------------
    # Pass 1: load all betas and compute per-frame measurements.
    # We always populate per_detection (for the plot and per-frame
    # statistics) even when --aggregation=median_betas, because the
    # per-frame view is useful for diagnosing capture issues and for
    # validating that the aggregation didn't hide a problem.
    # ------------------------------------------------------------------
    all_betas: list[np.ndarray] = []
    frame_indices: list[int] = []
    per_detection: dict[str, list[float]] = {
        k: [] for k in (
            "height", "chest", "waist", "hips",
            "shoulder_width", "arm_span",
        )
    }

    for path in tqdm(npz_files, desc="Processing detections", unit="det"):
        betas = load_betas(path)
        all_betas.append(betas)
        frame_indices.append(frame_index(path))
        verts, joints = tpose_mesh(model, betas)
        m = measure(verts, joints, faces)
        for k in per_detection:
            per_detection[k].append(m[k])

    all_betas_arr = np.array(all_betas)
    video_name = os.path.basename(os.path.normpath(args.results_dir))

    # Output collection: print to stdout AND save to a .txt for later reference.
    lines: list[str] = []

    def emit(s: str = "") -> None:
        print(s)
        lines.append(s)

    emit(f"\n=== Per-detection statistics  [{video_name}  gender={args.gender}] ===")
    for key in per_detection:
        gt_mm = ground_truth.get(key) if ground_truth else None
        emit(format_stats(key, per_detection[key], gt_mm=gt_mm))

    # ------------------------------------------------------------------
    # Pass 2: canonical per-subject measurement via the chosen aggregation.
    # This is the number that should be reported as THE measurement for
    # this subject. The per-frame statistics above are diagnostic.
    # ------------------------------------------------------------------
    if args.aggregation in ("median_betas", "mean_betas"):
        if args.aggregation == "median_betas":
            # Robust path: reject betas outliers, then take median.
            cleaned, mask = reject_betas_outliers(all_betas_arr, args.sigma_threshold)
            n_rejected = int((~mask).sum())
            emit(
                f"\n=== Median-betas aggregation "
                f"(rejected {n_rejected}/{len(all_betas_arr)} detections at "
                f"σ>{args.sigma_threshold}) ==="
            )
            if len(cleaned) == 0:
                emit(
                    "  WARNING: outlier rejection removed all detections. "
                    "Falling back to the full set. Consider raising --sigma-threshold."
                )
                cleaned = all_betas_arr
            agg_betas = np.median(cleaned, axis=0)
        else:  # mean_betas
            emit("\n=== Mean-betas aggregation (no outlier rejection) ===")
            agg_betas = np.mean(all_betas_arr, axis=0)

        verts, joints = tpose_mesh(model, agg_betas)
        m = measure(verts, joints, faces)
        for key, val in m.items():
            if val is None:
                continue
            gt_mm = ground_truth.get(key) if ground_truth else None
            err_str = ""
            if gt_mm is not None:
                err = val * 1000 - gt_mm
                err_str = f"  |  error vs GT: {err:+.1f} mm ({err / gt_mm * 100:+.1f}%)"
            emit(f"  {key:<18}: {val * 1000:.1f} mm{err_str}")

    elif args.aggregation == "per_frame":
        emit("\n=== Per-frame aggregation (canonical = mean of per-frame measurements) ===")
        emit("  See per-detection statistics block above.")

    # ------------------------------------------------------------------
    # Plot and save.
    # ------------------------------------------------------------------
    if not args.no_plot:
        plot_path = os.path.join(args.results_dir, f"{video_name}.png")
        plot_measurements(frame_indices, per_detection, plot_path,
                          window=args.rolling_window, ground_truth=ground_truth)

    txt_path = os.path.join(args.results_dir, f"{video_name}.txt")
    with open(txt_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Results saved to {txt_path}")


if __name__ == "__main__":
    main()
