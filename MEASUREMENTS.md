# Body Measurement Methodology

This document describes how `measure_bodies.py` derives body measurements from
SMPLest-X inference results, the assumptions behind each step, and the known
limitations of the approach.

---

## 1. Why this works without camera calibration

SMPL-X is a parametric 3D body model: the 10-dimensional shape vector `betas`
encodes body proportions in absolute metric units (metres) directly on the
canonical mesh, independent of the camera that produced the input image. Once
`betas` has been estimated from a frame, the body's geometry is fully
determined and can be queried in any pose.

The pipeline therefore:

1. Reads `betas` from each detection's `.npz` file (written by `main/inference.py`).
2. Forwards the SMPL-X model in canonical T-pose (all joint rotations zeroed).
3. Measures the resulting mesh.

This decouples measurement from pose-estimation noise and from camera
intrinsics — `transl` and global orientation are intentionally ignored.

> **Important caveat.** Decoupling from camera parameters does *not* mean the
> measurements are calibration-free in absolute terms. `betas` is the *output*
> of SMPLest-X regression, which is itself sensitive to the input image's
> perspective, distortion, and framing. Garbage `betas` in → garbage
> measurements out. The capture protocol and image-side undistortion still
> matter; this stage just isolates the geometric extraction from the
> camera-projection problem.

---

## 2. Running the pipeline

**Step 1 — inference** (writes rendered frames + `.npz` shape files):

```bash
sh scripts/inference.sh smplest_x_h myvideo.mp4 30
# .npz files written to: demo/results/myvideo/smplx/{frame:06d}_{bbox_id}.npz
# rendered overlay:       demo/result_myvideo.mp4
```

**Step 2 — measure**:

```bash
python measure_bodies.py demo/results/myvideo/
# optional: provide known measurements for error reporting
python measure_bodies.py demo/results/myvideo/ \
    --ground-truth 1815,1070,990,1040,470,1400
```

Output is printed to stdout and saved to `demo/results/myvideo/myvideo.txt`.
A per-frame scatter plot with rolling mean is saved as `demo/results/myvideo/myvideo.png`.

---

## 3. SMPL-X model output

`smplx.SMPLX(...)` with the constructor flags used in `measure_bodies.py`
returns:

| Field      | Shape          | Units  | Notes |
|------------|----------------|--------|-------|
| `vertices` | `(10475, 3)`   | metres | Canonical mesh surface |
| `joints`   | `(144, 3)` *   | metres | Body + hands + face landmarks |
| `faces`    | `(20908, 3)`   | int    | Triangle indices into `vertices` |

\* SMPL-X returns **127 joints** by default; with `use_face_contour=True`
(which the script sets) the output expands to **144** by appending 17 face
contour landmarks. ([vchoutas/smplx Issue #14][smplx-issue14],
[vchoutas/smplx demo][smplx-demo])

The first **22 entries of `joints` are the SMPL/SMPL-X body joints**, in this
order:

```
 0 pelvis        1 left_hip      2 right_hip
 3 spine1        4 left_knee     5 right_knee
 6 spine2        7 left_ankle    8 right_ankle
 9 spine3       10 left_foot    11 right_foot
12 neck         13 left_collar  14 right_collar
15 head         16 left_shoulder 17 right_shoulder
18 left_elbow   19 right_elbow  20 left_wrist  21 right_wrist
```

Source: [Meshcapade SMPL wiki][meshcapade-smpl] and the canonical
[vchoutas/smplx joint_names.py][smplx-joint-names]. The body-joint indexing is
identical between SMPL, SMPL+H and SMPL-X, so this mapping is stable across
the model family.

The Y axis is up in SMPL-X coordinates; the X axis is the lateral (left-right)
direction; Z is depth.

---

## 4. Computational pipeline

For each `.npz` detection file, `measure_bodies.py`:

1. Loads the 10 SMPL-X shape `betas`.
2. Runs SMPL-X in canonical T-pose: zero global orientation, zero body pose,
   zero hand/jaw/eye poses, zero expression. This produces a pose-independent
   mesh that depends only on the subject's shape parameters.
3. Extracts `vertices` and `joints`.
4. Computes the six measurements described in §6 below.

After processing all detections, the script aggregates per-detection
measurements into per-subject summary statistics (mean, standard deviation,
range) and produces a per-frame plot with rolling mean.

> The recommended aggregation is **median of `betas` across frames first**
> (the default `--aggregation median_betas`), then run the model and measure
> once on the median shape. Per-frame measurements have correlated errors (one
> bad pose ruins all measurements from that frame together) which simple
> per-measurement averaging masks. Median-`betas` aggregation surfaces bad
> frames as outliers in shape space and is robust to them. See §9.

---

## 5. Slice-height anatomy: how SMPL-X spine joints map to the human torso

The pipeline derives chest, waist, and hip slice heights from SMPL-X joint
positions rather than hard-coding them. This makes the measurements adapt to
each subject's torso proportions but introduces a question: *what anatomical
level does each SMPL-X spine joint actually correspond to?*

**SMPL-X has only three spine joints** (`spine1`, `spine2`, `spine3`) plus the
pelvis and neck. The biological spine has 24 vertebrae (7 cervical, 12
thoracic, 5 lumbar) plus the sacrum. The three SMPL-X spine joints are
**linear-blend-skinning anchors** fit to bulk torso deformation during model
training — they are not anatomical landmarks and do not correspond to specific
vertebrae. Their exact heights drift with body proportions: a tall-torso
subject and a short-torso subject have the same three joint indices but at
different absolute heights.

In approximate terms, on a neutral mesh:

| Joint    | Approx. % of stature | Approximate anatomical region |
|----------|---------------------|-------------------------------|
| `spine1` | ~58–62%             | Lower lumbar / iliac crest area |
| `spine2` | ~65–70%             | Lower thoracic / floating-rib level |
| `spine3` | ~76–80%             | Upper thoracic / scapular level |
| `neck`   | ~83–87%             | Cervicothoracic junction (~C7-T1) |

These percentages are observational (from neutral-`betas` SMPL-X meshes), not
specifications, and vary by ±2–3% with subject `betas`. **They should not be
labelled with specific vertebral codes** (e.g. "L3-L4") — that level of
precision is not provided by the model.

---

## 6. Per-measurement methodology

### Height — vertex extent

```python
height = vertices[:, 1].max() - vertices[:, 1].min()
```

Vertical extent of the mesh from highest vertex (top of cranium) to lowest
vertex (heel). This corresponds to **stature** as defined by ISO 8559-1:2017:
vertical distance from the floor to the highest point of the head, subject
standing erect with head in the Frankfort horizontal plane.

Two implementation assumptions:

- **SMPL-X has no hair geometry.** The top vertex is the cranial vault, which
  matches stadiometer "top of head" measurement. (If the model had hair, this
  would over-read by hair height.)
- **In T-pose the lowest vertex is the heel sole.** Verified empirically; the
  big toes can extend slightly forward but not below the heel in the canonical
  rest pose.

### Chest, Waist, Hips — cross-section perimeter

Each circumference is computed by intersecting a horizontal plane with every
mesh triangle and summing the resulting edge lengths:

```
For each triangle:
  - Identify which vertices lie above/below plane Y = y
  - For each edge that crosses the plane:
      t = (y - y_i) / (y_j - y_i)        # linear interpolation
      p = v_i + t * (v_j - v_i)          # crossing point in 3D
  - A triangle contributes 0 or 2 crossing points
  - Segments are stitched into closed contours; the largest is returned
```

Only the largest closed contour at each slice height is used. This correctly
excludes spurious smaller loops that appear when the plane grazes the upper
arms — summing all contours would over-read by 100–300 mm in those cases.

**Slice heights:**

| Measurement | Slice height | Approx. % of stature | Anatomical region |
|-------------|-------------|---------------------|-------------------|
| Chest       | midpoint of `spine2` and `spine3` | ~71–74% | Mid/upper thorax — broadly in the bust/pectoral band |
| Waist       | `spine1`                          | ~58–62% | Lower lumbar — close to the natural waist for many subjects |
| Hips        | mean Y of `left_hip` and `right_hip` joints | ~50–53% | Greater trochanter level — the widest section of the lower body |

**Note on the chest level.** The natural-bra-band level (men: nipple line;
women: fullest part of bust) sits around 72–75% of stature in typical adults.
The mid-`spine2`/`spine3` slice lands close to this for men and approximately
correct for women, but a fixed slice does not move to "fullest part of the
bust" for subjects with significant bust development. For high-precision
female chest measurement, a vertex-landmark based slice (rather than
joint-derived) is preferable — see §7.

**Note on the waist level.** [ISO 8559-1:2017][iso-8559] and the [WHO
guidelines][who-waist] define the waist circumference as the smallest
horizontal circumference of the natural waist between the lower rib and the
iliac crest, typically 2–4 cm above the navel. The `spine1` joint sits in
this region but is not guaranteed to coincide with the *narrowest* point —
for athletic V-shape subjects it falls slightly above the narrowest girth,
for subjects with central adiposity it may fall slightly below. Empirically
the systematic error is ±10–20 mm in the slice height itself, which
propagates to circumference error of similar magnitude.

**Earlier slice heights were anatomically incorrect.** A previous version
used `(spine3 + neck) / 2` for chest (~83% of stature, clavicle / shoulder
girdle level) and `spine2` for waist (~67% of stature, lower thorax). The
clavicle level is the widest upper-body cross-section in males due to
trapezius and shoulder-blade geometry, producing systematic positive errors
of 50–100 mm on chest circumference. Both slice heights have been corrected
in the current implementation.

### Shoulder width — joint-to-joint horizontal distance

```python
shoulder_width = abs(joints[L_SHOULDER, 0] - joints[R_SHOULDER, 0])
```

X-axis distance between left and right shoulder joints. **This is not
biacromial breadth.** The SMPL-X shoulder joints are placed at the
glenohumeral centers (deep in the shoulder capsule), which sit medially and
inferiorly to the acromion processes by approximately 30–50 mm on each side.

The downstream consequence: **joint-to-joint shoulder width systematically
under-reads biacromial breadth by 60–100 mm** depending on subject's shoulder
muscle development. The two measurements correlate well across subjects, but
they are not interchangeable in absolute terms.

For applications that need true biacromial breadth (e.g. apron / vest sizing
based on shoulder width), either:

- Switch to a vertex-landmark approach using SMPL-X mesh vertex IDs at the
  acromion processes (see [SMPL-Anthropometry][smpl-anthro]), or
- Apply a calibrated additive offset (subject-population specific, typically
  +60 to +90 mm) derived from ground-truth tape measurements.

### Arm span — wrist-to-wrist horizontal distance

```python
arm_span = abs(joints[L_WRIST, 0] - joints[R_WRIST, 0])
```

Horizontal distance between left and right wrist joints in T-pose. Valid
because T-pose places both arms horizontal and lateral. This corresponds to
**wrist-to-wrist span**, not fingertip-to-fingertip arm span as commonly
defined in anthropometry — see ground-truth protocol in §8.

For per-arm sleeve/arm length (relevant to garment sizing), this measurement
is *not* sufficient and should be replaced with shoulder-to-wrist distance
per arm. See §10.

---

## 7. Comparison with SMPL-Anthropometry's reference approach

The canonical reference for this kind of anthropometry on SMPL/SMPL-X meshes
is [DavidBoja/SMPL-Anthropometry][smpl-anthro], which uses a more
sophisticated approach:

> "LENGTHS are defined using 2 landmarks — the measurement is found as the
> distance between the landmarks. CIRCUMFERENCES are defined with landmarks
> and joints — the measurement is found by cutting the body model with the
> plane defined by a point (landmark point) and normal (vector connecting
> the two joints)."  
> — [SMPL-Anthropometry README][smpl-anthro]

Two key differences from `measure_bodies.py`:

1. **Landmarks vs. joints for slice anchors.** SMPL-Anthropometry uses
   specific *vertex IDs* on the mesh (which are stable across body shapes
   because mesh topology is fixed) to anchor each slice. `measure_bodies.py`
   uses joint positions (which drift with body proportions). The vertex
   approach gives more anatomically consistent measurements across subjects.
2. **Plane normal direction.** SMPL-Anthropometry defines the slice plane
   with a normal *along a chosen joint-to-joint vector* (e.g. from a
   neighbouring spine joint), which slightly tilts the slice to match the
   body's local axis. `measure_bodies.py` uses pure horizontal slices
   (Y-axis normal). For a strictly upright T-pose, these are nearly
   equivalent; for any pose other than perfect T-pose, the SMPL-Anthropometry
   approach is more robust.

The current implementation is simpler and correct for **canonical T-pose
only**. If the pipeline is ever adapted to measure posed meshes (e.g. for
real-time fitting visualization), it should switch to the SMPL-Anthropometry
approach.

---

## 8. Ground-truth measurement protocol

To compare computed values against real subjects, the tape protocol must
match what the model computes. All protocols below follow ISO 8559-1:2017
where applicable.

| Measurement | Protocol |
|-------------|----------|
| **Height** | Subject standing barefoot, heels together, head in Frankfort horizontal plane (lower orbital margin and tragion roughly horizontal — informally, eyes looking straight ahead). Stadiometer reading from floor to vertex (top of head). [[ISO 8559-1]][iso-8559] |
| **Chest** | Tape around the chest at the **nipple line** (men) or **fullest part of the bust** (women), arms relaxed at sides, measured at the end of normal exhale. Tape held horizontal and snug but not compressing soft tissue. Note: the script's slice may not exactly coincide with "fullest of bust" for women — see §6. |
| **Waist** | Tape around the **natural waist** — the smallest horizontal circumference between the lower rib and the iliac crest, typically 2–4 cm above the navel. Subject relaxed, breathing normally, arms at sides. [[ISO 8559-1]][iso-8559], [[Wikipedia: Waist]][wiki-waist] |
| **Hips** | Tape around the **maximum circumference of the buttocks**, feet together, weight evenly distributed. Tape held horizontal. Approximately at the level of the greater trochanters, where SMPL-X places the hip joints. |
| **Shoulder width (joint-to-joint)** | The script measures glenohumeral-to-glenohumeral distance, *not* biacromial breadth. To match: use a sliding caliper or ruler held horizontally across the upper torso, measuring point-to-point between the shoulder joint centers (approximately 3–5 cm medial to and below each acromion). For most validation purposes, **measure biacromial breadth (acromion-to-acromion)** with calipers and apply a known offset of typically +60 to +90 mm to convert the script output, rather than trying to reproduce the script's exact reference points by tape. |
| **Arm span (wrist-to-wrist)** | Subject standing, arms extended horizontally at shoulder height, palms forward. Measure straight-line horizontal distance between the two wrist joints (radial styloid to radial styloid) — *not* fingertip-to-fingertip. |

---

## 9. Aggregating measurements across detections

A SMPLest-X run produces one `betas` estimate per detection. For a single
subject filmed across many frames, **`betas` should be constant** —
inter-frame variability is model noise, not real body change. There are
several ways to aggregate, in order of increasing robustness:

1. **Per-frame measurement, then mean across frames.** (`--aggregation per_frame`)
   Simple but sensitive to per-frame outliers (e.g. frames where the detector
   grabbed a wrong bounding box or the regressor produced an extreme `betas`).
2. **Per-frame measurement, then median across frames.** Robust to outlier
   frames because a single bad frame barely shifts the median.
3. **Median of `betas` across frames, then measure once.** (`--aggregation median_betas`, default)
   Robust *and* removes correlated within-frame errors. A single bad frame
   produces bad `betas` for *all* measurements together — per-measurement
   averaging hides this, but `betas`-space averaging surfaces it as a
   statistical outlier in shape space (detected via 2-σ rejection on each
   `betas` dimension before taking the median).

Option 3 is the default and recommended aggregation strategy. Adjust the
outlier rejection threshold with `--sigma-threshold` (default: 2.0).

---

## 10. Known limitations

- **Circumference accuracy.** SMPL-X vertices approximate the skin surface
  but the mesh under-resolves folds, creases, and soft-tissue deformation.
  Reported errors in the literature for SMPL-based circumference measurement
  on real subjects are 5–20 mm for waist/hip and somewhat larger for chest,
  even with multi-view reconstruction pipelines that produce excellent
  meshes. ([Lou et al., 2025][lou-2025] report 5.0 ± 5.3 mm linear errors
  on real-world data with a multi-view system; circumference errors are
  larger.) For monocular SMPLest-X regression the realistic floor is
  ~30–80 mm uncorrected on circumferences, falling to 15–35 mm with
  per-subject linear bias correction against ground truth.

- **Soft-tissue compression.** Tape measurement compresses soft tissue;
  3D-reconstructed meshes do not. [Bonnet et al. (2025)][bonnet-2025] report
  systematic over-estimation of waist circumference by 3D scan vs. manual
  measurement, attributable to absent tissue compression. This bias is
  consistent across subjects and is a candidate for linear correction.

- **Biacromion under-estimation.** The same Bonnet et al. study reports
  systematic *under*-estimation of biacromion length by 3D scanning, due to
  standardized scanning posture and the joint-vs-acromion distinction
  discussed in §6. This is also linearly correctable.

- **Chest slice drift across body proportions.** SMPL-X spine joint heights
  vary with `betas`; for tall or short-torsoed subjects the chest slice may
  land at a different anatomical level (see §5). Worst-case drift is around
  ±2–3% of stature, ~30–60 mm.

- **Female chest measurement.** A fixed slice height does not match
  "fullest part of the bust" for subjects with significant bust development.
  For high-precision female bust measurement, a vertex-landmark approach
  per gendered SMPL-X model is preferable.

- **Inter-frame `betas` variability.** Each detection is an independent
  estimate. Use the rolling mean from the plot — or better, the median
  `betas` (§9) — to read the stable per-person estimate.

- **No sleeve / arm length.** The current script measures wrist-to-wrist
  arm span only. For per-arm length (e.g. shoulder-to-wrist, relevant for
  garment sleeve sizing), an additional measurement should be added using
  the existing joints array: `np.linalg.norm(joints[L_SHOULDER] −
  joints[L_WRIST])` and similarly for the right side.

---

## 11. References

- **ISO 8559-1:2017** — *Size designation of clothes — Part 1: Anthropometric
  definitions for body measurement.* The canonical reference for clothing-
  industry anthropometric protocols. <https://www.iso.org/standard/61686.html>
- **DavidBoja/SMPL-Anthropometry** — Reference implementation of measurement
  extraction from SMPL/SMPL-X meshes using vertex landmarks.
  <https://github.com/DavidBoja/SMPL-Anthropometry>
- **vchoutas/smplx** — Canonical SMPL-X PyTorch implementation; joint names
  in [`smplx/joint_names.py`][smplx-joint-names].
- **Meshcapade SMPL wiki** — Reference table of SMPL/SMPL-X joint indices.
  <https://github.com/Meshcapade/wiki/blob/main/wiki/SMPL.md>
- **Pavlakos et al., CVPR 2019** — *Expressive Body Capture: 3D Hands, Face,
  and Body from a Single Image.* The SMPL-X paper.
  <https://smpl-x.is.tue.mpg.de/>
- **Yin et al., TPAMI 2025** — *SMPLest-X: Ultimate Scaling for Expressive
  Human Pose and Shape Estimation.* <https://arxiv.org/abs/2501.09782>
- **Lou et al. (2025)**, *Accurate 3D anthropometric measurement using
  compact multi-view imaging.* Measurement, Volume 244, 2025.
  <https://www.sciencedirect.com/science/article/abs/pii/S0263224125001368>
- **Bonnet et al. (2025)**, on systematic biases in 3D scan vs. manual
  measurement (biacromion under-estimation, waist over-estimation due to
  absent soft-tissue compression).
- **Kennedy et al. (2012)**, *Waist Measurements Compared: Definitions
  (ISO vs CAESAR) and Instruments (Manual vs 3D Scanned Data).*
  <https://www.researchgate.net/publication/256706476>

[smplx-joint-names]: https://github.com/vchoutas/smplx/blob/main/smplx/joint_names.py
[smplx-issue14]: https://github.com/vchoutas/smplx/issues/14
[smplx-demo]: https://github.com/vchoutas/smplx
[meshcapade-smpl]: https://github.com/Meshcapade/wiki/blob/main/wiki/SMPL.md
[smpl-anthro]: https://github.com/DavidBoja/SMPL-Anthropometry
[iso-8559]: https://www.iso.org/standard/61686.html
[wiki-waist]: https://en.wikipedia.org/wiki/Waist
[who-waist]: https://www.who.int/data/gho/indicator-metadata-registry/imr-details/2380
[lou-2025]: https://www.sciencedirect.com/science/article/abs/pii/S0263224125001368
[bonnet-2025]: https://www.researchgate.net/figure/Distances-of-circumference-paths-through-mesh-vertices-of-a-registered-SMPL-body-model_fig5_338747533
