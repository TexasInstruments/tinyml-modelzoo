# Radar Posture & Fall Detection — Data Preprocessing Pipeline User Guide

This guide documents the complete radar data preprocessing pipeline for the xWRL6432 mmWave radar sensor. The pipeline transforms raw CSV captures into model-ready tensor windows through 5 token-gated stages.

---

## Quick Start: Choose Your Path

| Your Data Type | Recommended Config | Description |
|---|---|---|
| **Raw CSV from TI capture tools** (bin*, point_*, track_* columns) | `feature_extraction_name: "Generic_392Input_PointCloudWithTrack_SimpleWindow"` | Full pipeline (Stages 2–5) — default preset |
| **Pre-cleaned / external features** (already windowed or no standard columns) | `feature_extraction_name: "Generic_392Input_SimpleWindowOnly"` | Stage 5 only — windowing + quality index |
| **Custom pipeline** | Inline `data_proc_transforms` + individual params | Full manual control |

> **Example config:** See `pose_and_fall_radar_config.yaml` for a working baseline.

---

## Pipeline Overview

```
Raw CSV/NPY/PKL/TXT
        │
        ▼
┌──────────────────┐
│ Stage 0: Load    │  (auto header detection)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐     "Radar Point Cloud Data"  → point_cloud_* params
│ Stage 2: Point   │ ──────────────────────────▶
│ Cloud Process    │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐     "Radar Track Data"        → track_* params
│ Stage 3: Track   │ ──────────────────────────▶
│ Data Process     │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐     "Scaling"                 → radar_data_scaling_track_velocity
│ Stage 4: Scaling │ ──────────────────────────▶
└────────┬─────────┘
         │
         ▼
┌──────────────────┐     "Simple Window"           → frame_size, stride_size,
│ Stage 5: Window  │ ──────────────────────────▶     radar_frames_quality_index,
│ + Quality Index  │                                radar_feature_ordering
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Output Tensors   │  X: [N, frame_size × features_per_radar_frame]
│                  │  Y: [N] class indices
└──────────────────┘
```

**Key principle:** Each stage runs **IFF its token string is present** in `data_proc_transforms`. Order in the list does not matter — the internal signal-chain order is fixed.

> **Note:** Stage 1 (Radar Range Data) is available but not included in the default presets. Enable it inline if you need FFT bin selection.

---

## Presets (Recommended)

| Preset Name | Stages Enabled | Input Requirements | Features/Frame | Total Input Size | Use Case |
|---|---|---|---|---|---|
| `Generic_392Input_PointCloudWithTrack_SimpleWindow` | 2–5 (Point Cloud, Track, Scaling, Window) | TI capture tool CSV (point_*, track_*) | 49 | 392 (49 × 8) | **Default** — raw data, full pipeline |
| `Generic_392Input_SimpleWindowOnly` | 5 only | Any numeric CSV / pre-windowed data | Depends on input | 392 (configurable) | Pre-cleaned data, external feature extractor |

> **Note:** Both presets set `frame_size=8`. `stride_size` differs: `0.125` (dense) for preset 1, `0.5` (sparse) for preset 2.

### Preset 1 Detail: `Generic_392Input_PointCloudWithTrack_SimpleWindow`
- **Point Cloud (5 features × 8 pts = 40):** x, y, z, doppler, snr — relative coords (y only), spatial filter ±10m
- **Track (9 features):** x, y, z, velocity (3), acceleration (3) — velocity/accel bounds ±10, spatial z-only ±10
- **Scaling:** track_velocity × 30
- **Window:** frame_size=8, stride=1 (0.125), per_window quality (min 12 pts), frame_major ordering

---

## Complete Parameter Reference

### Core Pipeline Control

| Parameter | Type | Default | Description |
|---|---|---|---|
| `feature_extraction_name` | string | — | Preset name (overrides inline params if set) |
| `data_proc_transforms` | list[string] | `[]` | Tokens to enable stages. Valid: `["Radar Range Data", "Radar Point Cloud Data", "Radar Track Data", "Scaling", "Simple Window"]` |
| `frame_size` | int | 8 | Window size = number of radar frames per tensor window |
| `stride_size` | float (0–1) | 0.125 | Stride as fraction of `frame_size`. Stride = `int(frame_size × stride_size)` |
| `gof_test` | bool | false | Enable goodness-of-fit test (debug) |

---

### Stage 1: Radar Range Data — FFT Bin Selection
**Token:** `"Radar Range Data"` *(not in default presets)*

| Parameter | Type | Default | Description |
|---|---|---|---|
| `radar_range_bins_bounds` | [int, int] or null | null | Inclusive `[min_bin, max_bin]` indices to keep from `bin*` columns. `null` = keep all. |

**Example:** `[6, 31]` → keeps `bin6` through `bin31` (26 bins). Use to crop near-field/far-field clutter.

---

### Stage 2: Radar Point Cloud Data — Point Cloud Processing
**Token:** `"Radar Point Cloud Data"`

#### Feature Selection
| Parameter | Type | Default (Preset 1) | Options |
|---|---|---|---|
| `radar_point_cloud_data_selector` | list[string] | `['x','y','z','doppler','snr']` | Subset of `['x','y','z','doppler','snr']` |

#### Coordinate Transform
| Parameter | Type | Default (Preset 1) | Description |
|---|---|---|---|
| `radar_point_cloud_coordinate_system` | string | `'relative'` | `'absolute'` or `'relative'` (make coords relative to track centroid) |
| `radar_point_cloud_relative_coordinate_filter` | list[string] | `['y']` | Axes to make relative when `coordinate_system='relative'` |

#### Spatial Filtering
| Parameter | Type | Default (Preset 1) | Description |
|---|---|---|---|
| `radar_point_cloud_spatial_filter_bounds` | [float×6] | `[-10,10, -10,10, -10,10]` | `[min_x, max_x, min_y, max_y, min_z, max_z]` — points outside dropped |
| `radar_point_cloud_non_spatial_filter_bounds` | [float×4] | `[null,null,null,null]` | `[min_doppler, max_doppler, min_snr, max_snr]` — `null` = no filter |

#### Per-Frame Point Selection
| Parameter | Type | Default (Preset 1) | Description |
|---|---|---|---|
| `radar_max_points_per_frame` | int | 8 | Max points to keep per frame after filtering/sorting |
| `radar_min_points_per_frame` | int | 0 | Min points required per frame (used with `per_radar_frame` quality mode) |
| `radar_point_cloud_structure_strategy` | string | `'highest and lowest'` | `'highest only'`, `'lowest only'`, `'highest and lowest'` (interleave) |
| `radar_point_cloud_sort_by` | string | `'z'` | Feature to sort by: `'x'`, `'y'`, `'z'`, `'doppler'`, `'snr'` |

---

### Stage 3: Radar Track Data — Track Feature Processing
**Token:** `"Radar Track Data"`

| Parameter | Type | Default (Preset 1) | Options / Description |
|---|---|---|---|
| `radar_track_data_selector` | list[string] | `['x','y','z','velocity','acceleration']` | Subset of `['velocity','x','y','z','acceleration']` |
| `radar_track_velocity_bounds` | [float, float] | `[-10,10]` | `[min_vel, max_vel]` applied to all `track_velocity_*` columns |
| `radar_track_acceleration_bounds` | [float, float] | `[-10,10]` | `[min_acc, max_acc]` applied to all `track_acc_*` columns |
| `radar_track_spatial_filter_bounds` | [float×6] | `[null×4, -10,10]` | `[min_x,max_x, min_y,max_y, min_z,max_z]` for track position columns |

**Track column mapping:**
- `velocity` → `track_velocity_x`, `track_velocity_y`, `track_velocity_z`
- `x` → `track_x`
- `y` → `track_y`
- `z` → `track_z`
- `acceleration` → `track_acc_x`, `track_acc_y`, `track_acc_z`

> **Ghost track filtering** is automatic — validates `track_id`/`tid` column integrity.

---

### Stage 4: Scaling — Independent Column Scaling
**Token:** `"Scaling"`

| Parameter | Type | Default (Preset 1) | Description |
|---|---|---|---|
| `radar_data_scaling_track_velocity` | float | 30 | Multiply all `track_velocity_*` columns by this factor |

> Scaling is independent of other stages — runs whenever `"Scaling"` token is present.

---

### Stage 5: Simple Window — Quality Index + Sliding Window
**Token:** `"Simple Window"`

#### Windowing
| Parameter | Type | Default (Preset 1) | Description |
|---|---|---|---|
| `frame_size` | int | 8 | Number of radar frames per window (set at top level) |
| `stride_size` | float (0–1) | 0.125 (P1) / 0.5 (P2) | Stride fraction (set at top level) |

#### Quality Index Modes
| Parameter | Type | Default (Preset 1) | Modes |
|---|---|---|---|
| `radar_frames_quality_index` | string or null | `'per_window'` | `'per_radar_frame'`, `'per_window'`, `null` (disabled) |

| Mode | Behavior | Required Params |
|---|---|---|
| `'per_radar_frame'` | Drop individual frames with < `radar_min_points_per_frame` real points | `radar_min_points_per_frame` |
| `'per_window'` | Drop entire windows with < `radar_min_points_per_window` total points | `radar_min_points_per_window` |
| `null` | No quality filtering — keep all windows/frames | — |

#### Quality Thresholds
| Parameter | Type | Default (Preset 1) | Used With |
|---|---|---|---|
| `radar_min_points_per_frame` | int | 0 | `per_radar_frame` mode |
| `radar_min_points_per_window` | int | 12 | `per_window` mode (recommended) |

#### Feature Ordering
| Parameter | Type | Default (Preset 1) | Description |
|---|---|---|---|
| `radar_feature_ordering` | string | `'frame_major'` | `'frame_major'`: `[f0_feat0, f0_feat1..., f1_feat0, f1_feat1...]`<br>`'feature_major'`: `[f0_feat0, f1_feat0..., f0_feat1, f1_feat1...]` |

---

## Use-Case Scenarios

### 1. Standard Pose & Fall Detection (Default)
```yaml
data_processing_feature_extraction:
  feature_extraction_name: "Generic_392Input_PointCloudWithTrack_SimpleWindow"
  frame_size: 8
  stride_size: 0.125
```
**Result:** 392-input tensor (49 features × 8 frames), dense sliding window.

### 2. Pre-Cleaned / External Feature Data
```yaml
data_processing_feature_extraction:
  feature_extraction_name: "Generic_392Input_SimpleWindowOnly"
  frame_size: 8
  stride_size: 0.5
```
**Result:** Only windowing + quality check. No column name requirements.

### 3. Custom: Add Range Bin Selection
```yaml
data_processing_feature_extraction:
  feature_extraction_name: "Generic_392Input_PointCloudWithTrack_SimpleWindow"
  data_proc_transforms:
    - "Radar Range Data"
    - "Radar Point Cloud Data"
    - "Radar Track Data"
    - "Scaling"
    - "Simple Window"
  radar_range_bins_bounds: [6, 31]
```

### 4. Custom: Disable Quality Index (Keep All Windows)
```yaml
data_processing_feature_extraction:
  feature_extraction_name: "Generic_392Input_PointCloudWithTrack_SimpleWindow"
  radar_frames_quality_index: null
```

### 5. Custom: Sparse Windowing for Large Datasets
```yaml
data_processing_feature_extraction:
  feature_extraction_name: "Generic_392Input_PointCloudWithTrack_SimpleWindow"
  stride_size: 0.5  # 50% overlap instead of 12.5%
```
Reduces training samples ~4× for faster iteration.

---

## Input Data Requirements

### For Full Pipeline (Preset 1)
CSV must contain these **case-sensitive** columns (per TI capture tool format):

| Stage | Expected Columns |
|---|---|
| Point Cloud | `point_x0`, `point_y0`, `point_z0`, `point_doppler0`, `point_snr0`, `point_x1`... |
| Track | `track_id` (or `tid`), `track_x`, `track_y`, `track_z`, `track_velocity_x`, `track_velocity_y`, `track_velocity_z`, `track_acc_x`, `track_acc_y`, `track_acc_z` |

### For SimpleWindowOnly (Preset 2)
**No specific column names required** — any numeric CSV with consistent columns works. The pipeline only applies sliding window + quality index.

---

## Output Tensor Shape

| Config | Features/Frame | Frame Size | Tensor Shape (X) |
|---|---|---|---|
| PointCloud+Track (Preset 1) | 49 | 8 | `[N, 392]` |
| SimpleWindowOnly (Preset 2) | *input-dependent* | 8 | `[N, frame_size × input_features]` |

Labels (Y): `[N]` integer class indices (0 = standing, 1 = sitting, 2 = walking, 3 = falling, 4 = laying).

---

## Common Pitfalls

| Symptom | Likely Cause | Fix |
|---|---|---|
| `KeyError: 'point_x0'` | CSV missing point cloud columns | Use `SimpleWindowOnly` preset or verify capture tool output |
| All windows dropped | `radar_min_points_per_window` too high | Lower threshold or check data quality |
| Mismatched input size | `frame_size` ≠ model expectation | Keep `frame_size=8` for RF_DualStream_PointCloud_19K / RF_MLP_4L_19K |
| int32 output with `quantization=0` | Known TVM behavior for classification logits | Set `output_int=false` in compilation target (handled automatically) |

---

## Files in This Example

| File | Purpose |
|---|---|
| `pose_and_fall_radar_config.yaml` | Baseline training config (raw data, full pipeline) |
| `README.md` | This guide |

---

## Further Reading

- **Preset definitions:** `tinyml_modelmaker/ai_modules/radar/constants.py` → `FEATURE_EXTRACTION_PRESET_DESCRIPTIONS`
- **Dataset loader:** `tinyml-tinyverse/tinyml_tinyverse/common/datasets/radar_dataset.py` → `GenericRadarDataset`
- **Model zoo:** `tinyml-modelzoo/examples/radar_posture_and_fall_detection/`

---

*Generated for Tensorlab EdgeAI — Texas Instruments*