# Radar Data Processing Pipeline - Complete Reference

> **Complete guide to the radar data processing pipeline in Tensorlab EdgeAI**  

---

## Common Use Case: Using Pre-Cleaned / Already Processed Datasets

**If you DON'T have data from TI's capture tools, or your CSV has different column names, or you've already done feature extraction externally** — you can SKIP the feature extraction stages entirely and use ONLY the windowing stage.

### Option 1: Use the SimpleWindowOnly Preset (RECOMMENDED)

```yaml
data_processing_feature_extraction:
  feature_extraction_name: "Generic_Radar_SimpleWindowOnly"
  # Only runs "Simple Window" stage - no column name requirements!
```

### Option 2: Inline Config with Only Simple Window

```yaml
data_processing_feature_extraction:
  data_proc_transforms:
    - "Simple Window"          # ONLY this stage runs
  frame_size: 8                # MUST match your model
  stride_size: 0.5
  variables: 1
```

## IMPORTANT: Column Name Requirements for Raw CSV Processing

The 5-stage pipeline expects **SPECIFIC COLUMN NAMES** in your raw CSV files. The pipeline is designed to work with data captured using TI's Unified Data Capture Tools (Future Scope).

### Expected Column Names (case-sensitive)

| Data Type | Column Names |
|-----------|--------------|
| **Range Data** | `bin0`, `bin1`, `bin2`, ... `binN` (FFT range profile) |
| **Point Cloud** | `point_x0`, `point_y0`, `point_z0`, `point_doppler0`, `point_snr0`, `point_x1`, `point_y1`, ... (up to max points) |
| **Track Data** | `track_id` (or `tid`) |
| | `track_distance_y`, `track_height_z`, `track_azimuth_x` |
| | `track_velocity_x`, `track_velocity_y`, `track_velocity_z` |
| | `track_acc_x`, `track_acc_y`, `track_acc_z` |

### Ghost Track Filtering
The pipeline automatically validates the `track_id`/`tid` column. 

**If `track_id`/`tid` column is missing**: The pipeline  **continues with all tracks treated as valid** (no frames dropped).

---



## Pipeline Architecture Overview

The `GenericRadarDataset` class (`tinyml_tinyverse/common/datasets/radar_dataset.py`) implements a **5-stage pipeline** controlled entirely by the `data_proc_transforms` list. There are **no separate boolean enable flags** — a stage runs iff its token is present in `data_proc_transforms`.

```yaml
data_processing_feature_extraction:
  data_proc_transforms:
    - "Radar Range Data"
    - "Radar Point Cloud Data"
    - "Radar Track Data"
    - "Scaling"
    - "Simple Window"
```


## Radar Data Processing Parameters 


### 1. Radar Range Data
**Purpose**: Select a subset of range bins (FFT range profile columns `bin*`).

**Activation**:
```yaml
data_proc_transforms:
  - "Radar Range Data"
```

**Configuration**:
```yaml
radar_range_bins_bounds: [6, 31]  # [min_bin_index, max_bin_index] — inclusive
```

**Behavior**:
| Value | Behavior |
|-------|----------|
| `null` / not set | Keep all `bin*` columns |
| `[min, max]` | Keep bins where `min ≤ bin_index ≤ max` |

---

### Stage 2: Radar Point Cloud Data
**Purpose**: Process point cloud detections — selection, coordinate transform, filtering, per-frame point selection.

**Activation**:
```yaml
data_proc_transforms:
  - "Radar Point Cloud Data"
```

**Configuration**:
```yaml
# Feature selection (which point features to keep)
radar_point_cloud_data_selector: ['y', 'z', 'snr']
# Options: 'x', 'y', 'z', 'doppler', 'snr'

# Coordinate system
radar_point_cloud_coordinate_system: 'relative'  # 'absolute' or 'relative'
radar_point_cloud_relative_coordinate_filter: ['y']  # axes to make relative

# Spatial filter bounds (x, y, z)
# [min_x, max_x, min_y, max_y, min_z, max_z]
radar_point_cloud_spatial_filter_bounds: [0.0, 5.0, -3.0, 3.0, -60.0, 60.0]

# Non-spatial filter bounds (doppler, snr)
# [min_doppler, max_doppler, min_snr, max_snr]
radar_point_cloud_non_spatial_filter_bounds: [-3.0, 3.0, 5.0, 30.0]

# Point selection per frame
radar_max_points_per_frame: 8
radar_min_points_per_frame: 0
radar_point_cloud_structure_strategy: 'highest and lowest'
radar_point_cloud_sort_by: 'z'
```

**Coordinate Transform (Relative Mode)**:
When `coordinate_system: 'relative'` and `relative_coordinate_filter` includes axes:
```python
# y made relative to track position
point_y_relative = point_y - track_distance_y

# z made relative to track height
point_z_relative = point_z - track_height_z
```

**Point Selection Strategy**:
| Strategy | Description |
|----------|-------------|
| `'highest only'` | Keep top N points by sort feature |
| `'lowest only'` | Keep bottom N points by sort feature |
| `'highest and lowest'` | Interleave: highest, lowest, 2nd highest, 2nd lowest... |

`radar_point_cloud_sort_by: 'z'` helps in sorting the points accordingly (custom case for point cloud structure strategy) Eg. when set to `y` - sorts all the points wrt `point_y` values in a radr frame and then chooses based on the strategy used (`highest only`/`lowest only`)

---

### Stage 3: Radar Track Data
**Purpose**: Process track-level features per frame — ghost track removal, feature selection, filtering.

**Activation**:
```yaml
data_proc_transforms:
  - "Radar Track Data"
```

**Configuration**:
```yaml
# Feature selection
radar_track_data_selector: ['z', 'velocity', 'acceleration']
# Options: 'velocity', 'x', 'y', 'z', 'acceleration'

# Velocity filter [min_vel, max_vel] — applied to all track_velocity_* columns
radar_track_velocity_bounds: [-20.0, 20.0]

# Acceleration filter [min_acc, max_acc] — applied to all track_acc_* columns
radar_track_acceleration_bounds: [-20.0, 20.0]

# Spatial filter [min_x, max_x, min_y, max_y, min_z, max_z]
radar_track_spatial_filter_bounds: [-10.0, 10.0, -5.0, 5.0, -90.0, 90.0]
```

**Track Column Prefixes**:
| Feature Group | Columns |
|---------------|---------|
| `velocity` | `track_velocity_x`, `track_velocity_y`, `track_velocity_z` |
| `distance` | `track_distance_y` |
| `height` | `track_height_z` |
| `azimuth` | `track_azimuth_x` |
| `acceleration` | `track_acc_x`, `track_acc_y`, `track_acc_z` |

---

### Stage 4: Scaling
**Purpose**: Apply scaling factor to specific column groups. **Independent of "Radar Track Data" stage**.

**Activation**:
```yaml
data_proc_transforms:
  - "Scaling"  # Runs even without "Radar Track Data"!
```

**Configuration**:
```yaml
# Multiply all track_velocity_* columns by this factor
radar_data_scaling_track_velocity: 30
```
---

### Stage 5: Simple Window (Frame Quality + Windowing)
**Purpose**: Per-frame point quality control + sliding window creation for model input.

**Activation**:
```yaml
data_proc_transforms:
  - "Simple Window"
```

**Configuration**:
```yaml
# Window parameters
frame_size: 8                         # Frames per window (CRITICAL for model input shape)
stride_size: 0.5                      # Stride as fraction of frame_size (0-1)

# Point cloud structure
radar_max_points_per_frame: 8         # Max points to keep per frame
radar_min_points_per_frame: 0         # Min points required per frame (per_radar_frame mode)
radar_min_points_per_window: 12       # Min total points across window (per_window mode)
radar_point_cloud_structure_strategy: 'highest and lowest'
radar_point_cloud_sort_by: 'height'   # Sort feature: 'distance', 'height', 'azimuth', 'doppler', 'snr'

# Quality indexing modes
radar_frames_quality_index: 'per_window'  # 'per_radar_frame', 'per_window', or unset

# Feature ordering
radar_feature_ordering: 'frame_major'     # 'frame_major' or 'feature_major'
```

**Quality Index Modes**:

| Mode | Description | Parameters | Requires "Simple Window" |
|------|-------------|------------|-------------------------|
| `unset` / empty | No quality filtering | N/A | No |
| `'per_radar_frame'` | Drop individual frames with < `radar_min_points_per_frame` real points | `radar_min_points_per_frame` | **No** (works standalone) |
| `'per_window'` | Drop entire windows with < `radar_min_points_per_window` total real points across all frames in window | `radar_min_points_per_window` | **Yes** (needs windowing) |

> **New in v1.4.0**: `per_radar_frame` quality filtering now works **independently** of the "Simple Window" transform. You can use it with just the feature extraction stages, or even with pre-cleaned data in pass-through mode in case the dataset is labelled with respect to standard point cloud column names, else skips quality indexing.

---

## Feature Extraction Presets

### Preset 1: Generic_392Input_PointCloudWithTrack_SimpleWindow (Recommended for Pose/Fall)

**Use Case**: Complete raw CSV processing with full feature extraction and quality filtering.

**Resulting Input Tensor**: `(N, 8 × 31) = (N, 248)` — matches `RF_DualStream_PointCloud_19K` default.

```python
Generic_392Input_PointCloudWithTrack_SimpleWindow = dict(
    data_processing_feature_extraction=dict(
        data_proc_transforms=[
            'Radar Point Cloud Data',
            'Radar Track Data',
            'Simple Window'
        ],
        frame_size=8,
        stride_size=0.5,
        radar_point_cloud_data_selector=['y', 'z', 'snr'],
        radar_point_cloud_coordinate_system='relative',
        radar_point_cloud_relative_coordinate_filter=['y'],
        radar_point_cloud_spatial_filter_bounds=[-10.0, 10.0, -10.0, 10.0, None, None],
        radar_point_cloud_non_spatial_filter_bounds=[None, None, None, None],
        radar_track_data_selector=['z', 'velocity', 'acceleration'],
        radar_track_velocity_bounds=[-10.0, 10.0],
        radar_track_acceleration_bounds=[-10.0, 10.0],
        radar_track_spatial_filter_bounds=[None, None, -10.0, 10.0, None, None],
        radar_max_points_per_frame=8,
        radar_min_points_per_window=12,
        radar_point_cloud_structure_strategy='highest and lowest',
        radar_point_cloud_sort_by='z',
        radar_frames_quality_index='per_window',
        radar_feature_ordering='frame_major',
        variables=1,
    ),
    common=dict(task_type=TASK_TYPE_RADAR_CLASSIFICATION),
)
```

### Preset 2: Generic_Radar_SimpleWindowOnly

**Use Case**: Pre-cleaned/refined datasets where CSV already has correct per-frame features. Only windowing is needed.

```python
Generic_Radar_SimpleWindowOnly = dict(
    data_processing_feature_extraction=dict(
        data_proc_transforms=['Simple Window'],
        frame_size=8,
        stride_size=0.5,
        variables=1,
    ),
    common=dict(task_type=TASK_TYPE_RADAR_CLASSIFICATION),
)
```

**Notes**:
- No range bin, point cloud, track, or scaling stages run
- Input CSV must already have the correct per-frame feature columns
- `frame_size` must match what the model expects

---



## Table of All Configurable Parameters

### Data Processing Feature Extraction Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `feature_extraction_name` | str | None | Preset name from constants.py |
| `data_proc_transforms` | list | [] | **Main pipeline gate** — tokens: "Radar Range Data", "Radar Point Cloud Data", "Radar Track Data", "Scaling", "Simple Window" |
| `frame_size` | int | 8 | **Frames per window (critical for model input shape)** |
| `stride_size` | float | 0.5 | Stride as fraction of frame_size (0-1) |
| `target_variables` | list | [] | Target column names |
| `feature_size_per_frame` | int | None | Override auto-detected features/frame |
| `gof_test` | bool | False | Goodness-of-fit test |

### Radar-Specific Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `radar_range_bins_bounds` | list[2] | None | `[min_bin, max_bin]` for range bin selection |
| `radar_point_cloud_data_selector` | list[str] | all 5 | Point features to keep: 'x', 'y', 'z', 'doppler', 'snr' |
| `radar_point_cloud_coordinate_system` | str | 'absolute' | 'absolute' or 'relative' |
| `radar_point_cloud_relative_coordinate_filter` | list[str] | [] | Axes to make relative: 'y', 'z' |
| `radar_point_cloud_spatial_filter_bounds` | list[6] | None | `[min_x, max_x, min_y, max_y, min_z, max_z]` |
| `radar_point_cloud_non_spatial_filter_bounds` | list[4] | None | `[min_dop, max_dop, min_snr, max_snr]` |
| `radar_track_data_selector` | list[str] | all 5 | Track features: 'velocity', 'x', 'y', 'z', 'acceleration' |
| `radar_track_velocity_bounds` | list[2] | None | `[min_vel, max_vel]` |
| `radar_track_acceleration_bounds` | list[2] | None | `[min_acc, max_acc]` |
| `radar_track_spatial_filter_bounds` | list[6] | None | `[min_x, max_x, min_y, max_y, min_z, max_z]` |
| `radar_data_scaling_track_velocity` | float | None | Scale factor for `track_velocity_*` columns |
| `radar_max_points_per_frame` | int | 8 | Max points per frame after selection |
| `radar_min_points_per_frame` | int | 0 | Min real points required per frame (per_radar_frame mode) |
| `radar_min_points_per_window` | int | 0 | Min total real points across window (per_window mode) |
| `radar_point_cloud_structure_strategy` | str | 'highest and lowest' | Point selection strategy |
| `radar_point_cloud_sort_by` | str | 'height' | Feature to sort points by: 'x', 'y', 'z', 'doppler', 'snr' |
| `radar_frames_quality_index` | str | None | 'per_radar_frame', 'per_window', or unset |
| `radar_feature_ordering` | str | 'frame_major' | 'frame_major' or 'feature_major' |

---

## Model Integration: RF_DualStream_PointCloud_19K

### Architecture Overview

```
Input: [batch, frame_size × 31]
         │
         ├──▶ Position Stream (28 features/frame × frame_size)
         │       posz(1) + acc(3) + points(24)
         │       ──▶ BN → Linear(60) → ReLU → BN
         │
         └──▶ Velocity Stream (3 features/frame × frame_size)
                 velx(1) + vely(1) + velz(1)
                 ──▶ BN → Linear(20) → ReLU → BN
                      │
                      ▼
                 Fusion: Concat(60, 20) = 80
                      │
                      ▼
                 Linear(40) → ReLU → BN → Linear(20) → ReLU → BN → Linear(num_classes)
                      │
                      ▼
                 Softmax → [batch, num_classes]
```

### Per-Frame Feature Layout (31 features)

| Index | Feature | Source |
|-------|---------|--------|
| 0 | `track_height_z` | Track |
| 1 | `track_velocity_x` | Track |
| 2 | `track_velocity_y` | Track |
| 3 | `track_velocity_z` | Track |
| 4 | `track_acc_x` | Track |
| 5 | `track_acc_y` | Track |
| 6 | `track_acc_z` | Track |
| 7-30 | 8 points × [y, z, snr] | Point Cloud (24 features) |

**Total per frame**: 7 (track) + 24 (8 points × 3) = **31 features**

---

*Document Version: 1.0*  
*Updated: 2026-10-12*  
*Generated for Tensorlab EdgeAI — Texas Instruments*