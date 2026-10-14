# Radar Data Processing Pipeline — Complete Reference Guide

> **Complete guide to the Radar Data Processing Pipeline in Tensorlab**

---

## Table of Contents

1. [Quick Start: Identify Your Use Case Scenario](#quick-start-identify-your-use-case-scenario)
2. [All Configurable Parameters Reference](#all-configurable-parameters-reference)
3. [Use Case 1: Pre-Cleaned / Already Processed Datasets](#use-case-1-pre-cleaned--already-processed-datasets)
4. [Use Case 2: On-device Signal Chain Output from Data Capture Tools](#use-case-2-on-device-signal-chain-output-from-data-capture-tools)
5. [Use Case 3: Data from Other Sources (Column Renaming Required)](#use-case-3-data-from-other-sources-column-renaming-required)
6. [Frame Quality Filtering: per_radar_frame vs per_window](#frame-quality-filtering-per_radar_frame-vs-per_window)
7. [Feature Extraction Presets](#feature-extraction-presets)
8. [Inline Override: Customizing Presets in YAML](#inline-override-customizing-presets-in-yaml)
9. [Posture & Fall Detection: Complete Walkthrough](#posture--fall-detection-complete-walkthrough)
10. [Troubleshooting & Common Pitfalls](#troubleshooting--common-pitfalls)

---

## Quick Start: Identify Your Use Case Scenario

| Your Data Situation | Recommended Path |
|---------------------|------------------|
| **Already have per-frame features** (CSV columns = model input features, no bin*, point_*, track_* columns) | → [Use Case 1](#use-case-1-pre-cleaned--already-processed-datasets) — Use Generic_Radar_SimpleWindowOnly preset - add inline args - frame_size, stride_size for windowing customisation |
| **On-device Signal Chain Output from Data Recorder/Visualizer** (columns match expected names: bin*, point_x*, track_velocity_*, etc.) | → [Use Case 2](#use-case-2-raw-data-from-ti-unified-data-capture-tools) — Use Generic_248Input_Posture_and_Fall_Detection or Generic_392Input_PointCloudWithTrack_SimpleWindow |
| **Raw data from other tools / custom capture** (columns don't match TensorLab names) | → [Use Case 3](#use-case-3-raw-data-from-other-sources-column-renaming-required) — Rename columns first, then Use Case 2 |
| **Surface Classification** (FFT range profile only) | → Use Generic_130Input_Radar_SurfaceClassification preset - considering 26 bins per radar frame and a window size = 5, can be updated using necessary inline parameters in yaml |

---

## All Configurable Parameters Reference

### Data Processing Feature Extraction Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| feature_extraction_name | str | None | Preset name from constants.py |
| data_proc_transforms | list | [] | **Main pipeline gate** — tokens: "Radar Range Data", "Radar Point Cloud Data", "Radar Track Data", "Scaling", "Simple Window" |
| frame_size | int | 1 | **Frames per window (critical for model input shape)** |
| stride_size | float | 0.125 | Stride as fraction of frame_size (0-1) |

### Radar-Specific Parameters

> **Note**: Using a preset via feature_extraction_name is highly recommended. Any parameter whose value matches the preset's default **does not need to be specified in your YAML**. Only override parameters that differ from the preset defaults.

| Parameter | Type | Description | When to Use / Optional? |
|-----------|------|-------------|------------------------|
| radar_range_bins_bounds | list[2] | [min_bin, max_bin] for range bin selection | **Required** if "Radar Range Data" in data_proc_transforms. Set to crop near/far bins. Optional: omit to keep all bins. |
| radar_point_cloud_data_selector | list[str] | Point features to keep: x, y, z, doppler, snr | **Required** if "Radar Point Cloud Data" active/present in data_proc_transforms. Choose subset to reduce dimensionality (e.g., y,z,snr for fall detection). |
| radar_point_cloud_coordinate_system | str | absolute or relative | **Required** if "Radar Point Cloud Data" in data_proc_transforms. Use relative to make point cloud position-invariant. |
| radar_point_cloud_relative_coordinate_filter | list[str] | Axes to make relative: x, y, z | **Required** when coordinate_system: relative. Specifies which axes to transform (e.g., y for distance-only). |
| radar_point_cloud_spatial_filter_bounds | list[6] | [min_x, max_x, min_y, max_y, min_z, max_z] | **Optional**. Spatial gate to drop points outside sensing zone. Use None to disable. |
| radar_point_cloud_non_spatial_filter_bounds | list[4] | [min_dop, max_dop, min_snr, max_snr] | **Optional**. Doppler/SNR gate to drop noisy/implausible points. Use None to disable. |
| radar_track_data_selector | list[str] | Track features: velocity, x, y, z, acceleration | **Required** if "Radar Track Data" active. Select subset to control features/frame (e.g., z,velocity,acceleration). |
| radar_track_velocity_bounds | list[2] | [min_vel, max_vel] | **Optional**. Filter out physically implausible track velocities (ghost tracks). Eg: [-10, 10] m/s. |
| radar_track_acceleration_bounds | list[2] | [min_acc, max_acc] | **Optional**. Filter out implausible accelerations. Eg: [-10, 10] m/s². |
| radar_track_spatial_filter_bounds | list[6] | [min_x, max_x, min_y, max_y, min_z, max_z] | **Optional**. Gate track position to sensing zone. Use None to disable. |
| radar_data_scaling_track_velocity | float | Scale factor for track_velocity_* columns | **Optional**. Independent of "Radar Track Data". Use when velocity magnitude is too small vs position features (e.g., 30). |
| radar_max_points_per_frame | int | Max points per frame after selection | **Required** if "Radar Point Cloud Data" or "Simple Window" active. |
| radar_min_points_per_frame | int | Min real points required per frame (per_radar_frame mode) | **Required** only if radar_frames_quality_index: per_radar_frame. Set >0 to drop sparse frames. |
| radar_min_points_per_window | int | Min total real points across window (per_window mode) | **Required** only if radar_frames_quality_index: per_window. Set >0 to drop low-quality windows. |
| radar_point_cloud_structure_strategy | str | Point selection strategy | **Required** if point cloud active. Options: highest only, lowest only, highest and lowest. |
| radar_point_cloud_sort_by | str | Feature to sort points by: x, y, z, doppler, snr | **Required** if point cloud active. Determines sort order for point selection strategy. |
| radar_frames_quality_index | str | per_radar_frame, per_window, or unset | **Optional**. Quality filtering mode. None = disabled. per_radar_frame works standalone. per_window needs "Simple Window". |
| radar_feature_ordering | str | frame_major or feature_major | **Optional**. Flattening order for window tensor. Must match model expectation. |

---

### Preset Default Values Reference

The table below shows the default values for each parameter across all built-in presets. Use this to know what values a preset provides — any value you want to change must be overridden inline in your YAML.

**Note**: Parameters related to point cloud processing (radar_point_cloud_*, radar_max_points_per_frame, radar_point_cloud_structure_strategy, radar_point_cloud_sort_by) are **only applicable when "Radar Point Cloud Data" is in data_proc_transforms**. 

| Parameter | Generic_392Input_PointCloudWithTrack_SimpleWindow | Generic_Radar_SimpleWindowOnly | Generic_130Input_Radar_SurfaceClassification | Generic_248Input_Posture_and_Fall_Detection |
|-----------|-----------------------------------------------------|-----------------------------------|------------------------------------------------|-----------------------------------------------|
| data_proc_transforms | [Radar Point Cloud Data, Radar Track Data, Simple Window] | [Simple Window] | [Radar Range Data, Simple Window] | [Radar Point Cloud Data, Radar Track Data, Simple Window] |
| frame_size | 8 | 8 | 5 | 8 |
| stride_size | 0.125 | 0.125 | 0.2 | 0.125 |
| radar_range_bins_bounds | — | — | [6, 31] | — |
| radar_point_cloud_data_selector | [x,y,z,doppler,snr] | — | — | [y,z,snr] |
| radar_point_cloud_coordinate_system | relative | — | — | relative |
| radar_point_cloud_relative_coordinate_filter | [x,y,z] | — | — | [y] |
| radar_point_cloud_spatial_filter_bounds | [None,None,None,None,None,None] | — | — | [None,None,-10,10,-10,10] |
| radar_point_cloud_non_spatial_filter_bounds | [None,None,None,None] | — | — | [None,None,None,None] |
| radar_track_data_selector | [x,y,z,velocity,acceleration] | — | — | [z,velocity,acceleration] |
| radar_track_velocity_bounds | [None,None] | — | — | [-10,10] |
| radar_track_acceleration_bounds | [None,None] | — | — | [-10,10] |
| radar_track_spatial_filter_bounds | [None,None,None,None,None,None] | — | — | [None,None,None,None,-10,10] |
| radar_data_scaling_track_velocity | — | — | — | — |
| radar_max_points_per_frame | 8 | — | — | 8 |
| radar_min_points_per_frame | — | — | — | — |
| radar_min_points_per_window | 12 | — | — | 12 |
| radar_point_cloud_structure_strategy | highest and lowest | — | — | highest and lowest |
| radar_point_cloud_sort_by | z | — | — | z |
| radar_frames_quality_index | per_window | — | None | per_window |
| radar_feature_ordering | frame_major | — | frame_major | frame_major |

> **Legend**: — = parameter not used by this preset (stage not active). Parameters not listed in a preset use the pipeline's global defaults. Point cloud parameters marked — for presets without "Radar Point Cloud Data" are irrelevant for those presets.

---

## Use Case 1: Pre-Cleaned / Already Processed Datasets

> **Scenario**: You already have a dataset where each row = one frame with the exact features your model expects. No bin*, point_*, track_* columns exist. You just need **windowing** (sliding window with stride).

### When to Use
- Features already extracted externally (MATLAB, Python, custom scripts)
- CSV columns = final per-frame features (e.g., 31 columns per frame)
- No radar-specific column names present
- You only need: frame_size, stride, optional quality filtering

### Recommended: Use the Preset

```yaml
data_processing_feature_extraction:
  feature_extraction_name: "Generic_Radar_SimpleWindowOnly"
  # That's it! Only "Simple Window" stage runs.
```

### Inline Config (Full Control)

```yaml
data_processing_feature_extraction:
  data_proc_transforms:
    - "Simple Window"          # ONLY this stage runs
  frame_size: 8                # Frames per window (MUST match model)
  stride_size: 0.5             # Stride as fraction of frame_size (0-1)
  # Feature ordering in flattened window
  radar_feature_ordering: 'frame_major'  # or 'feature_major'
```

### Important Notes
| Parameter | Requirement |
|-----------|-------------|
| Column names | **Don't matter** — pipeline treats all columns as features |
| Quality filtering | Only supportive in case of column names present |

---

## Use Case 2: On-device Signal Chain Output from Data Capture Tools

> **Scenario**: Data recorded using TI's mmWave Data Recorder / Visualizer tools. Column names **already match** TensorLab's expected format.

### Expected Column Names (Case-Sensitive)

| Data Type | Column Pattern | Example Columns |
|-----------|----------------|-----------------|
| **Range Data** | bin0, bin1, bin2... binN | bin0 through bin63 (FFT range profile) |
| **Point Cloud** | point_x{idx}, point_y{idx}, point_z{idx}, point_doppler{idx}, point_snr{idx} | point_x0, point_y0, point_z0, point_doppler0, point_snr0, point_x1... |
| **Track Data** | track_id or tid | **Required for ghost track filtering** |
| | track_x, track_y, track_z | Track position |
| | track_velocity_x, track_velocity_y, track_velocity_z | Track velocity |
| | track_acc_x, track_acc_y, track_acc_z | Track acceleration |

### Ghost Track Filtering
- **If track_id/tid exists**: Pipeline validates each frame.
- **If track_id/tid missing**: Pipeline **continues with all tracks treated as valid** (no frames dropped). Logs a warning.

### Recommended Presets

#### For Posture & Fall Detection (Point Cloud + Track)
```yaml
data_processing_feature_extraction:
  feature_extraction_name: "Generic_248Input_Posture_and_Fall_Detection"
  # Result: 31 features/frame x 8 frames = 248 input features
  # Matches RF_DualStream_PointCloud_19K default
```

#### For Maximum Features (All 5 PC features + All Track features)
```yaml
data_processing_feature_extraction:
  feature_extraction_name: "Generic_392Input_PointCloudWithTrack_SimpleWindow"
  # Result: 49 features/frame x 8 frames = 392 input features
```

#### For Surface Classification (Range Only)
```yaml
data_processing_feature_extraction:
  feature_extraction_name: "Generic_130Input_Radar_SurfaceClassification"
  # Result: 26 range bins (6-31) x 5 frames = 130 input features
```

---

## Use Case 3: Data from Other Sources (Column Renaming Required)

> **Scenario**: You have raw radar data from non-TI tools or different column naming conventions. **You must rename columns to match TensorLab's expected names** before the pipeline can process them.

##### Required Column Mapping can be checked in above table. Make a note its case-sensitive.



> Future Releases will support automation of Columns Renaming (Through a Config Builder Tool)

---

## Frame Quality Filtering: per_radar_frame vs per_window

### Comparison Table

| Mode | Parameters | Drops | Requires "Simple Window" |
|------|------------|-------|-------------------------|
| **unset / empty** | N/A | Nothing | No |
| **per_radar_frame** | radar_min_points_per_frame | Individual frames with < N real points | **No** (works standalone!) |
| **per_window** | radar_min_points_per_window | Entire windows with < N total points across all frames | **Yes** (needs windowing) |

### Key Insight: per_radar_frame Works Standalone!

> per_radar_frame quality filtering now works **independently** of the "Simple Window" transform. You can use it:
> - With just feature extraction stages (no windowing)
> - With pre-cleaned data having required point column names

```yaml
# Example: Quality filtering WITHOUT windowing
data_processing_feature_extraction:
  data_proc_transforms:
    - "Radar Point Cloud Data"
    - "Radar Track Data"
    # NO "Simple Window"!
  radar_frames_quality_index: 'per_radar_frame'
  radar_min_points_per_frame: 3
  # ... other params
```

### Visual: How Quality Modes Work

```
per_radar_frame (min_points=3):
Frame 0: 5 pts -> KEPT
Frame 1: 2 pts -> DROPPED
Frame 2: 4 pts -> KEPT
Frame 3: 1 pts -> DROPPED
Result: Frames [0, 2] kept (gaps allowed)

per_window (min_points=12, frame_size=4):
Window [0,1,2,3]: 5+2+4+1 = 12 pts -> KEPT
Window [1,2,3,4]: 2+4+1+0 = 7 pts  -> DROPPED
Window [2,3,4,5]: 4+1+0+6 = 11 pts -> DROPPED
Result: Only windows meeting threshold kept (consecutive frames required)
```

---

## Feature Extraction Presets

Presets live in tinyml-modelmaker/tinyml_modelmaker/ai_modules/radar/constants.py under FEATURE_EXTRACTION_PRESET_DESCRIPTIONS.

### Preset 1: Generic_392Input_PointCloudWithTrack_SimpleWindow
**Use Case**: Complete raw CSV processing with full feature extraction and quality filtering.  
**Resulting Input Tensor**: `(N, 8 x 49) = (N, 392)` — 5 PC features x 8 points + 9 track features = 49/frame.

```python
Generic_392Input_PointCloudWithTrack_SimpleWindow=dict(
    data_processing_feature_extraction=dict(
        data_proc_transforms=['Radar Point Cloud Data', 'Radar Track Data', 'Simple Window'],
        frame_size=8,
        stride_size=0.125,
        radar_point_cloud_data_selector=['x','y', 'z','doppler','snr'],
        radar_point_cloud_coordinate_system='relative',
        radar_point_cloud_relative_coordinate_filter=['x','y''z'],
        radar_point_cloud_spatial_filter_bounds=[None, None, None, None, None, None],
        radar_track_data_selector=['x', 'y', 'z', 'velocity', 'acceleration'],
        radar_track_velocity_bounds=[None, None],
        radar_track_acceleration_bounds=[None, None],
        radar_track_spatial_filter_bounds=[None, None, None, None, None, None],
        radar_max_points_per_frame=8,
        radar_min_points_per_window=12,
        radar_point_cloud_structure_strategy='highest only',
        radar_point_cloud_sort_by='z',
        radar_frames_quality_index='per_window',
        radar_feature_ordering='frame_major',
    ),
    common=dict(task_type=TASK_TYPE_RADAR_POINT_CLOUD_CLASSIFICATION), )
```

---

### Preset 2: Generic_Radar_SimpleWindowOnly
**Use Case**: Pre-cleaned/refined datasets where CSV already has correct per-frame features. Only windowing is needed.

```python
Generic_Radar_SimpleWindowOnly = dict(
    data_processing_feature_extraction=dict(
        data_proc_transforms=['Simple Window'],
        frame_size=8,
        stride_size=0.125,
    ),
    common=dict(task_type=TASK_TYPE_RADAR_POINT_CLOUD_CLASSIFICATION),
)
```
**Notes**:
- No range bin, point cloud, track, or scaling stages run
- Input CSV must already have the correct per-frame feature columns

---

### Preset 3: Generic_130Input_Radar_SurfaceClassification
**Use Case**: Surface Classification using FFT range bins only.

```python
Generic_130Input_Radar_SurfaceClassification = dict(
    data_processing_feature_extraction=dict(
        data_proc_transforms=['Radar Range Data', 'Simple Window'],
        radar_range_bins_bounds=[6, 31],
        frame_size=5,
        stride_size=0.2,
        radar_frames_quality_index=None,
        variables=1,
    ),
    common=dict(task_type=TASK_TYPE_RADAR_SURFACE_CLASSIFICATION),
)
```
**Result**: 26 bins (6-31 inclusive) x 5 frames = 130 features

---

### Preset 4: Generic_248Input_Posture_and_Fall_Detection **RECOMMENDED FOR POSE/FALL**
**Use Case**: Optimized for human posture and fall detection (matches RF_DualStream_PointCloud_19K).

```python
Generic_248Input_Posture_and_Fall_Detection = dict(
    data_processing_feature_extraction=dict(
        data_proc_transforms=['Radar Point Cloud Data', 'Radar Track Data', 'Simple Window'],
        frame_size=8,
        stride_size=0.125,
        radar_point_cloud_data_selector=['y', 'z','snr'],
        radar_point_cloud_coordinate_system='relative',
        radar_point_cloud_relative_coordinate_filter=['y'],
        radar_point_cloud_spatial_filter_bounds=[None, None, -10.0, 10.0, -10.0, 10.0],
        radar_track_data_selector=['z', 'velocity', 'acceleration'],
        radar_track_velocity_bounds=[-10.0, 10.0],
        radar_track_acceleration_bounds=[-10.0, 10.0],
        radar_track_spatial_filter_bounds=[None, None, None, None, -10.0, 10.0],
        radar_max_points_per_frame=8,
        radar_min_points_per_window=12,
        radar_point_cloud_structure_strategy='highest and lowest',
        radar_point_cloud_sort_by='z',
        radar_frames_quality_index='per_window',
        radar_feature_ordering='frame_major',
    ),
    common=dict(task_type=TASK_TYPE_RADAR_POINT_CLOUD_CLASSIFICATION),
)
```

**Per-Frame Feature Layout (31 features)**:
| Index | Feature | Source |
|-------|---------|--------|
| 0 | track_height_z | Track |
| 1 | track_velocity_x | Track |
| 2 | track_velocity_y | Track |
| 3 | track_velocity_z | Track |
| 4 | track_acc_x | Track |
| 5 | track_acc_y | Track |
| 6 | track_acc_z | Track |
| 7-30 | 8 points x [y, z, snr] | Point Cloud (24 features) |

**Total per frame**: 7 (track) + 24 (8 points x 3) = **31 features**  
**Window (8 frames)**: 31 x 8 = **248 features** → Matches model input!

---

## Inline Override: Customizing Presets in YAML

> **Power User Feature**: Use a preset as base, then override specific parameters inline.

### How It Works
```yaml
data_processing_feature_extraction:
  feature_extraction_name: "Generic_248Input_Posture_and_Fall_Detection"
  
  # Override ANY parameter from the preset:
  frame_size: 16                    # Change window size
  stride_size: 0.25                 # Change stride
  radar_min_points_per_window: 20   # Stricter quality
  radar_max_points_per_frame: 12    # More points per frame
  radar_point_cloud_data_selector: ['x', 'y', 'z', 'snr']  # Add x coordinate
```



### Override Example: Custom Fall Detection Config
```yaml
data_processing_feature_extraction:
  feature_extraction_name: "Generic_248Input_Posture_and_Fall_Detection"
  
  # Custom: 16 frames, no overlap, stricter quality, all point features
  frame_size: 16
  stride_size: 1.0
  radar_min_points_per_window: 20
  radar_point_cloud_data_selector: ['x', 'y', 'z', 'doppler', 'snr']
  radar_point_cloud_sort_by: 'y'  
  
  # Keep rest from preset (relative coords, track features, etc.)
```


*Generated for Tensorlab EdgeAI — Making mmWave Radar ML Accessible*