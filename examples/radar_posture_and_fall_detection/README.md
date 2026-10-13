# Radar Posture & Fall Detection

Classify a person's posture (standing, sitting, walking, falling, laying) from a TI mmWave radar sensor, using the point cloud + track outputs the radar's on-device signal chain already produces — no raw ADC support.

This guide covers the data processing configuration, training, quantization, and compilation for the `xWRL6432` target using TensorLab EdgeAI.

- **Example config:** [`config.yaml`](./config.yaml)
- **Dataset:** [posture_and_fall_detection_dataset.zip](https://software-dl.ti.com/C2000/esd/mcu_ai/datasets/posture_and_fall_detection_dataset.zip)

---

## Overview

This example config demonstrates retraining through Tensorlab and final compiled artifacts generation for using TI mmWave sensors with machine learning for human posture and fall classification using the xWRL6432 sensor. The application classifies five pose states:

- **Falling** (class 0)
- **Laying** (class 1)
- **Sitting** (class 2)
- **Standing** (class 3)
- **Walking** (class 4)

The model uses the radar's on-device signal chain outputs — point cloud detections and track data — as input. No raw ADC access is required.

---

## Quick Start

To train and evaluate the model using the provided dataset and preset configuration:

### Prerequisites

TO BE ADDED

### Steps

TO BE ADDED


---
### Data Collection Procedure (for Custom Data)

1. Arrange environment per Physical Setup above
2. Configure mmWave Data Recorder with appropriate `.cfg` file
3. Use hotkey-triggered recording to capture only desired motion segments:
   - Press hotkey → start recording
   - Perform pose/motion
   - Press hotkey → stop recording
4. Save each class to separate subdirectory under `dataset/classes/<class_name>/`

---

## YAML Configuration for Posture & Fall Detection

### Identify Your Data Source

| Case | Description | Action |
|------|-------------|--------|
| **A** | Data from TI Data Recorder / Visualizer (standard columns: `point_x*`, `track_id`, `track_velocity_*`) | Use preset directly |
| **B** | Data from other tools / custom firmware (different column names) | Rename columns first → see [Column Mapping](#column-mapping-for-non-ti-data) |
| **C** | Pre-cleaned / externally featurized data (no `point_*`, `track_*`, `bin*` columns) | Use `Generic_Radar_SimpleWindowOnly` preset |

### 3.1 Recommended: Use the Preset (Simplest)

```yaml
data_processing_feature_extraction:
  feature_extraction_name: "Generic_248Input_Posture_and_Fall_Detection"
  variables: 1
```

**What this preset configures:**
- **Point Cloud**: Selects `y` (distance), `z` (height), `snr` — makes `y` relative to track centroid
- **Track**: Selects `z` (height), `velocity`, `acceleration` — applies ±10 m/s and ±10 m/s² bounds
- **Window**: 8 frames, stride 0.125 (1-frame overlap), per-window quality (min 12 points)
- **Output**: 31 features/frame × 8 frames = **248 features** → matches `RF_DualStream_PointCloud_19K`

### 3.2 Full Inline Configuration (Maximum Control)

```yaml
data_processing_feature_extraction:
  # Option 1: Start from preset and override specific parameters
  feature_extraction_name: "Generic_248Input_Posture_and_Fall_Detection"
  
  # --- Common Overrides ---
  frame_size: 16                    # More temporal context (default: 8)
  stride_size: 0.5                  # Less overlap, faster training (default: 0.125)
  radar_min_points_per_window: 20   # Stricter quality (default: 12)
  radar_max_points_per_frame: 12    # More points per frame (default: 8)
  
  # --- Point Cloud Overrides ---
  radar_point_cloud_data_selector: ['x', 'y', 'z', 'snr']  # Add x (azimuth)
  radar_point_cloud_sort_by: 'y'    # Sort by distance instead of height
  
  # --- Quality Mode Override ---
  # radar_frames_quality_index: 'per_radar_frame'  # Per-frame instead of per-window
  # radar_min_points_per_frame: 3                  # Min points per frame
  
  # --- Disable Quality ---
  # radar_frames_quality_index: None
  
  variables: 1
```

```yaml
data_processing_feature_extraction:
  # Option 2: Fully inline (no preset) — for complete customization
  data_proc_transforms:
    - "Radar Point Cloud Data"
    - "Radar Track Data"
    - "Simple Window"
  
  # Window parameters
  frame_size: 8
  stride_size: 0.125
  
  # Point Cloud
  radar_point_cloud_data_selector: ['y', 'z', 'snr']
  radar_point_cloud_coordinate_system: 'relative'
  radar_point_cloud_relative_coordinate_filter: ['y']
  radar_point_cloud_spatial_filter_bounds: [None, None, -10.0, 10.0, -10.0, 10.0]
  radar_point_cloud_non_spatial_filter_bounds: [None, None, None, None]
  radar_max_points_per_frame: 8
  radar_point_cloud_structure_strategy: 'highest and lowest'
  radar_point_cloud_sort_by: 'z'
  
  # Track
  radar_track_data_selector: ['z', 'velocity', 'acceleration']
  radar_track_velocity_bounds: [-10.0, 10.0]
  radar_track_acceleration_bounds: [-10.0, 10.0]
  radar_track_spatial_filter_bounds: [None, None, None, None, -10.0, 10.0]
  
  # Window Quality
  radar_min_points_per_window: 12
  radar_frames_quality_index: 'per_window'
  radar_feature_ordering: 'frame_major'
  
  variables: 1
```

### 3.3 For Pre-Cleaned Data (No Radar Columns)

```yaml
data_processing_feature_extraction:
  feature_extraction_name: "Generic_Radar_SimpleWindowOnly"
  frame_size: 8           # Must match model input
  stride_size: 0.5
  # Optional: per-frame quality if your data has standard point columns
  # radar_frames_quality_index: 'per_radar_frame'
  # radar_min_points_per_frame: 3
```

---
## Expected Column Names (Case-Sensitive)

If your data uses different column names (Case B above), rename to TensorLab's expected format:

| Data Type | Column Pattern | Example Columns |
|-----------|----------------|-----------------|
| **Range Data** | `bin0`, `bin1`, `bin2`... `binN` | `bin0` through `bin63` (FFT range profile) |
| **Point Cloud** | `point_x{idx}`, `point_y{idx}`, `point_z{idx}`, `point_doppler{idx}`, `point_snr{idx}` | `point_x0`, `point_y0`, `point_z0`, `point_doppler0`, `point_snr0`, `point_x1`... |
| **Track Data** | `track_id` or `tid` | Required for ghost track filtering |
| | `track_x`, `track_y`, `track_z` | Track position |
| | `track_velocity_x`, `track_velocity_y`, `track_velocity_z` | Track velocity |
| | `track_acc_x`, `track_acc_y`, `track_acc_z` | Track acceleration |


---

## Complete Training Configuration

```yaml
common:
  task_type: "radar_point_cloud_classification"
  target_device: "xWRL6432"
  training_device: "cuda"

dataset:
  dataset_name: "radar_human_pose_detection"
  input_data_path: "https://software-dl.ti.com/C2000/esd/mcu_ai/datasets/posture_and_fall_detection_dataset.zip"
  split_type: "within_files"      # Split inside each recording file
  split_factor: [0.7, 0.2, 0.1]   # Train / Val / Test

data_processing_feature_extraction:
  feature_extraction_name: "Generic_248Input_Posture_and_Fall_Detection"
  variables: 1

training:
  model_name: "RF_DualStream_PointCloud_19K"
  quantization: 1                         # 0 = No Quantization, 1 = Generic Quantization
  quantization_method: "QAT"
  quantization_weight_bitwidth: 8
  quantization_activation_bitwidth: 8
  training_epochs: 250
  learning_rate: 0.001
  optimizer: "adamw"
  lr_scheduler: "CosineAnnealingLR"
  weight_decay: 0
  output_int: false
  num_gpus: 0
```

### Dataset Split Methods

| `split_type` | When to Use |
|--------------|-------------|
| `within_files` | Single long recording|
| `amongst_files` | Each recording session = one posture class (Recommneded with atleast good amount of dataset) | 

---


### Simple Window (`"Simple Window"`)
- **Windowing**: 8 consecutive frames → 1 training sample (248 features)
- **Stride**: 0.125 = 1 frame overlap → dense sampling for short fall events
- **Quality**: `per_window` drops windows with < 12 total real points across 8 frames
- **Consecutive check**: Rejects windows spanning gaps from dropped ghost frames

---

## Model Architectures

| Model | Shape | Params | Description |
|-------|-------|--------|-------------|
| **`RF_DualStream_PointCloud_19K`** (default) | Dual-stream MLP | ~19K | Splits each frame into **position stream** (point coords + track height + accel = 28 features) and **velocity stream** (3 track velocity components). Separate BN → Linear → ReLU → BN branches, then fusion. Natural fit: shape vs. motion. |
| `RF_MLP_4L_19K` | Plain 4-layer MLP | ~19K | BN → FC(64) → ReLU → BN → FC(32) → ReLU → BN → FC(16) → ReLU → BN → FC(num_classes) |

---

## Output Tensor

- **X**: `[N_windows, 248]` = 8 frames × 31 features/frame (`frame_major` order)
- **Y**: `[N_windows]` integer class index:
  - `0` = Falling
  - `1` = Laying
  - `2` = Standing
  - `3` = Sitting
  - `4` = Walking

---

## Quantization

This example uses **QAT (Quantization-Aware Training)**:
- `quantization: 1` (Generic quantization)
- `quantization_method: "QAT"`
- 8-bit weights and activations
- Two phases: float training (250 epochs) → quantization fine-tune
- `output_int: false` — exported model dequantizes output to float (radar supports float)

---

## Compilation

The quantized/float ONNX model is compiled with **TVM + TI MCU NNC** backend into C artifacts for xWRL6432 (Cortex-M4):

```
compilation/artifacts/
├── tvmgen_default.h   # I/O structs, tvmgen_default_run() entry point
└── mod.a              # Precompiled static library
```

Firmware links against `mod.a` and calls `tvmgen_default_run()` with the 248-value feature vector. The same preprocessing must be replicated on-device (or done by the sensor's onboard tracker/CFAR chain).


---

## Key Reference Files

| Component | Location |
|-----------|----------|
| Model architectures | `tinyml-modelzoo/tinyml_modelzoo/models/radar.py` |
| Radar Data Processing Pipeline (full reference) | [Radar Data Processing Pipeline Reference Guide](tinyml-modelzoo/docs/RADAR_PROCESSING_PIPELINE.md) |


---

## Need More Help?

- Search for issues or post questions on the [TI E2E Forum](https://e2e.ti.com/)
- See the full pipeline reference: [RADAR_PROCESSING_PIPELINE.md](tinyml-modelzoo/docs/RADAR_PROCESSING_PIPELINE.md)
- TensorLab EdgeAI documentation in the main repository - For More Clarity on Setup Environment

---

*Generated for Tensorlab EdgeAI — Texas Instruments*  
*Compatible with TI mmWave radar sensors*