# Radar Surface Classification

Classify the surface or material underneath/in front of a TI mmWave radar sensor — e.g. floor type, terrain, or ground material — directly from raw **FFT range-profile data**, with no point cloud or tracker involved. The shape of the energy return across range bins *is* the feature, because different materials reflect radar energy differently at different distances.

This guide covers the data processing configuration, training, quantization, and compilation for the `xWRL6432` target using TensorLab EdgeAI.

- **Example config:** [`config.yaml`](./config.yaml)
- **Dataset:** [radar_grass_surface_dataset.zip](https://software-dl.ti.com/C2000/esd/mcu_ai/datasets/radar_grass_surface_dataset.zip) (Grass vs Not Grass)

---

## Overview

This example demonstrates using TI mmWave sensors with machine learning for surface/material classification using the xWRL6432 sensor. The application classifies surface types such as:

- **Grass vs Not Grass** 
- **Wet vs Dry** 

The model uses the radar's on-device signal chain output — the FFT range profile (`bin0`...`binN` columns) — as input. No point cloud or track data is required.

---

## Quick Start

To train and evaluate the model using the provided dataset and preset configuration:

### Prerequisites

TO BE ADDED

### Steps

TO BE ADDED


The config uses the `Generic_130Input_Radar_SurfaceClassification` preset which configures the pipeline: range bin selection + windowing.

---

## YAML Configuration for Surface Classification

### Identify Your Data Source

| Case | Description | Action |
|------|-------------|--------|
| **A** | Data from TI Data Recorder / Visualizer (standard columns: `bin0`, `bin1`, `bin2`...) | Use preset directly |
| **B** | Data from other tools (different column names) | Rename columns first → see [Column Mapping](#column-mapping-for-non-ti-data) |
| **C** | Pre-cleaned / externally featurized data (no `bin*` columns needed) | Use `Generic_Radar_SimpleWindowOnly` preset |

### 3.1 Recommended: Use the Preset (Simplest)

```yaml
data_processing_feature_extraction:
  feature_extraction_name: "Generic_130Input_Radar_SurfaceClassification"
  variables: 1
```

**What this preset configures:**
- **Range Data**: Keeps bins 6–31 (26 bins), drops near-field (0–5) and far-field noise (≥32)
- **Window**: 5 frames, stride 0.2 (1-frame overlap), no quality filtering
- **Output**: 26 bins/frame × 5 frames = **130 features** → matches `RF_MLP_4L_19K`

### 3.2 Full Inline Configuration (Maximum Control)

```yaml
data_processing_feature_extraction:
  # Option 1: Start from preset and override specific parameters
  feature_extraction_name: "Generic_130Input_Radar_SurfaceClassification"
  
  # --- Common Overrides ---
  frame_size: 8                     # More temporal context (default: 5)
  stride_size: 0.5                  # Less overlap (default: 0.2)
  radar_range_bins_bounds: [4, 40]  # Wider range bin selection (default: [6, 31])
  
  # --- Feature Ordering ---
  # radar_feature_ordering: 'feature_major'  # Group by bin across frames
  
  variables: 1
```

```yaml
data_processing_feature_extraction:
  # Option 2: Fully inline (no preset) — for complete customization
  data_proc_transforms:
    - "Radar Range Data"
    - "Simple Window"
  
  # Range Data
  radar_range_bins_bounds: [6, 31]  # Keep bins 6 through 31 inclusive
  
  # Window parameters
  frame_size: 5
  stride_size: 0.2
  
  # Window Quality (optional - typically not needed for range data)
  # radar_frames_quality_index: 'per_window'
  # radar_min_points_per_window: 10
  radar_feature_ordering: 'frame_major'
```

### 3.3 For Pre-Cleaned Data (No Range Bins)

```yaml
data_processing_feature_extraction:
  feature_extraction_name: "Generic_Radar_SimpleWindowOnly"
  frame_size: 5           # Must match model input
  stride_size: 0.2
```

---

## Column Mapping for Non-TI Data

If your data uses different column names (Case B above), rename to TensorLab's expected format:

| Data Type | Your Column Pattern | → Rename To |
|-----------|---------------------|-------------|
| **Range Data** | `range_bin_0`, `fft_bin_0`, `fft_0` | `bin0` |
| | `range_bin_1`, `fft_bin_1`, `fft_1` | `bin1` |
| | `range_bin_2`, `fft_bin_2`, `fft_2` | `bin2` |
| | ... | `bin3`, `bin4`... `binN` |


---

## Complete Training Configuration

```yaml
common:
  task_type: "radar_surface_classification"
  target_device: "xWRL6432"

dataset:
  dataset_name: "radar_surface_classification"
  input_data_path: "https://software-dl.ti.com/C2000/esd/mcu_ai/datasets/radar_grass_surface_dataset.zip"
  split_type: "amongst_files"      # Whole files per split (no cross-file windows)
  split_factor: [0.7, 0.2, 0.1]    # Train / Val / Test

data_processing_feature_extraction:
  feature_extraction_name: "Generic_130Input_Radar_SurfaceClassification"

training:
  model_name: "RF_MLP_4L_19K"
  training_epochs: 100
  learning_rate: 0.0001
  optimizer: "sgd"
  output_int: false
  num_gpus: 0
```

### Dataset Split Methods

| `split_type` | When to Use |
|--------------|-------------|
| `amongst_files` | Each recording = one surface class (default, recommended) |
| `within_files` | Single recording contains multiple surface segments |

> **Why `amongst_files`?** Surface classification typically collects one surface type per recording session. `amongst_files` ensures no window spans two different surfaces, preventing label leakage.


---

## Model Architecture

| Model | Shape | Params | Description |
|-------|-------|--------|-------------|
| **`RF_MLP_4L_19K`** (default) | 4-layer MLP | ~19K | `BatchNorm → FC(64) → ReLU → BN → FC(32) → ReLU → BN → FC(16) → ReLU → BN → FC(num_classes)`. Plain MLP is appropriate — 130-value input (5 stacked 26-bin profiles) is low-dimensional, non-spatial. |

---

## Output Tensor

- **X**: `[N_windows, 130]` = 5 frames × 26 range bins (`frame_major` order)
- **Y**: `[N_windows]` integer class index — one per surface

---

## Quantization

By default, generic quantization runs if not disabled. To disable:

```yaml
training:
  quantization: 0
```

---

## Compilation

The ONNX model is compiled with **TVM + TI MCU NNC** backend into C artifacts for xWRL6432 (Cortex-M4):

```
compilation/artifacts/
├── tvmgen_default.h   # I/O structs, tvmgen_default_run() entry point
└── mod.a              # Precompiled static library
```

Firmware links `mod.a` and calls `tvmgen_default_run()`, feeding the 130-value flattened range-bin feature vector. The same range-bin-selection + windowing logic must be replicated on-device from the live FFT output.


---

## Key Reference Files

| Component | Location |
|-----------|----------|
| Model architecture | `tinyml-modelzoo/tinyml_modelzoo/models/radar.py` → `RF_MLP_4L_19K` |
| Radar Data Processing Pipeline (full reference) | `tinyml-modelzoo/docs/RADAR_PROCESSING_PIPELINE.md` |

---


## Need More Help?

- Search for issues or post questions on the [TI E2E Forum](https://e2e.ti.com/)
- See the full pipeline reference: [RADAR_PROCESSING_PIPELINE.md](tinyml-modelzoo/docs/RADAR_PROCESSING_PIPELINE.md)
- TensorLab EdgeAI documentation in the main repository

---

*Generated for Tensorlab EdgeAI — Texas Instruments*  
*Compatible with TI mmWave radar sensors*