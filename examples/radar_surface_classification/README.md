# Tensorlab - Radar Surface Classification Guide

This guide documents the **complete radar range-data preprocessing, training, quantization, compilation, and deployment pipeline** for **surface/material classification** using TI mmWave radar sensors (e.g., distinguishing floor surfaces, terrain types, or materials from FFT range-bin signatures).

---

## Quick Start

| Your Starting Point | Recommended Flow | Description |
|---|---|---|
| **Raw CSV from TI capture tools** (`bin*` columns present) | Full pipeline: Range Bin Selection → Windowing → Training → Quantization → Compilation | Default preset |
| **Already trained ONNX model** | Quantization → Compilation → Deployment | Skip training |

> **Example config:** See `xwrl6432_surface_classification.yaml` for a working baseline for radar surface classification.

---

## Complete Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        TRAINING PHASE (Python)                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Raw CSV/NPY/PKL/TXT (bin0, bin1, ... binN columns = FFT range profile)   │
│         │                                                                   │
│         ▼                                                                   │
│  ┌──────────────────┐     "Radar Range Data"        → radar_range_bins_   │
│  │ Stage 1: Range   │ ──────────────────────────────▶ bounds               │
│  │ Bin Selection    │                                                        │
│  └────────┬─────────┘                                                        │
│           │                                                                   │
│           ▼                                                                   │
│  ┌──────────────────┐     "Simple Window"           → frame_size,          │
│  │ Stage 5: Window  │ ──────────────────────────────▶ stride_size,          │
│  │ (+ Quality Index)│                                 radar_frames_         │
│  └────────┬─────────┘                                 quality_index         │
│           │                                                                   │
│           ▼                                                                   │
│  ┌──────────────────┐                                                        │
│  │ Output Tensors   │  X: [N, frame_size × num_range_bins]                  │
│  │                  │  Y: [N] class indices                                 │
│  └────────┬─────────┘                                                        │
│           │                                                                   │
│           ▼                                                                   │
│  ┌──────────────────┐     Model: RF_MLP_4L_19K                              │
│  │ Training         │ ──────────────────────────────▶ (or RF_DualStream_   │
│  │ (PyTorch)        │                                   PointCloud_19K)     │
│  └────────┬─────────┘                                                        │
│           │                                                                   │
│           ▼                                                                   │
│  ┌──────────────────┐                                                        │
│  │ ONNX Export      │  model.onnx (float32)                                 │
│  └────────┬─────────┘                                                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       QUANTIZATION PHASE (Python)                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  float32 ONNX ──▶ TI Quantization (QAT/PTQ) ──▶ quantized ONNX (int8)      │
│       │                  (tinyml-modelmaker)         │                     │
│       │                         │                      │                     │
│       │                         ▼                      │                     │
│       │              ┌──────────────────┐              │                     │
│       │              │ Calibration Data │              │                     │
│       │              │ (representative  │              │                     │
│       │              │  samples)        │              │                     │
│       │              └──────────────────┘              │                     │
│       │                                                │                     │
│       ▼                                                ▼                     │
│  training/base/                                training/quantization/        │
│  model.onnx                                    model.onnx                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      COMPILATION PHASE (TVM + TI NNC)                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  quantized ONNX ──▶ TVM Compiler ──▶ C Artifacts                           │
│       │                (TI MCU NNC)     │                                  │
│       │                         ┌──────┴──────┐                            │
│       │                         ▼             ▼                            │
│       │                  ┌─────────┐    ┌─────────┐                        │
│       │                  │ tvmgen_ │    │ lib0.c  │                        │
│       │                  │default.h│    │ lib1.c  │                        │
│       │                  │ mod.a   │    │         │                        │
│       │                  └─────────┘    └─────────┘                        │
│       │                                                                   │
│       ▼                                                                   │
│  compilation/artifacts/                                                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Key principle:** Each preprocessing stage runs **IFF its token string is present** in `data_proc_transforms`. Order in the list does not matter — the internal signal-chain order is fixed (Range Data → Point Cloud → Track → Scaling → Window).

> **Note:** Surface classification only uses Stage 1 (Range Data) + Stage 5 (Window). Point Cloud / Track / Scaling stages are not used because this task works directly on FFT range-bin amplitudes, not detected points/tracks.

---

## Preset (Recommended)

| Preset Name | Stages Enabled | Input Requirements | Range Bins | Frame Size | Total Input Size |
|---|---|---|---|---|---|
| `Generic_130Input_Radar_RangeClassification` | 1, 5 (Range Data, Window) | TI capture tool CSV (`bin*` columns) | 26 (bins 6–31) | 5 | 130 (26 × 5) |

### Preset Detail: `Generic_130Input_Radar_RangeClassification`

- **Range Data:** keeps `bin6`...`bin31` inclusive (26 bins) — crops near-field/far-field clutter outside this range
- **Window:** `frame_size=5`, `stride_size=0.2` (stride = 1 frame, dense sliding window)
- **Quality Index:** disabled (`radar_frames_quality_index=None`) — no point-count-based frame/window dropping, since there's no point cloud in this pipeline

> Defined in `tinyml-modelmaker/tinyml_modelmaker/ai_modules/radar/constants.py` → `FEATURE_EXTRACTION_PRESET_DESCRIPTIONS['Generic_130Input_Radar_RangeClassification']`.

---

## Complete Parameter Reference

### Core Pipeline Control

| Parameter | Type | Default (Preset) | Description |
|---|---|---|---|
| `feature_extraction_name` | string | `Generic_130Input_Radar_RangeClassification` | Preset name (overrides inline params if set) |
| `data_proc_transforms` | list[string] | `["Radar Range Data", "Simple Window"]` | Tokens to enable stages. Valid: `["Radar Range Data", "Radar Point Cloud Data", "Radar Track Data", "Scaling", "Simple Window"]` |
| `frame_size` | int | 5 | Window size = number of radar frames per tensor window |
| `stride_size` | float (0–1) | 0.2 | Stride as fraction of `frame_size`. Stride = `int(frame_size × stride_size)` |

---

### Stage 1: Radar Range Data — FFT Bin Selection

**Token:** `"Radar Range Data"`

| Parameter | Type | Default (Preset) | Description |
|---|---|---|---|
| `radar_range_bins_bounds` | [int, int] or null | `[6, 31]` | Inclusive `[min_bin, max_bin]` indices to keep from `bin*` columns. `null` = keep all bins. |

**Behavior:** CSV must contain `bin0`, `bin1`, ..., `binN` columns (one per FFT range bin, per TI capture tool format). The stage selects only columns whose numeric suffix falls within `[min_bin, max_bin]` inclusive, casts to float, and drops everything else. With `[6, 31]`, this keeps 26 bins (`bin6`...`bin31`), cropping clutter close to the radar (bins 0–5) and beyond the region of interest (bins ≥32).

---

### Stage 5: Simple Window — Sliding Window (+ Optional Quality Index)

**Token:** `"Simple Window"`

#### Windowing

| Parameter | Type | Default (Preset) | Description |
|---|---|---|---|
| `frame_size` | int | 5 | Number of radar frames per window (set at top level) |
| `stride_size` | float (0–1) | 0.2 | Stride fraction (set at top level). Stride = `int(frame_size × stride_size)` = 1 frame |

#### Quality Index Modes

| Parameter | Type | Default (Preset) | Modes |
|---|---|---|---|
| `radar_frames_quality_index` | string or null | `null` (disabled) | `'per_radar_frame'`, `'per_window'`, `null` |

> For range classification, quality index is disabled by default since the pipeline has no point cloud / track data to validate — every FFT frame is already a complete, valid measurement.

#### Feature Ordering

| Parameter | Type | Default | Description |
|---|---|---|---|
| `radar_feature_ordering` | string | `'frame_major'` | `'frame_major'`: `[f0_bin0, f0_bin1..., f1_bin0, f1_bin1...]`<br>`'feature_major'`: `[f0_bin0, f1_bin0..., f0_bin1, f1_bin1...]` |

---

## Training Configuration

### Model Architectures

| Model | Description | Input Size | Parameters | Best For |
|---|---|---|---|---|
| `RF_MLP_4L_19K` | 4-layer MLP, ~19K params | 130 | ~19K | **Default** — simple, fast inference on range-bin vectors |


### Key Training Parameters (from `xwrl6432_surface_classification.yaml`)

```yaml
training:
  model_name: RF_MLP_4L_19K
  dataset_loader: GenericRadarDataset
  training_backend: tinyml_tinyverse

  quantization: 1                  # 0=float32, 1=generic quantization
  quantization_method: "QAT"
  quantization_weight_bitwidth: 8
  quantization_activation_bitwidth: 8
  auto_quantization: false

  training_epochs: 100
  early_stopping: false
  early_stopping_patience: 5

  batch_size: 64
  num_classes: 2
  learning_rate: 0.0001
  optimizer: sgd

  num_gpus: 0
  training_device: cpu

compilation:
  compile_preset_name: "default_preset"
```

> **Note:** `num_classes: 2` in the example config — update to match the number of surface/material classes in your dataset (one subdirectory per class under `dataset/classes/`).

---

## Dataset & Split Configuration

```yaml
dataset:
  dataset_name: radar_surface_classification
  input_data_path: /home/lte_phy/surface_classification/radar_surf_class_dataset.zip
  split_type: amongst_files
  split_factor: [0.7, 0.2, 0.1]
```

| Parameter | Value | Description |
|---|---|---|
| `split_type` | `amongst_files` | Splits by **whole files** — some recording files go entirely to train, others entirely to val/test. No window ever spans two files, and no file contributes to more than one split. Recommended for range classification where recordings are per-surface sessions. |
| `split_factor` | `[0.7, 0.2, 0.1]` | 70% train / 20% val / 10% test, applied at the file level |

> Contrast with `split_type: within_files` (used in point-cloud/track tasks like posture & fall detection), which splits each file internally by contiguous row ranges rather than splitting whole files.

---

## Quantization & Compilation Details

### Quantization Modes

| Mode | `quantization` | Output Type | Dequantization Needed? |
|---|---|---|---|
| Float32 (baseline) | 0 | float32 | No |
| Generic Quantized (QAT, default here) | 1 | int8 | **YES** |

### Generated Artifacts (Quantized Model)

```
compilation/artifacts/
├── tvmgen_default.h    # Header with I/O structs + dequant params
└── mod.a               # Precompiled static library
```

---

## Use-Case Scenarios

### 1. Standard Surface Classification (Default)
```yaml
data_processing_feature_extraction:
  feature_extraction_name: "Generic_130Input_Radar_RangeClassification"
```
**Result:** 130-input tensor (26 range bins × 5 frames), dense sliding window, no quality filtering.

### 2. Custom: Narrower Range-Bin Window
```yaml
data_processing_feature_extraction:
  feature_extraction_name: "Generic_130Input_Radar_RangeClassification"
  radar_range_bins_bounds: [8, 23]
```
**Result:** 16 bins instead of 26 — crops more aggressively if only near/mid-range bins carry surface signal.

### 3. Custom: Larger Temporal Context
```yaml
data_processing_feature_extraction:
  feature_extraction_name: "Generic_130Input_Radar_RangeClassification"
  frame_size: 10
  stride_size: 0.1
```
**Result:** Each window spans 10 frames (vs. 5), stride = 1 frame — more temporal context per window, more overlap between windows.

### 4. Custom: Sparse Windowing for Large Datasets
```yaml
data_processing_feature_extraction:
  feature_extraction_name: "Generic_130Input_Radar_RangeClassification"
  stride_size: 1.0
```
**Result:** Non-overlapping windows — fewer training samples, faster iteration.

---

## Input Data Requirements

CSV must contain **case-sensitive** `bin*` columns (per TI capture tool FFT range-profile format):

| Stage | Expected Columns |
|---|---|
| Range Data | `bin0`, `bin1`, `bin2`, ... `binN` (one column per FFT range bin, in order) |

Each CSV file should correspond to one recording session over a single surface/material class, placed under `dataset/classes/<class_name>/`.

---

## Output Tensor Shape

| Config | Range Bins | Frame Size | Tensor Shape (X) |
|---|---|---|---|
| `Generic_130Input_Radar_RangeClassification` (default) | 26 | 5 | `[N, 130]` |

**Labels (Y):** `[N]` integer class indices — one index per surface/material class, derived from the subdirectory name under `dataset/classes/`.

---

## Files in This Example

| File | Purpose |
|---|---|
| `xwrl6432_surface_classification.yaml` | Baseline training config (raw FFT range data, quantized) |
| `README.md` | This guide |

---

## Key Reference Files in Repository

| Component | Location |
|---|---|
| Preset definitions | `tinyml_modelmaker/ai_modules/radar/constants.py` → `FEATURE_EXTRACTION_PRESET_DESCRIPTIONS['Generic_130Input_Radar_RangeClassification']` |
| Dataset loader | `tinyml-tinyverse/tinyml_tinyverse/common/datasets/radar_dataset.py` → `GenericRadarDataset._process_range_bins` |
| TVM compilation | `tinyml-tinyverse/tinyml_tinyverse/common/compilation/` |
| Training entry point | `tinyml-tinyverse/tinyml_tinyverse/references/radar_classification/train.py` |

---

*Generated for Tensorlab EdgeAI — Texas Instruments*
*Compatible with TI mmWave radar sensors*
