# Radar Surface Classification

Classify the surface or material underneath/in front of a TI mmWave radar sensor — e.g. floor type, terrain, or ground material — directly from raw **FFT range-profile data**, with no point cloud or tracker involved. Here, the shape of the energy return across range bins *is* the feature, because different materials reflect radar energy differently at different distances.

- **Example config:** [`config.yaml`](./config.yaml)
- **Dataset:** [radar_grass_surface_dataset.zip](https://software-dl.ti.com/C2000/esd/mcu_ai/datasets/radar_grass_surface_dataset.zip)
- **Full Pipeline Reference:** [`docs/radar/radar_data_processing_pipeline.md`](../../docs/radar/radar_data_processing_pipeline.md)

---

###  IMPORTANT: Column Name Requirements (CONSTRAINT)

The pipeline expects **SPECIFIC COLUMN NAMES** for raw CSV processing. Your CSV must contain:

| Data Type | Column Names |
|-----------|--------------|
| **Range Data** | `bin0`, `bin1`, `bin2`, ... `binN` (FFT range profile) |

See **major instructions** in the [full radar data processing pipeline reference](tinyml-modelzoo/docs/RADAR_PROCESSING_PIPELINE.md) to map your use-case scenario.

---


## 1. What the radar gives us

Surface classification works only on one processing stage: directly on the **FFT range profile**. After range-FFT, each capture frame is a 1D array of amplitude values, one per range bin (`bin0`, `bin1`, ..., `binN`), where bin index maps to distance from the sensor. 

It only needs **range-bin selection** (crop to the region of interest) and **windowing** (give the model some temporal context) — no point cloud, track, or scaling stages apply, since there's no point cloud or track data in the input at all (in the current version).

---

## 2. Data Processing & Feature Extraction

### Pipeline shape

```
Raw On-device Signal Chain Processed Output CSV (bin0, bin1, ..., binN columns = per-frame FFT range profile)
        │
        ▼
┌─────────────────────┐
│ Radar Range Data      │  Keep only bin columns within [min_bin, max_bin],
│ ("Stage 1")           │  cropping near-field clutter and far-field noise
└──────────┬───────────┘
           ▼
┌─────────────────────┐
│ Simple Window          │  Slide an N-frame window over time, optionally
│ ("Stage 5")           │  quality-filter, flatten to one feature vector
└──────────┬───────────┘
           ▼
   X: [N_windows, frame_size × num_range_bins]   (130 for the default preset)
   Y: [N_windows]  surface/material class index
```

**Why only these two stages?** Here, `data_proc_transforms = ["Radar Range Data", "Simple Window"]` — Point Cloud/Track/Scaling never activate because this task's CSVs don't even have `point_*`/`track_*` columns.

### Stage 1: Radar Range Data — FFT bin selection

Enabled by the `"Radar Range Data"` token.

- CSV must contain `bin0, bin1, ..., binN` — one column per FFT range bin, per the TI capture tool format. The stage sorts these numerically by suffix and casts to float.
- `radar_range_bins_bounds: [min_bin, max_bin]` (default `[6, 31]`, inclusive) keeps only the bins in that range, discarding the rest:
  - **Low bins (0–5) are dropped** — these correspond to very close range
  - **High bins (≥32) are dropped** — beyond the sensor's useful range for this setup, mostly noise.
  - What's left (26 bins) is the window of distances where the actual surface reflection lives.
- `radar_range_bins_bounds: null` keeps every bin column 

### Stage 5: Simple Window — sliding window 

Enabled by the `"Simple Window"` token.

- **`frame_size`** (default `5`): number of consecutive range-profile frames bundled per training sample.
- **`stride_size`** (default `0.2`, fraction of `frame_size`): stride `= int(frame_size × stride_size)` = 1 frame — a dense, heavily-overlapping sliding window, which maximizes training samples extracted from each recording session.
- **Feature ordering** (`radar_feature_ordering: 'frame_major'`, default): flattens as `[frame0_bin0, frame0_bin1, ..., frame1_bin0, ...]`. Switch to `'feature_major'` only if a custom model expects each bin's time series grouped contiguously instead.

### Preset

| Preset | Stages | Range Bins | Frame Size | Total Input |
|---|---|---|---|---|
| `Generic_130Input_Radar_SurfaceClassification` **(default)** | Range Data + Window | 26 (bins 6–31) | 5 | **130** (26 × 5) |

Any inline parameter you set in `config.yaml` overrides that key from the preset; everything else falls back to preset defaults.

### Output tensor

- **X:** `[N_windows, 130]` = 5 frames × 26 range bins (`frame_major` order)
- **Y:** `[N_windows]` integer class index, one per surface/material subdirectory under `dataset/classes/<class_name>/`

### Dataset split

```yaml
dataset:
  split_type: amongst_files
  split_factor: [0.7, 0.2, 0.1]
```

`amongst_files` splits by **whole recording file** — some files go entirely to train, others entirely to val/test, so no window ever spans two files and no single file contributes to more than one split.

---

## 3. Training

### Model architecture

| Model | Shape | Params | Why |
|---|---|---|---|
| **`RF_MLP_4L_19K`** (default) | `BatchNorm → FC(64) → ReLU → BatchNorm → FC(32) → ReLU → BatchNorm → FC(16) → ReLU → BatchNorm → FC(num_classes)` | ~19K | A plain 4-layer MLP is appropriate here because the 130-value input (5 stacked 26-bin range profiles) is already a fairly low-dimensional, non-spatial feature vector.|

### Config (from `config.yaml`)

```yaml
training:
  model_name: RF_MLP_4L_19K
  training_epochs: 100
  learning_rate: 0.0001
  optimizer: sgd
  output_int: false
  num_gpus: 0
```
---
## 4. Quantization

Already by - default does Generic Quantization if not disabled in config. To disable, add to `config.yaml`:

```yaml
training:
  quantization: 0                      
```
---

## 5. Compilation

ONNX model is compiled via **TVM + TI's MCU NNC backend** targeting the xWRL6432's Cortex-M4:


**Generated artifacts** (`compilation/artifacts/`):

```
compilation/artifacts/
├── tvmgen_default.h   # I/O structs, tvmgen_default_run() entry point;
│                      # for a quantized model with output_int=true, also dequant scale/zero-point,
│                      # e.g.:  Tensor[(1, 130), float32] -> Tensor[(1, 2), uint8]
└── mod.a              # Precompiled static library archive
```

Firmware links `mod.a` and calls `tvmgen_default_run()`, feeding it the 130-value flattened range-bin feature vector built by replicating this same range-bin-selection + windowing logic on-device from the live FFT output.

---

## Files in This Example

| File | Purpose |
|---|---|
| `config.yaml` | End-to-end config: dataset, preprocessing preset, training (float by default) |
| `README.md` | This guide |

## Key Reference Files 

| Component | Location |
|---|---|
| Model architecture | `tinyml-modelzoo/tinyml_modelzoo/models/radar.py` → `RF_MLP_4L_19K` |
| Radar Data Processing Pipeline | `tinyml-modelzoo/docs/RADAR_PROCESSING_PIPELINE.md` |

---

*Generated for Tensorlab EdgeAI — Texas Instruments*
*Compatible with TI mmWave radar sensors (xWRL6432)*
