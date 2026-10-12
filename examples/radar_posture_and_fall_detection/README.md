# Radar Posture & Fall Detection

Classify a person's posture (standing, sitting, walking, falling, laying) from a TI mmWave radar sensor, using the point cloud + track outputs the radar's ondevice signal chain already produces — no raw ADC suopport.

This guide walks through **why** the data processing pipeline is shaped the way it is, then covers training, quantization, and compilation for the `xWRL6432` target.

- **Example config:** [`config.yaml`](./config.yaml)
- **Dataset:** [posture_and_fall_detection_dataset.zip](https://software-dl.ti.com/C2000/esd/mcu_ai/datasets/posture_and_fall_detection_dataset.zip)

See **major instructions** in the [full  radar data processing pipeline reference](tinyml-modelzoo/docs/RADAR_PROCESSING_PIPELINE.md) to map your use-case scenario.

---

## 1. What the radar gives us

Scenario : xWRL6432's Processed Output Data from device

| Stream | Columns | What it is |
|---|---|---|
| **Point cloud** | `point_x{i}`, `point_y{i}`, `point_z{i}`, `point_doppler{i}`, `point_snr{i}` | Up to several detected reflection points per frame — body-part-level detections (arms, torso, head), each with 3D position, radial velocity (Doppler), and signal quality (SNR) |
| **Track** | `track_id`/`tid`, `track_x/y/z`, `track_velocity_x/y/z`, `track_acc_x/y/z` | One tracked object (the person) per frame 

Posture and fall detection is fundamentally about **how the point cloud is distributed and moving around the tracked centroid over time**.

---

## 2. Data Processing & Feature Extraction

### Pipeline shape

```
Raw CSV (point_*, track_* columns) - Processed Data (On-Device Signal Chain)
        │
        ▼
┌─────────────────────┐
│ Track Data           │  Validate track (drop ghost tracks), select &
│ (stage: internal 3rd,│  bound track features
│  logically "Stage 3")│
└──────────┬───────────┘
           │ feeds relative-coordinate 
           ▼
┌─────────────────────┐
│ Point Cloud Data      │  Select point features, convert to coordinates
│ ("Stage 2")           │  relative to the track centroid, spatially filter,
│                       │  pick up to N points/frame
└──────────┬───────────┘
           ▼
┌─────────────────────┐
│ Scaling               │  Multiply track velocity by a fixed factor so its
│ ("Stage 4")           │  magnitude is comparable to position/point features 
(OPTIONAL)
└──────────┬───────────┘
           ▼
┌─────────────────────┐
│ Simple Window          │  Slide an N-frame window over time, drop
│ + Quality Index        │  low-quality windows, flatten to one feature vector
│ ("Stage 5")           │
└──────────┬───────────┘
           ▼
   X: [N_windows, frame_size × features_per_frame]   (392 for the default preset)
   Y: [N_windows]  posture class index
```

> **Important:** stage *execution order is fixed in code*, not by the order you list tokens in `data_proc_transforms`. Listing transforms in any order has the same effect.

### Track Data — "where is the person, and is this track real?"

Enabled by the `"Radar Track Data"` token.

- **Ghost-track filtering (automatic):** the radar tracker occasionally emits spurious/ghost tracks BASED ON tid/track_id.
- **Feature selection** (`radar_track_data_selector`): choose from `x, y, z, velocity, acceleration` — the default keeps all five groups (9 scalar features: x, y, z + 3 velocity components + 3 acceleration components).
- **Physical bounds as outlier filters:** `radar_track_velocity_bounds` (±10 m/s default) and `radar_track_acceleration_bounds` (±10 m/s² default) reject frames where the tracker's estimate is physically implausible for a person — usually tracker glitches, not real motion. `radar_track_spatial_filter_bounds` similarly bounds the track's x/y/z position (e.g. keep only tracks within ±10 m of the sensor on each axis) to exclude detections too far outside the sensing volume to be the subject.

### Point Cloud Data — "what does the body look like around that center?"

Enabled by the `"Radar Point Cloud Data"` token. This is where most of the posture signal lives.

- **Feature selection** (`radar_point_cloud_data_selector`): pick from `x, y, z, doppler, snr` per point. SNR helps the model weight confident detections more.
- **Relative coordinates — the key transform:** `radar_point_cloud_coordinate_system: relative` subtracts the track's centroid position from each point's coordinates (per axis listed in `radar_point_cloud_relative_coordinate_filter`, default `['y']` in the preset but commonly `['x','y','z']`). This makes the model learn the **shape of the point cloud around the person**, not their absolute location in the room — so posture classification generalizes across where in the radar's field of view the person happens to be standing.
- **Spatial filtering** (`radar_point_cloud_spatial_filter_bounds`, `[min_x,max_x,min_y,max_y,min_z,max_z]`): clips points far from the sensing volume (default ±10 m on each axis).
- **Non-spatial filtering** (`radar_point_cloud_non_spatial_filter_bounds`): optional Doppler/SNR range gates to drop noisy or implausible points.
- **Per-frame point selection:** a frame can have a variable number of detected points, but the model needs a fixed-size input. `radar_max_points_per_frame` (default 8) caps how many points are kept per frame; `radar_point_cloud_structure_strategy` controls *which* points survive when there are more than the cap:
  - `'highest only'` / `'lowest only'` — keep the top/bottom-N by `radar_point_cloud_sort_by` (default `'z'`, i.e. height)
  - `'highest and lowest'` (default) — interleave high and low points, which for a standing/falling human captures both head-region and floor-region reflections rather than clustering around one extreme

### Scaling — matching feature magnitudes

Enabled by the `"Scaling"` token, independent of every other stage. `radar_data_scaling_track_velocity` (default `30`) multiplies all `track_velocity_*` columns by a constant. Track velocity in m/s is numerically small compared to position/point-cloud features in meters and millimeter-scale noise; scaling it up keeps all the input tensor's feature columns in a comparable numeric range so the network doesn't implicitly down-weight velocity just because its raw values are small — this matters more for `RF_MLP_4L_19K`, since BatchNorm in `RF_DualStream_PointCloud_19K` already normalizes each stream separately.

### Simple Window — turning a frame stream into training samples

Enabled by the `"Simple Window"` token — this is where temporal context enters the picture. A single radar frame is a snapshot; posture (and especially a *fall*, which is a short dynamic event) is a function of several consecutive frames.

- **`frame_size`** (default `8`): number of consecutive frames bundled into one window — one training sample.
- **`stride_size`** (fraction of `frame_size`, default `0.125`): the window hop, computed as `int(frame_size × stride_size)`. With `frame_size=8, stride_size=0.125`, the stride is 1 frame — windows overlap heavily (dense sliding window), which maximizes the number of training samples extracted from limited fall-event recordings (falls are rare and short, so dense overlap matters for getting enough positive examples).
- **Quality index** (`radar_frames_quality_index: 'per_window'` by default): sums the real (non-padded) point counts across all frames in a candidate window and drops it if the total is below `radar_min_points_per_window` (default `12`). This throws out windows where the tracker mostly lost the person — e.g., momentarily out of the radar's field of view — which would otherwise inject label noise. The alternative mode, `'per_radar_frame'`, instead drops individual sparse *frames* before windowing (threshold `radar_min_points_per_frame`); `null` disables quality filtering entirely.
- A window is also rejected if its underlying frame indices aren't contiguous (i.e., it would span a gap left by previously dropped ghost/invalid frames) — this prevents a window from silently stitching together two unrelated time segments.
- **Feature ordering** (`radar_feature_ordering: 'frame_major'` default): flattens `[frame0_feat0, frame0_feat1, ..., frame1_feat0, ...]`. The alternative `'feature_major'` groups by feature across frames instead (`[frame0_feat0, frame1_feat0, ..., frame0_feat1, ...]`) — relevant mainly if a custom architecture expects per-feature time series contiguously.

### Presets

| Preset | Stages | Features/Frame | Frame Size | Total Input | Notes |
|---|---|---|---|---|---|
| `Generic_392Input_PointCloudWithTrack_SimpleWindow` **(default)** | Point Cloud + Track + Window | 49 (40 point-cloud + 9 track) | 8 | **392** | Full pipeline, raw TI capture CSV |
| `Generic_Radar_SimpleWindowOnly` | Window only | input-dependent (31 assumed) | 8 | configurable | Skip radar-specific processing — use for pre-cleaned/externally-featurized data |

Set `feature_extraction_name` to a preset and override individual parameters inline as needed — any key you set explicitly takes precedence over the preset's default for that key.

### Output tensor

- **X:** `[N_windows, 392]` = 8 frames × 49 features/frame (`frame_major` order)
- **Y:** `[N_windows]` integer class index — in this example: `0=Standing, 1=Sitting, 2=Walking, 3=Falling, 4=Laying`

---

## 3. Training

### Model architectures

| Model | Shape | Params | Why |
|---|---|---|---|
| **`RF_DualStream_PointCloud_19K`** (default) | Dual-stream MLP | ~19K | Splits each frame's features into a **position stream** (point coordinates + extra point-cloud features) and a **velocity stream** (the 3 track velocity components), each with its own `BatchNorm → Linear → ReLU → BatchNorm` branch, concatenated and fused through two more FC layers. Separating streams lets the network learn position-shape patterns and motion-dynamics patterns independently before combining them — a natural fit for a task where "shape" (posture) and "velocity" (fall dynamics) carry distinct information. |
| `RF_MLP_4L_19K` | Plain 4-layer MLP | ~19K | `BatchNorm → FC(64) → ReLU → BatchNorm → FC(32) → ReLU → BatchNorm → FC(16) → ReLU → BatchNorm → FC(num_classes)`. |


### Config (from `config.yaml`)

```yaml
training:
  model_name: RF_DualStream_PointCloud_19K
  quantization: 1                      # 0=float only, 1=generic int8 quant
  quantization_method: "QAT"
  quantization_weight_bitwidth: 8
  quantization_activation_bitwidth: 8
  training_epochs: 250
  learning_rate: 0.001
  optimizer: adamw
  lr_scheduler: CosineAnnealingLR
  weight_decay: 0
  output_int: false
  num_gpus: 0
```

Training runs a standard supervised loop (cross-entropy loss) and checkpoints the best validation-accuracy epoch. On completion, the float model is exported to ONNX (`model.onnx`) — this becomes the input to quantization.

Dataset split: `split_type: within_files` with `split_factor: [0.7, 0.2, 0.1]` — each recording file is split internally by contiguous row ranges across train/val/test, rather than assigning whole files to a split. This fits posture data where a single long recording session may contain multiple posture segments. #Recommended - `amongst_files` split method with each recording session containing one particular posture, split_type and split_factor can vary based on the dataset collected.

---

## 4. Quantization

Why quantize: the compiled model needs to run on the xWRL6432's Cortex-M4 MCU core, int8 arithmetic is dramatically cheaper and smaller than float32.

This example uses **QAT (Quantization-Aware Training)**: `quantization: 1` (generic quantization), `quantization_method: "QAT"`, 8-bit weights and activations. Training actually happens in **two phases automatically**:

1. **Float phase:** full `training_epochs` (250) run in float32, producing a baseline `checkpoint.pth` / `model.onnx`.
2. **Quantization fine-tune phase:** This produces a second `model.onnx` under a separate `training/quantization/` output path.

QAT (vs. PTQ) is the right choice here because it tends to recover more accuracy on networks sensitive to quantization — worthwhile given the dual-stream model already has a fairly narrow 19K-parameter budget to work with. `output_int: false` means the exported quantized model still dequantizes its output back to a human-readable score internally — set `true` only if your firmware wants to handle raw int8 output + separately-provided scale/zero-point itself (Since radar sensors support `float` type).

---

## 5. Compilation

The quantized (or float, if `quantization: 0`) ONNX model is compiled with **TVM + TI's MCU NNC (Neural Network Compiler)** backend into C artifacts linkable directly into firmware for the xWRL6432 (Cortex-M4):



**Generated artifacts** (`compilation/artifacts/`):

```
compilation/artifacts/
├── tvmgen_default.h   # I/O structs (tvmgen_default_inputs/outputs), tvmgen_default_run() entry
│                      # point, and — for quantized models — dequant scale/zero-point constants considering output_int: true 
└── mod.a              # Precompiled static library archive containing all compiled operator code
```

Intermediate TVM C-codegen sources (`lib0.c`, `lib1.c`, ...) are generated then deleted by default (pass `keep_libc_files: true` if you want to inspect them). Firmware links against `mod.a` and calls `tvmgen_default_run()` with the input struct populated from your 392-value feature vector — the same preprocessing done in Python must be replicated (or already be done by the sensor's onboard tracker/CFAR chain) on-device before this call.

---

## Files in This Example

| File | Purpose |
|---|---|
| `config.yaml` | End-to-end config: dataset, preprocessing preset, training, quantization |
| `README.md` | This guide |

## Key Reference Files 

| Component | Location |
|---|---|
| Model architectures | `tinyml-modelzoo/tinyml_modelzoo/models/radar.py` |
| Radar Data Processing Pipeline | `tinyml-modelzoo/docs/RADAR_PROCESSING_PIPELINE.md` |

---

*Generated for Tensorlab EdgeAI — Texas Instruments*
*Compatible with TI mmWave radar sensors (xWRL6432)*
