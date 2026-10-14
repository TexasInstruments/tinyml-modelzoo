# Radar Posture & Fall Detection 

Classify a person's posture (standing, sitting, walking, falling, laying) from a TI mmWave radar sensor, using the point cloud + track outputs the radar's on-device signal chain already produces — no raw ADC support within Tensorlab.

This guide covers the data processing configuration, training, quantization, and compilation for the xWRL6432 target using TensorLab EdgeAI Framework.

- **Example config:** [config.yaml](./config.yaml)
- **Dataset:** [posture_and_fall_detection_dataset.zip](https://software-dl.ti.com/C2000/esd/mcu_ai/datasets/posture_and_fall_detection_dataset.zip)

---

## Overview

This example config demonstrates retraining through Tensorlab and final compiled artifacts generation for using TI mmWave sensors with machine learning for human posture and fall classification using the xWRL6432 sensor module. The application classifies five pose states:

- **Falling** (class 0)
- **Laying** (class 1)
- **Sitting** (class 2)
- **Standing** (class 3)
- **Walking** (class 4)

The model uses the radar's on-device signal chain outputs — point cloud detections and track data — as input. Refer Radar ToolBox for procedures to flash, run, and customize the IWRL6432 Pose and Fall Detection Demo. 

---

## Quick Start

### Two Installation Paths

| Path | Use When | What You Get |
|------|----------|--------------|
| **Clone** | You want examples/, wrapper scripts, or to modify models | Full repo + editable install |
| **Wheel Support** | You only run configs, no model changes | Minimal pip install, no examples folder |

---

### Linux / macOS — Install (Recommended)

```bash
# 1. Enable long paths (required for some model files)
git config --global core.longpaths true

# 2. Clone and enter repo
git clone https://github.com/TexasInstruments/tinyml-modelzoo.git
cd tinyml-modelzoo

# 3. Create virtual environment & install (pulls modelmaker, tinyverse, torchmodelopt, ti_mcu_nnc as wheels)
python3.10 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -e .

# 4. Verify installation
python -c "import tinyml_modelmaker, tinyml_tinyverse, tinyml_torchmodelopt, tinyml_modelzoo; print('All packages OK')"

# 5. Set compiler for xWRL6432 (required for compilation step)
#export TI_CGT_ARMLLVM_PATH=/path/to/ti-cgt-armllvm_5.1.1.LTS  # Adjust path

# 6. Run training (outputs to ./data/projects/)
./run_tinyml_modelzoo.sh examples/radar_posture_and_fall_detection/config.yaml
```

---

### Windows (PowerShell) — Install (Recommended)

```powershell
# 1. Enable long paths (run once as Administrator)
git config --global core.longpaths true

# 2. Clone and enter repo
git clone https://github.com/TexasInstruments/tinyml-modelzoo.git
cd tinyml-modelzoo

# 3. Create virtual environment & install
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -e .

# 4. Verify installation
python -c "import tinyml_modelmaker, tinyml_tinyverse, tinyml_torchmodelopt, tinyml_modelzoo; print('All packages OK')"

# 5. Set compiler for xWRL6432 (required for compilation step)
#$env:TI_CGT_ARMLLVM_PATH = "C:\path\to\ti-cgt-armllvm_5.1.1.LTS"  # Adjust path

# 6. Run training (outputs to .\data\projects\)
# If script blocked: Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
./run_tinyml_modelzoo.ps1 examples/radar_posture_and_fall_detection/config.yaml
```

---

### Quick Commands (After Install)

| Action | Linux/macOS | Windows (PowerShell) |
|--------|-------------|----------------------|
| **Activate venv** | source venv/bin/activate | .\venv\Scripts\Activate.ps1 |
| **Run training** | ./run_tinyml_modelzoo.sh examples/radar_posture_and_fall_detection/config.yaml | ./run_tinyml_modelzoo.ps1 examples/radar_posture_and_fall_detection/config.yaml |
| **Run with extra args** | ./run_tinyml_modelzoo.sh examples/radar_posture_and_fall_detection/config.yaml training.epochs=50 | ./run_tinyml_modelzoo.ps1 examples/radar_posture_and_fall_detection/config.yaml training.epochs=50 |

---

### Wheel Install (Minimal — No Examples/dIRECT config Yaml Access)

```bash
# Linux/macOS/Windows (same command)
python -m venv venv
source venv/bin/activate          # Windows: .\venv\Scripts\Activate.ps1
pip install http://software-dl.ti.com/C2000/esd/mcu_ai/wheel/tinyml_modelzoo-1.5.0-py3-none-any.whl

# Download config manually from GitHub examples folder/ Create a new yaml config
# Then run directly:
python -m tinyml_modelmaker.run_tinyml_modelmaker path/to/config.yaml
```

> **Wheel install does NOT include** examples/ folder or run_tinyml_modelzoo.sh/.ps1 scripts.

---

### Bring Your Own Data (BYOD) / Bring Your Own Model (BYOM)

#### **BYOD: Use Your Own Radar Data**

1. **Organize data by class** (required for `amongst_files` split):
   ```
   my_dataset/
   ├── classes/
   │   ├── falling/
   │   │   ├── fall_001.csv
   │   │   └── fall_002.csv
   │   ├── sitting/
   │   │   ├── sit_001.csv
   │   │   └── ...
   │   └── standing/
   │       └── ...
   ```

2. **Update config.yaml** — point to your data:
   ```yaml
   dataset:
     dataset_name: "my_custom_posture"
     input_data_path: ""  # Empty = no auto-download
     split_type: "amongst_files"  # or "within_files" for long recordings
     split_factor: [0.7, 0.2, 0.1]
   ```

3. **Ensure column names match** (see [Expected Column Names](#expected-column-names-case-sensitive)) or use inline config (Section 3.2) to map your columns In case you want to use the Complete Data Processing Pipeline integrated with Tensorlab.

4. **Run training**:
   ```bash
   ./run_tinyml_modelzoo.sh examples/radar_posture_and_fall_detection/config.yaml
   ```

#### **BYOM: Add Your Own Model**

1. **Edit** `tinyml_modelzoo/tinyml_modelzoo/models/radar.py` and add your model class:
   ```python
   class MyCustomRadarModel(nn.Module):
       def __init__(self, config, input_features=248, variables=1, num_classes=5):
           super().__init__()
           self.config = config or {}
           self.input_features = self.config.get('input_features', input_features)
           self.variables = self.config.get('variables', variables)
           self.num_classes = self.config.get('num_classes', num_classes)
           
           input_size = self.input_features * self.variables
           
           # Your architecture here
           self.layers = nn.Sequential(
               nn.BatchNorm1d(input_size),
               nn.Linear(input_size, 128),
               nn.ReLU(),
               nn.BatchNorm1d(128),
               nn.Linear(128, num_classes)
           )
       
       def forward(self, x):
           x = x.view(x.size(0), -1)
           return self.layers(x)
   
   # Add to __all__ list at bottom of file:
   __all__ = [
       'RF_MLP_3L_19K', 'RF_MLP_4L_19K', 'RF_DualStream_PointCloud_19K',
       'MyCustomRadarModel'  # <-- Add here
   ]
   ```

2. **Register model metadata** in `tinyml_modelzoo/tinyml_modelzoo/model_descriptions/classification.py`:
   - Add a new entry in the `_model_descriptions` dictionary (see lines 1523-1576 for radar examples)
   - Add the model key to `enabled_models_list` (line 1606)
   
   ```python
   # In _model_descriptions dict (around line 1576):
   'MyCustomRadarModel': deep_update_dict(deepcopy(template_model_description), {
       'common': dict(
           generic_model=False,
           task_type=constants.TASK_TYPE_RADAR_POINT_CLOUD_CLASSIFICATION,  # or RADAR_SURFACE_CLASSIFICATION
           task_category=constants.TASK_CATEGORY_RADAR_CLASSIFICATION,
           model_details='Custom radar model for posture detection'
       ),
       'training': dict(
           model_training_id='MyCustomRadarModel',  # Must match class name in radar.py!
           model_name='MyCustomRadarModel',
           quantization=TinyMLQuantizationVersion.QUANTIZATION_GENERIC,
           properties=[dict(type="group", dynamic=True, script="radarclassification.py",
                          name="preprocessing_group", label="Preprocessing Parameters", default=[])] + template_gui_model_properties,
           target_devices=[
               constants.TARGET_DEVICE_xWRL6432,
           ],
       ),
   }),
   
   # In enabled_models_list (line 1606), add:
   enabled_models_list = [
       ...
       'RF_MLP_3L_11K', 'RF_MLP_4L_19K', 'RF_DualStream_PointCloud_19K',
       'MyCustomRadarModel',  # <-- Add here
   ]
   ```

3. **Use in config**:
   ```yaml
   training:
     model_name: "MyCustomRadarModel"
   ```

4. **Reinstall** (developer install only):
   ```bash
   pip install -e .
   ```

> **Developer install** (pip install -e .) picks up model changes instantly — no rebuild needed.

> **Note**: Model descriptions are Python dicts in classification.py. See existing radar entries (RF_MLP_3L_11K, RF_MLP_4L_19K, RF_DualStream_PointCloud_19K) as templates.

---

### Prerequisites Checklist

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.10.x | **Required** for ti_mcu_nnc compiler wheel |
| Git | Latest | With core.longpaths=true |
| Virtualenv | Built-in | python -m venv |
| TI ARM Compiler | 5.1.1.LTS | For compilation only; set TI_CGT_ARMLLVM_PATH |
| CUDA (optional) | 11.8+ | For GPU training; CPU works fine |

---

### Output Structure

After training completes, find artifacts at:

```
./data/projects/radar_point_cloud_classification/run/<RUN_ID>/
├── training/
│   ├── base/           # Float model (ONNX, checkpoint.pth)
│   └── quantization/   # Quantized model (ONNX, checkpoint.pth)
└── compilation/
    └── artifacts/      # C artifacts: tvmgen_default.h, mod.a
```

---

### Data Collection Procedure (for Custom Data)

1. Configure mmWave Data Recorder with appropriate `.cfg` file
2. Use hotkey-triggered recording to capture only desired motion segments:
   - Press hotkey → start recording
   - Perform pose/motion
   - Press hotkey → stop recording
3. Save each class to separate subdirectory under `dataset/classes/<class_name>/`

---

## YAML Configuration for Posture & Fall Detection

### Identify Your Data Source

| Case | Description | Action |
|------|-------------|--------|
| **A** | Data from TI Data Recorder / Visualizer (standard columns: point_x*, track_id, track_velocity_*) | Use preset directly |
| **B** | Data from other tools / custom firmware (different column names) | Rename columns first → see [Column Mapping](#column-mapping-for-non-ti-data) |
| **C** | Pre-cleaned / externally featurized data (no point_*, track_*, bin* columns) | Use Generic_Radar_SimpleWindowOnly preset |

---

### 3.1 Recommended: Use the Preset (Simplest)

```yaml
data_processing_feature_extraction:
  feature_extraction_name: "Generic_248Input_Posture_and_Fall_Detection"
```

**What this preset configures:**

| Pipeline Stage | Parameter | Value | Purpose |
|----------------|-----------|-------|---------|
| **Point Cloud Data** | `radar_point_cloud_data_selector` | `['y', 'z', 'snr']` | Keep: distance (y), height (z), SNR |
| | `radar_point_cloud_coordinate_system` | `'relative'` | Make coords relative to track |
| | `radar_point_cloud_relative_coordinate_filter` | `['y']` | Only make **distance (y)** relative to track centroid |
| | `radar_point_cloud_spatial_filter_bounds` | `[None, None, -10, 10, -10, 10]` | Filter z (height): ±10m; x,y unbounded |
| | `radar_point_cloud_non_spatial_filter_bounds` | `[None, None, None, None]` | No Doppler/SNR filtering |
| | `radar_max_points_per_frame` | `8` | Keep top 8 points per frame |
| | `radar_point_cloud_structure_strategy` | `'highest and lowest'` | Select highest & lowest points by sort key |
| | `radar_point_cloud_sort_by` | `'z'` | Sort by **height (z)** before selection |
| **Track Data** | `radar_track_data_selector` | `['z', 'velocity', 'acceleration']` | Keep: height, 3D velocity, 3D acceleration |
| | `radar_track_velocity_bounds` | `[-10, 10]` | Filter velocity: ±10 m/s |
| | `radar_track_acceleration_bounds` | `[-10, 10]` | Filter acceleration: ±10 m/s² |
| | `radar_track_spatial_filter_bounds` | `[None, None, None, None, -10, 10]` | Filter track z (height): ±10m |
| **Windowing** | `frame_size` | `8` | 8 consecutive radar frames = 1 sample |
| | `stride_size` | `0.125` | 1-frame overlap (8 × 0.125 = 1) |
| | `radar_min_points_per_window` | `12` | Min 12 real points across 8 frames |
| | `radar_frames_quality_index` | `'per_window'` | Quality check at window level |
| | `radar_feature_ordering` | `'frame_major'` | Flatten: frame0_feat0, frame0_feat1... |

**Output Calculation:**
```
Point Cloud: 3 features × 8 points × 8 frames = 192 features
Track:       7 features (z + vel_x,y,z + acc_x,y,z) × 8 frames = 56 features
Total:       192 + 56 = 248 features → matches RF_DualStream_PointCloud_19K
```

> **Important: Sensor Mounting Configuration**  
> The preset uses `y` (distance) and `z` (height).  
> depending on sensor mounting configurations, you MUST override:
> ```yaml
> radar_point_cloud_data_selector: ['x', 'z', 'snr']  # Use x instead of y
> radar_point_cloud_relative_coordinate_filter: ['x']  # Make x relative
> radar_point_cloud_sort_by: 'z'  # Still sort by height
> ```

---

### 3.2 Full Inline Configuration (Maximum Control)

#### Option 1: Override Preset Parameters

```yaml
data_processing_feature_extraction:
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
```

| Override | When to Use |
|----------|-------------|
| `frame_size: 16` | Longer falls, more temporal context needed |
| `stride_size: 0.5` | Faster training (2-frame stride), less data augmentation |
| `radar_min_points_per_window: 20` | Noisy environment, need cleaner windows |
| `radar_max_points_per_frame: 12` | Dense scenes, more points available |
| `data_selector: ['x','y','z','snr']` | Wall-mounted radar, need azimuth info |
| `sort_by: 'y'` | Sort by distance instead of height |
| `per_radar_frame` quality | Drop individual bad frames, not whole windows |
| `radar_frames_quality_index: None` | Disable quality entirely (pre-cleaned data) |

---

#### Option 2: Fully Inline (No Preset)

```yaml
data_processing_feature_extraction:
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
```

**Parameter-by-Parameter Explanation**

| Parameter | Type | Format | Description |
|-----------|------|--------|-------------|
| `data_proc_transforms` | list | `["Radar Point Cloud Data", "Radar Track Data", "Simple Window"]` | **Required** — enables pipeline stages |
| `frame_size` | int | `8` | Frames per window (must match model input) |
| `stride_size` | float | `0.125` | Stride as fraction of frame_size (0.125 = 1 frame) |
| `radar_point_cloud_data_selector` | list | `['y','z','snr']` | Which point features to keep (x,y,z,doppler,snr) |
| `radar_point_cloud_coordinate_system` | str | `'absolute' \| 'relative'` | Relative = subtract track position |
| `radar_point_cloud_relative_coordinate_filter` | list | `['y']` | Which axes to make relative (x,y,z) |
| `radar_point_cloud_spatial_filter_bounds` | list | `[min_x,max_x, min_y,max_y, min_z,max_z]` | Spatial bounds (None = no limit) |
| `radar_point_cloud_non_spatial_filter_bounds` | list | `[min_dop,max_dop, min_snr,max_snr]` | Doppler/SNR bounds |
| `radar_max_points_per_frame` | int | `8` | Max points kept per frame (zero-padded) |
| `radar_point_cloud_structure_strategy` | str | `'highest only' \| 'lowest only' \| 'highest and lowest'` | Point selection strategy |
| `radar_point_cloud_sort_by` | str | `'x' \| 'y' \| 'z' \| 'doppler' \| 'snr'` | Feature to sort points by |
| `radar_track_data_selector` | list | `['z','velocity','acceleration']` | Track feature groups (z, velocity, acceleration) |
| `radar_track_velocity_bounds` | list | `[min, max]` | Velocity bounds for all 3 axes |
| `radar_track_acceleration_bounds` | list | `[min, max]` | Acceleration bounds for all 3 axes |
| `radar_track_spatial_filter_bounds` | list | `[min_x,max_x, min_y,max_y, min_z,max_z]` | Track position bounds |
| `radar_min_points_per_window` | int | `12` | Min real points across window |
| `radar_frames_quality_index` | str | `'per_window' \| 'per_radar_frame' \| None` | Quality check level |
| `radar_feature_ordering` | str | `'frame_major' \| 'feature_major'` | Flattening order for ONNX |

**Bounds Format (All Bounds Use Same Pattern)**

```yaml
# Spatial: [min_x, max_x, min_y, max_y, min_z, max_z]
radar_point_cloud_spatial_filter_bounds: [None, None, -10.0, 10.0, -10.0, 10.0]

# Non-spatial: [min_doppler, max_doppler, min_snr, max_snr]
radar_point_cloud_non_spatial_filter_bounds: [None, None, 5.0, None]

# Track spatial: [min_x, max_x, min_y, max_y, min_z, max_z]
radar_track_spatial_filter_bounds: [None, None, None, None, -10.0, 10.0]

# Track velocity/acceleration: [min, max] (applied to all 3 axes)
radar_track_velocity_bounds: [-10.0, 10.0]
```

> **Tip**: Use `None` (or `null` in YAML) for "no bound".

---

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

**When to Use This Preset**

| Scenario | Example |
|----------|---------|
| **Pre-featurized data** | You already extracted 31 features/frame externally |
| **No point cloud/track columns** | CSV has generic `feat_0`, `feat_1`... columns |
| **Already windowed data** | Each row = one window (set `frame_size: 1`) |

**How It Works**
- **Assumes** input CSV rows = individual frames
- **Applies only** windowing: stacks `frame_size` rows → 1 sample
- **Input features** = `num_columns_in_csv` × `frame_size`

**Required Input Format**
```csv
# Each row = 1 radar frame, columns = your features
feat_0, feat_1, feat_2, ..., feat_30
0.1, 0.2, -0.5, ..., 1.3
0.2, 0.3, -0.4, ..., 1.2
...
```

**Model Compatibility**

| Model | Required Input |
|-------|----------------|
| `RF_DualStream_PointCloud_19K` | Depends on Frame Size and Number of input features in a radar frame | ✅ If `frame_size=8`, 31 cols |
| `RF_MLP_4L_19K` | Configurable | ✅ If input features match |
| `RF_MLP_3L_19K` | Configurable | ✅ If `frame_size × cols = 130` |

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
| `within_files` | Single long recording |
| `amongst_files` | Each recording session = one posture class (Recommended with at least good amount of dataset) | 

---

### Simple Window (`"Simple Window"`) Details

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
| `RF_MLP_3L_19K` | 3-layer MLP | ~11K | BN → FC(64) → ReLU → BN → FC(32) → ReLU → BN → FC(num_classes). Smaller, faster. |

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

Firmware links against `mod.a` and calls `tvmgen_default_run()` with the 248-value feature vector. The same preprocessing must be replicated in firmware code before passing to the model.

---

## Key Reference Files

| Component | Location |
|-----------|----------|
| Model architectures | `tinyml-modelzoo/tinyml_modelzoo/models/radar.py` |
| Radar Data Processing Pipeline (full reference) | [Radar Data Processing Pipeline Reference Guide](tinyml-modelzoo/docs/RADAR_PROCESSING_PIPELINE.md) |

---

## Need More Help?

- Search for issues or post questions on the [TI E2E Forum](https://e2e.ti.com/)
- See the full pipeline reference: [Radar_DataProcessing_Pipeline.md](tinyml-modelzoo/docs/RADAR_PROCESSING_PIPELINE.md)
- TensorLab EdgeAI documentation in the main repository - For More Clarity on Setup Environment

---

*Generated for Tensorlab EdgeAI — Texas Instruments*  
*Compatible with TI mmWave radar sensors*