# Radar Surface Classification

Classify the surface or material underneath/in front of a TI mmWave radar sensor — e.g. floor type, terrain, or ground material — directly from raw **FFT range-profile data**, with no point cloud or tracker involved. The shape of the energy return across range bins *is* the feature, because different materials reflect radar energy differently at different distances.

This guide covers the data processing configuration, training, quantization, and compilation for the xWRL6432 target using TensorLab EdgeAI.

- **Example config:** [config.yaml](./config.yaml)
- **Dataset:** [radar_grass_surface_dataset.zip](https://software-dl.ti.com/C2000/esd/mcu_ai/datasets/radar_grass_surface_dataset.zip) (Grass vs Not Grass)

---

## Overview

This example demonstrates using TI mmWave sensors with machine learning for surface/material classification using the xWRL6432 sensor. The application classifies surface types such as:

- **Grass vs Not Grass** 
- **Wet vs Dry** 

The model uses the radar's on-device signal chain output — the FFT range profile (`bin0`...`binN` columns) — as input. No point cloud or track data is required.

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
./run_tinyml_modelzoo.sh examples/radar_surface_classification/config.yaml
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
./run_tinyml_modelzoo.ps1 examples/radar_surface_classification/config.yaml
```

---

### Quick Commands (After Install)

| Action | Linux/macOS | Windows (PowerShell) |
|--------|-------------|----------------------|
| **Activate venv** | source venv/bin/activate | .\venv\Scripts\Activate.ps1 |
| **Run training** | ./run_tinyml_modelzoo.sh examples/radar_surface_classification/config.yaml | ./run_tinyml_modelzoo.ps1 examples/radar_surface_classification/config.yaml |
| **Run with extra args** | ./run_tinyml_modelzoo.sh examples/radar_surface_classification/config.yaml training.epochs=50 | ./run_tinyml_modelzoo.ps1 examples/radar_surface_classification/config.yaml training.epochs=50 |

---

### Wheel Install (Minimal — No Examples/  sample config YAML access)

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

1. **Organize data by class** (required for amongst_files split):
   ```
   my_dataset/
   ├── classes/
   │   ├── grass/
   │   │   ├── grass_001.csv
   │   │   └── grass_002.csv
   │   └── not_grass/
   │       ├── not_grass_001.csv
   │       └── ...
   ```

2. **Update config.yaml** — point to your data:
   ```yaml
   dataset:
     dataset_name: "my_custom_surface"
     input_data_path: ""  # Empty = no auto-download
     split_type: "amongst_files"
     split_factor: [0.7, 0.2, 0.1]
   ```

3. **Ensure column names match** (bin0, bin1, bin2... for range data) or use inline config (Section 3.2) to map your columns.

4. **Run training**:
   ```bash
   ./run_tinyml_modelzoo.sh examples/radar_surface_classification/config.yaml
   ```

#### **BYOM: Add Your Own Model**

1. **Edit** tinyml_modelzoo/tinyml_modelzoo/models/radar.py and add your model class:
   ```python
   class MyCustomRadarModel(nn.Module):
       def __init__(self, config, input_features=130, variables=1, num_classes=2):
           super().__init__()
           self.config = config or {}
           self.input_features = self.config.get('input_features', input_features)
           self.variables = self.config.get('variables', variables)
           self.num_classes = self.config.get('num_classes', num_classes)
           
           input_size = self.input_features * self.variables
           
           # Your architecture here
           self.layers = nn.Sequential(
               nn.BatchNorm1d(input_size),
               nn.Linear(input_size, 64),
               nn.ReLU(),
               nn.BatchNorm1d(64),
               nn.Linear(64, num_classes)
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

2. **Register model metadata** in tinyml_modelzoo/tinyml_modelzoo/model_descriptions/classification.py:
   - Add a new entry in the _model_descriptions dictionary (see lines 1523-1576 for radar examples)
   - Add the model key to enabled_models_list (line 1606)
   
   ```python
   # In _model_descriptions dict (around line 1576):
   'MyCustomRadarModel': deep_update_dict(deepcopy(template_model_description), {
       'common': dict(
           generic_model=False,
           task_type=constants.TASK_TYPE_RADAR_SURFACE_CLASSIFICATION,
           task_category=constants.TASK_CATEGORY_RADAR_CLASSIFICATION,
           model_details='Custom radar model for surface classification'
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
- **Output**: 26 bins/frame × 5 frames = **130 features** → matches `RF_MLP_4L_19K` and `RF_MLP_3L_19K`

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
  model_name: "RF_MLP_4L_19K"        # or "RF_MLP_3L_19K"
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
| **`RF_MLP_4L_19K`** (default) | 4-layer MLP | ~19K | `BatchNorm → FC(64) → ReLU → BN → FC(32) → ReLU → BN → FC(16) → ReLU → BN → FC(num_classes)`. Default input: 248 features. With 130-feature preset, first layer adapts automatically. |
| **`RF_MLP_3L_19K`** | 3-layer MLP | ~11K | `BatchNorm → FC(64) → ReLU → BN → FC(32) → ReLU → BN → FC(num_classes)`. Smaller footprint, faster inference. Default input: 130 features — **matches preset exactly**. |

Both models:
- Output **raw logits** (no Softmax/Sigmoid) — compatible with `CrossEntropyLoss`
- Work with the 130-feature preset (`Generic_130Input_Radar_SurfaceClassification`)
- Support quantization (QAT, 8-bit) and compilation for xWRL6432

### Model Selection Guide

| Scenario | Recommended Model |
|----------|-------------------|
| **Default / Best accuracy** | `RF_MLP_4L_19K` |
| **Smaller model / Faster inference** | `RF_MLP_3L_19K` |
| **Memory-constrained deployment** | `RF_MLP_3L_19K` (~40% fewer params) |

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

**Quantization config (default):**
- Method: QAT (Quantization Aware Training)
- Weight bitwidth: 8-bit
- Activation bitwidth: 8-bit
- Output int: true (for radar targets)

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

## Recent Fixes & Changes

### Bug Fix: `radar_range_bins_bounds` Handling
- **Issue**: Config with `radar_range_bins_bounds: [null, null]` caused `float(None)` error
- **Fixed in**: `tinyml-tinyverse/tinyml_tinyverse/common/utils/radar_utils.py` → `process_range_bins()`
- **Change**: Now uses `get_bounds_pairs()` to properly handle `None` bounds (no filtering)

### New Model Added: `RF_MLP_3L_19K`
- Added to `tinyml-modelzoo/tinyml_modelzoo/models/radar.py`
- 3-layer MLP (~11K params) for faster inference
- Default input features: 130 — matches preset exactly

---

## Key Reference Files

| Component | Location |
|-----------|----------|
| Model architectures | `tinyml-modelzoo/tinyml_modelzoo/models/radar.py` → `RF_MLP_3L_19K`, `RF_MLP_4L_19K` |
| Radar Data Processing Pipeline (full reference) | `tinyml-modelzoo/docs/RADAR_PROCESSING_PIPELINE.md` |
| Training script | `tinyml-tinyverse/tinyml_tinyverse/references/radar_classification/train.py` |
| Data processing utils | `tinyml-tinyverse/tinyml_tinyverse/common/utils/radar_utils.py` |

---

## Need More Help?

- Search for issues or post questions on the [TI E2E Forum](https://e2e.ti.com/)
- See the full pipeline reference: [RADAR_PROCESSING_PIPELINE.md](tinyml-modelzoo/docs/RADAR_PROCESSING_PIPELINE.md)
- TensorLab EdgeAI documentation in the main repository

---

*Generated for Tensorlab EdgeAI — Texas Instruments*  
*Compatible with TI mmWave radar sensors*