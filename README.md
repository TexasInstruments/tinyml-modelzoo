# Tiny ML ModelZoo

Texas Instruments' central repository for AI models, examples, and configurations for microcontroller (MCU) applications. Clone this repo, install it, and run any example config against your target device — training, quantization, and compilation all happen automatically underneath.

Detailed User Guide: [TI Tiny ML Tensorlab User Guide](https://software-dl.ti.com/C2000/esd/mcu_ai/user_guide/index.html)

```
tinyml-modelzoo/
├── examples/               # Ready-to-run example configurations
├── tinyml_modelzoo/
│   ├── models/             # Neural network model definitions
│   ├── model_descriptions/ # Model metadata for GUI integration
│   └── device_info/        # Target device performance data
├── run_tinyml_modelzoo.sh  # Training wrapper (Linux)
├── run_tinyml_modelzoo.bat # Training wrapper (Windows)
└── ADDING_NEW_MODELS.md    # Guide for adding custom models
```

---

## Table of Contents

- [Quick Start](#quick-start)
- [Supported Target Devices](#supported-target-devices)
- [Supported Task Categories](#supported-task-categories)
- [Choosing an Example](#choosing-an-example)
- [Examples Reference](#examples-reference)
  - [Classification](#classification)
  - [Regression](#regression)
  - [Forecasting](#forecasting)
  - [Anomaly Detection](#anomaly-detection)
  - [Audio Classification](#audio-classification)
  - [Image Classification](#image-classification)
  - [Radar Point Cloud Classification](#radar-point-cloud-classification)
- [Available Models](#available-models)
- [Adding New Models](#adding-new-models)
- [About the Task Types](#about-the-task-types)
- [Additional Resources](#additional-resources)
- [License](#license)

---

## Quick Start

### Prerequisites

1. Python 3.10 environment
2. Clone **only** this repository, then install it:
   ```bash
   cd tinyml-modelzoo
   pip install -e .
   ```
   This pulls in the rest of the toolchain as prebuilt wheels automatically -
   no need to clone anything else.

### Running an Example

**Linux:**
```bash
cd tinyml-modelzoo
./run_tinyml_modelzoo.sh examples/generic_timeseries_classification/config.yaml
```

**Windows (cmd):**
```bat
cd tinyml-modelzoo
run_tinyml_modelzoo.bat examples\generic_timeseries_classification\config.yaml
```

**Windows (PowerShell):**
```powershell
cd tinyml-modelzoo
./run_tinyml_modelzoo.ps1 examples/generic_timeseries_classification/config.yaml
```

### What Happens When You Run an Example?

1. **Dataset Download** - the required dataset is downloaded if not already present
2. **Data Processing** - feature extraction and preprocessing are applied
3. **Model Training** - the neural network is trained on your data
4. **Quantization** - the model is optimized for MCU deployment
5. **Compilation** - TI's Neural Network Compiler generates device-ready code

Output artifacts are saved to `./data/projects/<project_name>/`, relative to
whichever directory you ran `run_tinyml_modelzoo.sh`/`.bat`/`.ps1` from. To use
a different location, add this to the config's `common` section:
```yaml
common:
    projects_path: './your/choice'  # or an absolute path
```

---

## Supported Target Devices

### C2000 DSP Family (Texas Instruments)

| Device | NPU | Description | Notes |
|--------|-----|-------------|-------|
| F28P55 | Yes | C2000 32-bit MCU | Recommended for complex models |
| F28P65 | No | C2000 32-bit MCU, 150 MHz | High performance |
| F29H85 | No | C2000 64-bit MCU with C29x core | High capacity |
| F29P58 | No | C2000 64-bit MCU with C29x core | High capacity |
| F29P32 | No | C2000 64-bit MCU with C29x core | High capacity |
| F2837 | No | C2000 32-bit dual-core MCU, 200 MHz (xD) | General purpose |
| F2837xS | No | C2000 32-bit single-core MCU, 200 MHz (xS) | General purpose |
| F2838x | No | C2000 32-bit dual-core MCU, 200 MHz C28x+Cortex-M4 | Heterogeneous dual-core |
| F28P551x | No | C2000 32-bit MCU, 150 MHz C28x+CLA, 1.1-MB Flash | High performance |
| F28003 | No | C2000 32-bit MCU, 100 MHz | Cost-optimized |
| F28004 | No | C2000 32-bit MCU, 100 MHz | Cost-optimized |
| F2807x | No | C2000 32-bit MCU, 120 MHz C28x+CLA, 512-KB Flash | General purpose |
| F28002x | No | C2000 32-bit MCU, 100 MHz C28x+CLA, 256-KB Flash | Cost-optimized |
| F280013 | No | C2000 32-bit MCU, 100 MHz | Entry-level |
| F280015 | No | C2000 32-bit MCU, 120 MHz | Entry-level |
| F28E12 | No | C2000 32-bit MCU, 160 MHz, 64 KB flash | Entry-level |

### MSPM0 Family (Arm Cortex-M0+)

| Device | NPU | Description | Notes |
|--------|-----|-------------|-------|
| MSPM0G3507 | No | 80 MHz Arm Cortex-M0+ | Ultra-low power, classification only |
| MSPM0G3519 | No | 80 MHz Arm Cortex-M0+ | Ultra-low power |
| MSPM0G5187 | Yes | 80 MHz Arm Cortex-M0+ | Ultra-low power, NPU-accelerated |

### MSPM33C Family (Arm Cortex-M33)

| Device | NPU | Description | Notes |
|--------|-----|-------------|-------|
| MSPM33C32 | No | 160 MHz Arm Cortex-M33, TrustZone | 1MB flash, 256kB SRAM |
| MSPM33C34 | No | 160 MHz Arm Cortex-M33 | High performance |

### AM13 Family (Arm Cortex-M33)

| Device | NPU | Description | Notes |
|--------|-----|-------------|-------|
| AM13E2 | Yes | Arm Cortex-M33 MCU | NPU-accelerated |

### AM26x Family (Arm Cortex-R5)

| Device | NPU | Description | Notes |
|--------|-----|-------------|-------|
| AM263 | No | Quad-core Arm Cortex-R5F, 400 MHz | High performance |
| AM263P | No | Quad-core Arm Cortex-R5F, 400 MHz | High performance |
| AM261 | No | Single-core Arm Cortex-R5F, 400 MHz | Cost-optimized |

### Connectivity Devices (Wireless)

| Device | Hardware Accelerator | Description | Notes |
|--------|-----|-------------|-------|
| CC2755 | CDE | 96 MHz Arm Cortex-M33 wireless MCU | Optimized for PIR/wireless apps |
| CC1312 | No | Arm Cortex-M4F wireless MCU | Sub-1GHz |
| CC1314 | No | Arm Cortex-M33 wireless MCU | Sub-1GHz |
| CC1352 | No | Arm Cortex-M4F wireless MCU | Sub-1GHz |
| CC1354 | No | Arm Cortex-M33 wireless MCU | Sub-1GHz |
| CC35X1 | CDE | Arm Cortex-M33 wireless MCU | Wi-Fi + BLE combo |

The example tables further down reference these devices via short, named
sets (Set A, Set B, ...) rather than repeating full device lists — see
[Examples Reference](#examples-reference).

---

## Supported Task Categories

| Task Category | Description | Use Cases |
|----------------|-------------|-----------|
| **Time Series Classification** | Categorize time-series data into discrete classes | Fault detection, activity recognition, anomaly classification |
| **Time Series Regression** | Predict continuous values from time-series inputs | Torque estimation, speed prediction, load measurement |
| **Time Series Forecasting** | Predict future values based on historical patterns | Temperature prediction, demand forecasting |
| **Time Series Anomaly Detection** | Identify abnormal patterns using autoencoder-based models | Equipment health monitoring, predictive maintenance |
| **Audio Classification** | Classify audio signals from MFCC features | Keyword spotting, voice commands, sound event detection |
| **Image Classification** | Categorize images into classes | Visual inspection, object recognition |
| **Radar Point Cloud Classification** | Classify point-cloud frames from radar sensors | Human pose detection, fall detection |

For the reasoning behind how these categories differ from one another, see
[About the Task Types](#about-the-task-types).

---

## Choosing an Example

1. **Look for your use case** in the [Examples Reference](#examples-reference) tables below. If one matches (e.g. `motor_bearing_fault`, `pir_detection`), start there — it ships with a dataset, a tuned model, and a device-specific config.
2. **If nothing matches**, use the generic example for your task type instead (the first row in each table below, e.g. [generic_timeseries_classification](examples/generic_timeseries_classification/)) and point it at your own dataset. This is also the recommended **first example to run** to learn the toolchain.

---

## Examples Reference

Each example links to its config directory under `examples/`. The **Config**
column marks whether it's a **Generic** example (a `generic_timeseries_*`
config, meant to be adapted to your own dataset) or a **Dedicated** example
(a purpose-built config/model for that specific use case).

**Device sets** (referenced by name below, instead of repeating device lists):

| Set | Devices |
|-----|---------|
| **A** | F280013, F280015, F28003, F28004, F2807x, F28002x, F2837, F2837xS, F2838x, F28P55, F28P65, F28P551x, F29H85, MSPM0G3507, MSPM0G3519, MSPM0G5187, MSPM33C32, AM13E2, AM263 |
| **B** | Set A + Connectivity devices (CC1312, CC1314, CC1352, CC1354, CC2755, CC2745, CC35X1) |
| **C** | MSPM0G5187 only |
| **D** | MSPM0G3507, MSPM0G3519, MSPM0G5187 |
| **E** | MSPM0G3507, MSPM0G3519, MSPM0G5187, MSPM33C32 |
| **F** | CC2755, CC1312, CC1314, CC1352, CC1354, CC35X1, MSPM0G5187, MSPM0G3507, MSPM0G3519, MSPM33C32 |
| **G** | All devices listed under [Supported Target Devices](#supported-target-devices) |
| **H** | C2000, MSPM0, MSPM33C, AM26x, and AM13 families (all except Connectivity devices) |

### Classification

| Example | Config | Data Type | Devices | Description |
|---------|--------|-----------|---------|--------------|
| [generic_timeseries_classification](examples/generic_timeseries_classification/) | Generic | — | Set G | Classify sine/square/sawtooth waveforms. **Start here** to learn the toolchain. |
| [dc_arc_fault](examples/dc_arc_fault/) | Dedicated | Univariate | Set A | Detect DC arc faults from current waveforms for electrical safety. |
| [ac_arc_fault](examples/ac_arc_fault/) | Dedicated | Univariate | Set A | Detect AC arc faults in electrical systems. |
| [motor_bearing_fault](examples/motor_bearing_fault/) | Dedicated | Multivariate | Set A | Classify 5 bearing fault types + normal operation from vibration data. |
| [blower_imbalance](examples/blower_imbalance/) | Dedicated | Multivariate | Set A | Detect blade imbalance in HVAC blowers using 3-phase motor currents. |
| [fan_blade_fault_classification](examples/fan_blade_fault_classification/) | Dedicated | Multivariate | Set B | Detect faults in BLDC fans from accelerometer data. |
| [gearbox_fault_detection](examples/gearbox_fault_detection/) | Dedicated | Multivariate | Set D | Classify gearbox operating conditions (healthy vs broken tooth) from vibration data. |
| [grid_fault_detection](examples/grid_fault_detection/) | Dedicated | Multivariate | Set A | Detect electrical grid faults from sensor data. |
| [ecg_classification](examples/ecg_classification/) | Dedicated | Multivariate | Set D | Classify normal vs anomalous heartbeats from ECG signals. |
| [pir_detection](examples/pir_detection/) | Dedicated | Multivariate | Set F | Detect presence/motion using PIR sensor data. |
| [fall_detection_classification](examples/fall_detection_classification/) | Dedicated | Multivariate | Set C | Detect and classify Human Fall vs Activities of Daily Living (ADL). |
| [dynamic_hand_gesture_recognition](examples/dynamic_hand_gesture_recognition/) | Dedicated | Multivariate | Set C | Classify 4 dynamic hand gestures (circle, wave, tap, other) from 3-axis accelerometer data. |
| [electrical_fault](examples/electrical_fault/) | Generic | Multivariate | Set H | Classify transmission line faults using voltage and current (2-class and 6-class variants). |
| [grid_stability](examples/grid_stability/) | Generic | Multivariate | Set H | Predict power grid stability from node parameters. |
| [gas_sensor](examples/gas_sensor/) | Generic | Multivariate | Set H | Identify gas type and concentration from sensor array data. |
| [branched_model_parameters](examples/branched_model_parameters/) | Generic | Multivariate | Set H | Human Activity Recognition from accelerometer/gyroscope data. |
| [nilm_appliance_usage_classification](examples/nilm_appliance_usage_classification/) | Generic | Multivariate | Set H | Non-Intrusive Load Monitoring - identify active appliances. |
| [PLAID_nilm_classification](examples/PLAID_nilm_classification/) | Generic | Multivariate | Set H | Appliance identification using the PLAID dataset. |

### Regression

| Example | Config | Data Type | Devices | Description |
|---------|--------|-----------|---------|--------------|
| [generic_timeseries_regression](examples/generic_timeseries_regression/) | Generic | — | Set H | Generic regression example for continuous value prediction. |
| [mosfet_temp_prediction](examples/mosfet_temp_prediction/) | Dedicated | Multivariate | Set A | Predict MOSFET temperature from electrical parameters. |
| [torque_measurement_regression](examples/torque_measurement_regression/) | Generic | Multivariate | Set H | Predict PMSM motor torque from current measurements. |
| [induction_motor_speed_prediction](examples/induction_motor_speed_prediction/) | Generic | Multivariate | Set H | Predict induction motor speed from electrical signals. |
| [reg_washing_machine](examples/reg_washing_machine/) | Generic | Multivariate | Set H | Predict washing machine load weight. |

### Forecasting

| Example | Config | Data Type | Devices | Description |
|---------|--------|-----------|---------|--------------|
| [generic_timeseries_forecasting](examples/generic_timeseries_forecasting/) | Generic | — | Set H | Generic forecasting example for time series prediction. |
| [forecasting_pmsm_rotor_temp](examples/forecasting_pmsm_rotor_temp/) | Generic | Multivariate | Set H | Forecast PMSM rotor winding temperature. |
| [hvac_indoor_temp_forecast](examples/hvac_indoor_temp_forecast/) | Generic | Multivariate | Set H | Predict indoor temperature for HVAC control. |

### Anomaly Detection

| Example | Config | Data Type | Devices | Description |
|---------|--------|-----------|---------|--------------|
| [generic_timeseries_anomalydetection](examples/generic_timeseries_anomalydetection/) | Generic | — | Set H | Generic anomaly detection example using autoencoders. |
| [dc_arc_fault (DSI)](examples/dc_arc_fault/config_anomaly_detection_dsi.yaml) | Dedicated | Univariate | Set A | Detect anomalous DC arc patterns using autoencoder (DSI dataset). |
| [dc_arc_fault (DSK)](examples/dc_arc_fault/config_anomaly_detection_dsk.yaml) | Dedicated | Univariate | Set A | Detect anomalous DC arc patterns using autoencoder (DSK dataset). |
| [ecg_classification](examples/ecg_classification/config_anomaly_detection.yaml) | Dedicated | Multivariate | Set D | Detect anomalous heartbeat patterns from ECG signals. |
| [fan_blade_fault_classification](examples/fan_blade_fault_classification/config_anomaly_detection.yaml) | Dedicated | Multivariate | Set B | Detect anomalous fan blade behavior from accelerometer data. |
| [motor_bearing_fault](examples/motor_bearing_fault/config_anomaly_detection.yaml) | Dedicated | Multivariate | Set A | Detect anomalous bearing behavior from vibration data. |

### Audio Classification

| Example | Config | Data Type | Devices | Description |
|---------|--------|-----------|---------|--------------|
| [google_speech_command](examples/google_speech_command/) | Dedicated | Audio | Set C | 12-class keyword spotting from audio using MFCC + DSCNN model. |

### Image Classification

| Example | Config | Data Type | Devices | Description |
|---------|--------|-----------|---------|--------------|
| [MNIST_image_classification](examples/MNIST_image_classification/) | Dedicated | Image | Set E | Handwritten digit recognition (MNIST dataset). |
| [machine_readable_code_classification](examples/machine_readable_code_classification/) | Dedicated | Image | Set C | Classify QR codes, barcodes, and other symbols (28×28 images). |
| [coffee_bean_classification](examples/coffee_bean_classification/) | Dedicated | Image | Set C | Classify coffee bean quality from images. |

### Radar Point Cloud Classification

| Example | Config | Data Type | Devices | Description |
|---------|--------|-----------|---------|--------------|
| [radar_point_cloud_classification](examples/radar_point_cloud_classification/) | Dedicated | Point Cloud | MSPM0G3507 | Human pose and fall detection from radar point-cloud frames. |

---

## Available Models

Models are organized by task type. The **NPU** column indicates hardware acceleration support on TI devices with NPU (F28P55, AM13E2, MSPM0G5187).

**NPU-optimized models** follow specific layer constraints for hardware acceleration:
- All channels are multiples of 4 (m4)
- Kernel heights ≤ 7 for GCONV layers
- MaxPool kernels ≤ 4
- FC layer inputs ≥ 16 features (8-bit) or ≥ 8 features (4-bit)

For detailed guidelines, see [NPU Configuration Guidelines](docs/NPU_CONFIGURATION_GUIDELINES.md).

**When to use NPU-optimized models:**
- Target device has NPU (F28P55, AM13E2, MSPM0G5187)
- You need maximum inference speed
- Standard models show "fallback to software" warnings during compilation

### Classification Models

| Model Name | Parameters | Architecture | NPU | Description |
|------------|------------|--------------|-----|-------------|
| `CLS_100_NPU` | ~100 | CNN | Yes | Ultra-compact model |
| `CLS_500_NPU` | ~500 | CNN | Yes | Compact model |
| `CLS_1k_NPU` | ~1K | CNN | Yes | Lightweight 2-layer CNN |
| `CLS_1.2k_NPU` | ~1.2K | CNN | Yes | Compact model for ultra-low power devices |
| `CLS_1.5k_NPU` | ~1.5K | CNN | Yes | 3-layer model with balanced performance |
| `CLS_1.9k_NPU` | ~1.9K | CNN | Yes | Efficient 3-layer model |
| `CLS_2k_NPU` | ~2K | CNN | Yes | 2-layer model |
| `CLS_2.8k_NPU` | ~2.8K | CNN | Yes | Improved accuracy with compact footprint |
| `CLS_3.1k_NPU` | ~3.1K | CNN | Yes | Higher accuracy model |
| `CLS_ResAdd_3k` | ~3K | ResNet (Add) | No | Residual connections with addition |
| `CLS_ResCat_3k` | ~3K | ResNet (Cat) | No | Residual connections with concatenation |
| `CLS_3.9k_NPU` | ~3.9K | CNN | Yes | Advanced 3-layer model |
| `CLS_4k_NPU` | ~4K | CNN | Yes | Balanced model |
| `CLS_4.2k_NPU` | ~4.2K | CNN | Yes | Optimized 4-layer model |
| `CLS_5k_NPU` | ~5K | CNN | Yes | Mid-range model |
| `CLS_6k_NPU` | ~6K | CNN (DW-Sep) | Yes | Depthwise separable |
| `CLS_8k_NPU` | ~8K | CNN (DW-Sep) | Yes | Depthwise separable |
| `CLS_13k_NPU` | ~13K | CNN | Yes | Higher capacity |
| `CLS_20k_NPU` | ~20K | CNN | Yes | High capacity |
| `CLS_40k_NPU` | ~40K | CNN | Yes | Advanced model for complex tasks |
| `CLS_55k_NPU` | ~55K | CNN | Yes | Maximum accuracy |
| `ArcFault_model_200_t` | ~200 | Specialized | No | Arc fault detection |
| `ArcFault_model_300_t` | ~300 | Specialized | No | Arc fault with more capacity |
| `ArcFault_model_700_t` | ~700 | Specialized | No | Arc fault medium model |
| `ArcFault_model_1400_t` | ~1.4K | Specialized | No | Arc fault high accuracy |
| `GearboxFault_model_1.2k_t` | ~1.2K | CNN | Yes | Gearbox fault detection |
| `GearboxFault_model_1.5k_t` | ~1.5K | CNN | Yes | Gearbox fault with more capacity |
| `MotorFault_model_1_t` | Varies | Specialized | No | Motor bearing fault detection |
| `MotorFault_model_2_t` | Varies | Specialized | No | Motor fault variant 2 |
| `MotorFault_model_3_t` | Varies | Specialized | No | Motor fault variant 3 |
| `FanImbalance_model_1_t` | Varies | Specialized | No | Fan blade imbalance detection |
| `FanImbalance_model_2_t` | Varies | Specialized | No | Fan imbalance variant 2 |
| `FanImbalance_model_3_t` | Varies | Specialized | No | Fan imbalance variant 3 |
| `PIRDetection_model_1_t` | Varies | Specialized | No | PIR-based presence detection |

### Regression Models

| Model Name | Parameters | Architecture | NPU | Description |
|------------|------------|--------------|-----|-------------|
| `REGR_500_NPU` | ~500 | CNN | Yes | Compact regression |
| `REGR_1k` | ~1K | CNN | No | Lightweight regression model |
| `REGR_2k_NPU` | ~2K | CNN | Yes | 2-layer model |
| `REGR_3k` | ~3K | MLP | No | 4-layer fully connected network |
| `REGR_4k` | ~4K | CNN | No | 2 Conv+BN+ReLU + Linear |
| `REGR_6k_NPU` | ~6K | CNN (DW-Sep) | Yes | Depthwise separable convolutions |
| `REGR_8k_NPU` | ~8K | CNN | Yes | 3-layer model |
| `REGR_10k` | ~10K | CNN | No | 3 Conv+BN+ReLU + 2 Linear |
| `REGR_13k` | ~13K | CNN | No | High capacity regression |
| `REGR_20k_NPU` | ~20K | CNN | Yes | High capacity with MaxPool |

### Anomaly Detection Models

Note: For NPU models, encoder convolutions are NPU-accelerated but decoder upsampling falls back to CPU.

| Model Name | Parameters | Architecture | NPU | Description |
|------------|------------|--------------|-----|-------------|
| `AD_500_NPU` | ~500 | CNN AE | Yes | 2-layer autoencoder |
| `AD_1k` | ~1K | Autoencoder | No | Compact autoencoder |
| `AD_2k_NPU` | ~2K | CNN AE | Yes | 2-layer autoencoder |
| `AD_4k` | ~4K | Autoencoder | No | 3-layer CNN autoencoder |
| `AD_6k_NPU` | ~6K | CNN AE (DW-Sep) | Yes | Depthwise separable encoder |
| `AD_8k_NPU` | ~8K | CNN AE | Yes | 3-layer autoencoder |
| `AD_10k_NPU` | ~10K | CNN AE | Yes | 3-layer autoencoder |
| `AD_16k` | ~16K | Autoencoder | No | 4-layer CNN autoencoder |
| `AD_17k` | ~17K | Autoencoder | No | Fan blade anomaly detection |
| `AD_20k_NPU` | ~20K | CNN AE | Yes | High capacity autoencoder |
| `AD_Linear` | Varies | Linear AE | No | 3-layer deep linear autoencoder |
| `Ondevice_Trainable_AD_Linear` | Varies | Linear AE | No | On-device trainable variant |

### Forecasting Models

Note: LSTM models are not NPU-supported.

| Model Name | Parameters | Architecture | NPU | Description |
|------------|------------|--------------|-----|-------------|
| `FCST_500_NPU` | ~500 | CNN | Yes | Compact forecasting |
| `FCST_1k_NPU` | ~1K | CNN | Yes | 2-layer model |
| `FCST_2k_NPU` | ~2K | CNN | Yes | 2-layer model |
| `FCST_3k` | ~3K | MLP | No | 4-layer fully connected |
| `FCST_4k_NPU` | ~4K | CNN | Yes | 3-layer model |
| `FCST_6k_NPU` | ~6K | CNN (DW-Sep) | Yes | Depthwise separable convolutions |
| `FCST_8k_NPU` | ~8K | CNN | Yes | 3-layer model |
| `FCST_10k_NPU` | ~10K | CNN | Yes | 3-layer model |
| `FCST_13k` | ~13K | CNN | No | 2 Conv+BN+ReLU + Linear |
| `FCST_20k_NPU` | ~20K | CNN | Yes | High capacity with MaxPool |
| `FCST_LSTM8` | Varies | LSTM | No | Single LSTM (hidden=8) + Linear |
| `FCST_LSTM10` | Varies | LSTM | No | Single LSTM (hidden=10) + Linear |

### Audio Classification Models

| Model Name | Parameters | Architecture | NPU | Description |
|------------|------------|--------------|-----|-------------|
| `DSCNN_NPU` | ~9K | DSCNN | Yes | Depthwise separable CNN for keyword spotting; input (1, 49, 10) MFCC |

### Image Classification Models

| Model Name | Parameters | Architecture | NPU | Description |
|------------|------------|--------------|-----|-------------|
| `Lenet5` | ~60K | LeNet-5 | No | Classic CNN for image classification |
| `MobileNetV1_58k_NPU` | ~58K | MobileNetV1-style DW-Sep | Yes | Compact NPU-optimized image classifier |
| `MobileNetV2_58k_NPU` | ~58K | MobileNetV2-style DW-Sep | Yes | Inverted residual image classifier |

### Radar Point Cloud Classification Models

| Model Name | Parameters | Architecture | NPU | Description |
|------------|------------|--------------|-----|-------------|
| `Pose_and_Fall_model` | Varies | Linear (4-layer) | No | Human pose and fall detection from radar point-cloud data |

---

## Adding New Models

Want to add your own model? See the comprehensive guide: **[ADDING_NEW_MODELS.md](ADDING_NEW_MODELS.md)**

Key steps:
1. Add model class to `tinyml_modelzoo/models/`
2. Add class name to the file's `__all__` list
3. (Optional) Add device performance info to `device_info/run_info.py`
4. (Optional) Add model description to `model_descriptions/` for GUI integration

No changes required outside this repo.

---

## About the Task Types

**Classification** outputs a probability distribution over predefined classes. Best for: "Is this an A fault, B fault, or C fault?", "Which type of activity is this?"

**Regression** outputs a continuous numerical value. Best for: "What is the current torque?", "What will the temperature be?"

**Forecasting** predicts future values in a time series. Best for: "What will happen next?"

**Anomaly Detection** uses autoencoders to learn "normal" patterns; reconstruction error indicates anomalies. Best for: "Is this behavior normal?"

**Audio Classification** extracts MFCC features from a fixed-length audio window and classifies into keyword or sound categories. Best for: "What keyword was spoken?", "What sound event occurred?"

These categories can look similar from a distance, so here's how to tell them apart:

- **Anomaly Detection vs. Classification** — "Is it normal, or an anomaly?" is anomaly detection (binary outcome). "Is it normal, anomaly type A, type B, or type C?" is classification (multiple categories).
- **Classification vs. Regression** — predicting a **discrete** target (Class A / B / C, ...) from independent variables is classification; predicting a **continuous** target is regression.
- **Regression vs. Forecasting** — predicting a continuous target **Y** at the **same** time instant as its inputs is regression; predicting a variable's value at a **future** time instant is forecasting.

---

## Additional Resources

- [TI's Neural Network Compiler Documentation](https://software-dl.ti.com/mctools/nnc/mcu/users_guide/)
- [NPU Configuration Guidelines](docs/NPU_CONFIGURATION_GUIDELINES.md) - Design models optimized for TI NPU acceleration
- [Edge AI Studio for MCUs](https://www.ti.com/tool/download/EDGE-AI-STUDIO-MCU/) - No-code GUI for data collection & model development

---

## License

BSD 3-Clause License. See [LICENSE](LICENSE) for details.

---

**Questions or Issues?** Open an issue on [GitHub](https://github.com/TexasInstruments/tinyml-tensorlab/issues).
