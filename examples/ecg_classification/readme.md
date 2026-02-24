# ECG Classification


## Overview

  The ECG Classification application is an Edge AI solution that runs on the MSPM0G5187 microcontroller with integrated Neural Processing Unit (NPU). It classifies electrocardiogram signals into different cardiac conditions in real-time, enabling portable and low-power heart monitoring devices. This reference design combines the AFE1594 analog front-end with machine learning inference to achieve high classification accuracy for cardiac health assessment.

## Problem and Solution

  - Cardiovascular diseases are the leading cause of death globally, accounting for approximately 17.9 million deaths annually
  - Early detection of cardiac abnormalities through continuous ECG monitoring can significantly improve patient outcomes
  - Traditional ECG analysis requires expensive equipment and trained medical professionals
  - Edge AI enables real-time cardiac classification on portable, low-power devices without cloud connectivity

## Key Performance Targets

  - Real-time ECG classification
  - Low power consumption suitable for wearable devices
  - Greater than 95% classification accuracy
  - Immediate visual feedback via LED indicators

## System Components

  1. Hardware:
    - MSPM0G5187 microcontroller with integrated NPU https://www.ti.com/product/MSPM0G5187
    - AFE1594 Analog Front End for ECG signal acquisition
    - LP-MSPM0G5187 LaunchPad Development Kit
  2. Software:
    - Code Composer Studio 12.x or later
    - MSPM0 SDK 2.08.00 or later
    - TI Edge AI Studio

## Dataset Information

  The example uses the ecg_classification_4class dataset which contains labeled ECG recordings categorized into different cardiac conditions:

  - Normal: Healthy sinus rhythm
  - Mild: Minor cardiac abnormalities
  - Other: Other cardiac conditions requiring attention

## Feature Extraction Pipeline

  1. ECG Signal Acquisition: 2500 samples per frame
  2. Signal Normalization
  3. Roundoff processing for INT8 compatibility
  4. Single frame input to model

## Model Architecture Options (Available on Tensorlab CLI Tools)

  1. ECG_55k_NPU (Default):
    - CNN architecture optimized for NPU
    - ~55K parameters
    - NPU compatible
    - Balanced accuracy and performance


## Model Performance

  - Accuracy: ~97%

## Training and Deployment Process

NOTE: Running the config yaml handles everything including feature extraction, training, quantization, and compilation.

  1. Training:
    - Use TI Edge AI Studio (GUI) or tinyml-tensorlab (CLI)
    - Batch size: 12, Learning rate: 0.001, Optimizer: Adam
    - Weight decay: 4e-5
    - Training epochs: 25
  2. Quantization:
    - INT8 quantization enabled by default
    - Maintains accuracy while reducing model size
  3. Compilation:
    - TI Neural Network Compiler converts trained model
    - Generates model.a, interface headers, and configuration
    - Optimized for NPU execution

## How to Run

  After completing the repository setup, run the following command from the tinyml-modelzoo directory:

```bash
  Windows:
  .\run_tinyml_modelzoo.bat examples\ecg_classification\config_MSPM0.yaml
```

```bash
  Linux:
  ./run_tinyml_modelzoo.sh examples/ecg_classification/config_MSPM0.yaml
```
## Available Default Configurations For Each Device Family

 
  - config.yaml - AM13E2
  - config_MSPM0.yaml - MSPM0G5187

## References

  - MSPM0G5187 Technical Reference Manual https://www.ti.com/product/MSPM0G5187
  - AFE1594 ECG Analog Front End https://www.ti.com/product/AFE1594
  - https://software-dl.ti.com/mctools/nnc/mcu/users_guide/
  - TI Model Training Guide: https://github.com/TexasInstruments/tinyml-tensorlab/tree/main
  - https://en.wikipedia.org/wiki/Electrocardiography



