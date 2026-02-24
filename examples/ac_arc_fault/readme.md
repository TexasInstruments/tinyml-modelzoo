# AC Arc Fault Detection

### Laavanaya Dhawan, Nathan Nohr, Vinamra Shrivastava, Akshat Aggarwal

## Overview

  The AC Arc Fault Detection application is an Edge AI solution that runs on the MSPM0G5187 microcontroller with integrated Neural Processing Unit (NPU). It detects series arc faults in residential and commercial electrical systems, which are a leading cause of electrical fires. This reference design combines the TIDA-010971 analog front end with machine learning inference to achieve high detection accuracy while maintaining immunity to masking loads per UL 1699 guidelines.

## Problem and Solution

  - Arc faults cause over 30,000 home fires annually, resulting in hundreds of deaths and over $1 billion in property damage
  - Traditional circuit breakers fail to detect arc faults because they operate at different frequencies
  - Edge AI enables complex pattern recognition, multi-feature analysis, and adaptive detection
  
## Key Performance Targets

  - Less than 10ms response time
  - Less than 10mW MCU active power consumption
  - Greater than 95% detection accuracy
  - Compliance with UL 1699 requirements

## System Components

1. Hardware:
- MSPM0G5187 microcontroller with integrated NPU [Link](https://www.ti.com/product/MSPM0G5187)
- TIDA-010971 Analog Front End with PCB Rogowski coil [Link](https://www.ti.com/lit/df/slvrbz4/slvrbz4.pdf?ts=1771928618141&ref_url=https%253A%252F%252Fwww.google.com%252F)

2. Software:
- Code Composer Studio 12.x or later
- MSPM0 SDK 2.08.00 or later
- TI Edge AI Studio

 ## Dataset Labelling

 For preparing and labelling your AC arc fault dataset, use the `AFCI_LabellingScript.py` script. Refer to
 [readme_labelling.md](readme_labelling.md) for detailed instructions on dataset format, labelling modes, and usage.


## Dataset Operations

  Data collection requires multiple scenarios:
  - Normal Operation: Inductive loads, LED lighting, power supplies, mixed loads
  - Arc Fault: Series and parallel faults
  - Masking Loads: Vacuum cleaners, drills, dimmers

  Our example dataset(ac_arc_fault_log300.zip) that is used in the config file includes:
  - 100+ captures per load type
  - Appliances: Dimmer, SMPS, Drill, Compressor, Resistive, Vaccum

## Feature Extraction Pipeline

  1. ADC Sampling: 512 samples at 107 kSps (~4.76ms per frame)
  2. Real FFT: 512-point FFT using ARM CMSIS-DSP
  3. Complex Magnitude Calculation
  4. DC Removal
  5. Binning: Average 8 adjacent FFT bins → 32 features
  6. Normalization to INT8 range
  7. Frame Concatenation: Stack 8 frames (256 total features)
  
## Model Architecture Options(GUI only Options)

  Four pre-configured model architectures:

1. ArcFault_model_200_t:

- Simplest, smallest & fastest model
- ~200 parameters, 3.6KB flash, 197.82μs inference time
- 99.60% accuracy

2. ArcFault_model_300_t:

- ~300 parameters, 3.9KB flash, 246.91μs inference time
- 99.60% accuracy

3. ArcFault_model_700_t:

- Sweet spot between speed & memory
- ~800 parameters, 4.5KB flash, 288.36μs inference time
- 99.42% accuracy

4. ArcFault_model_1400_t (Recommended):

- Most accurate model
- ~1600 parameters, 5.6KB flash, 397.51μs inference time
- 99.88% accuracy

## Model Architecture Options(Available on Tensorlab CLI Tools)

  Eleven pre-configured model architectures:

1. CLS_100_NPU:

    - Ultra-compact model
    - ~100 parameters, CNN architecture
    - NPU compatible

2. CLS_500_NPU:

    - Compact model
    - ~500 parameters, CNN architecture
    - NPU compatible
3. CLS_1k_NPU:
    - Lightweight 2-layer CNN
    - ~1K parameters, CNN architecture
    - NPU compatible
4. CLS_2k_NPU:

    - 2-layer model
    - ~2K parameters, CNN architecture
    - NPU compatible

5. CLS_ResAdd_3k:

    - Residual connections with addition
    - ~3K parameters, ResNet (Add) architecture
    - CPU only

6. CLS_ResCat_3k:

    - Residual connections with concatenation
    - ~3K parameters, ResNet (Cat) architecture
    - CPU only

7. CLS_4k_NPU:

    - Balanced model
    - ~4K parameters, CNN architecture
    - NPU compatible

8. CLS_6k_NPU:

    - Depthwise separable convolutions
    - ~6K parameters, CNN (DW-Sep) architecture
    - NPU compatible

9. CLS_8k_NPU:

    - Depthwise separable convolutions
    - ~8K parameters, CNN (DW-Sep) architecture
    - NPU compatible

10. CLS_13k_NPU:

    - Higher capacity model
    - ~13K parameters, CNN architecture
    - NPU compatible

11. CLS_20k_NPU:

    - High capacity model
    - ~20K parameters, CNN architecture
    - NPU compatible

## Training and Deployment Process

NOTE: Running the config yaml takes care of everything including feature extraction, training, quantization and compilation. 

1. Training:
- Use TI Edge AI Studio (GUI) or tinyml-tensorlab (CLI)
- Batch size: 50, Learning rate: 0.04, Optimizer: SGD
- Enable Quantize-Aware Training for INT8 accuracy
2. Quantization:
- INT8 quantization required-Enabled in the config file by default aswell as in Edge AI Studio
- ~4x reduction in model size
3. Compilation:
- TI Neural Network Compiler converts trained model
- Generates model.a, interface headers, and configuration

## Performance Metrics

  - End-to-end latency: <150ms (including 8-frame voting)
  - Model size: ~5.6KB flash
  - SRAM usage: ~24KB (out of 32KB available)
  - Detection accuracy: >99%
  - False positive rate: 0.01%
  - Precision: 99.97%
  - Recall: 99.72%
  - F1-Score: 99.84%

  This application enables engineers to develop Arc Fault Circuit Interrupter (AFCI) products for compliance with National Electrical Code (NEC) requirements while leveraging the power efficiency and performance of Edge AI on microcontrollers.

 ## How to Run

 After completing the repository setup, run the following command from the `tinyml-modelzoo` directory:


**Windows:** 
 ```bash
 .\run_tinyml_modelzoo.bat examples\ac_arc_fault\config_MSPM0.yaml
 ```

**Linux:**
 ```bash
 .\run_tinyml_modelzoo.bat examples\ac_arc_fault\config_MSPM0.yaml
 ```


## References

- MSPM0G5187 Technical Reference Manual [Link](https://www.ti.com/product/MSPM0G5187)
- UL 1699 Standard for Arc-Fault Circuit Interrupters [Link](https://code-authorities.ul.com/wp-content/uploads/2014/05/Dini2.pdf)
- [TI Neural Network Compiler Guide](https://software-dl.ti.com/mctools/nnc/mcu/users_guide/)
- TI Model Training Guide: [tinyml-tensorlab](https://github.com/TexasInstruments/tinyml-tensorlab/tree/main)
- [AC Arc Fault Detection Theory](https://en.wikipedia.org/wiki/Arc-fault_circuit_interrupter)
- EdgeAI Software Guide: SDK_INSTALL_DIR/docs/english/middleware/edgeAI/MSPM0_EdgeAI_User_Guide.html