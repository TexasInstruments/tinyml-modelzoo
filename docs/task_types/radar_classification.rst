====================
Radar Classification
====================

Radar classification assigns a label to windows of radar point-cloud features, for example human pose
(standing, sitting, lying) or a fall. It is new in Tiny ML ModelZoo 1.5.0.

Overview
--------

**What it does**: Takes a window of consecutive radar frames (point-cloud features per frame) and outputs a
class label.

**Example**: 8 frames of point-cloud height and position features -> "standing", "falling", ...

**Use case with an example**: :doc:`/examples/radar_pose_and_fall_detection`

Task Settings
-------------

.. list-table::
   :widths: 30 70

   * - ``task_type``
     - ``radar_classification``
   * - ``target_module``
     - ``radar`` (inferred from ``task_type`` if you omit it)
   * - Target device
     - ``IWRL6432`` (the only radar target device in 1.5.0; compiled with the TI Arm Clang toolchain)
   * - Data folder
     - ``classes/<class_name>/*.csv`` (or ``*.npy``)
   * - Early stopping
     - On by default in 1.5.0 (``training.early_stopping``, ``early_stopping_patience: 5``);
       see :doc:`/features/early_stopping`

Configuration
-------------

This is ``examples/radar_pose_and_fall_detection/config.yaml``:

.. code-block:: yaml

   common:
     task_type: radar_classification
     target_module: radar
     target_device: IWRL6432

   dataset:
     dataset_name: radar_human_pose_detection
     input_data_path: https://software-dl.ti.com/C2000/esd/mcu_ai/datasets/radar_human_pose_detection.zip
     split_type: within_files
     split_factor: [0.8, 0.1, 0.1]

   data_processing_feature_extraction:
     data_proc_transforms:
       - SimpleWindow
     frame_size: 8
     stride_size: 0.5

   training:
     model_name: Pose_and_Fall_model
     batch_size: 64
     training_epochs: 160
     early_stopping: true
     early_stopping_patience: 5
     num_gpus: 0
     quantization: 0
     learning_rate: 0.0001
     optimizer: SGD

   testing: {}
   compilation: {}

Run it from the tinyml-modelzoo folder:

.. code-block:: bash

   ./run_tinyml_modelzoo.sh examples/radar_pose_and_fall_detection/config.yaml

See :doc:`/getting_started/quickstart` for install options.

Dataset Format
--------------

.. code-block:: text

   my_radar_dataset/
   └── classes/
       ├── class_a/
       │   └── capture1.csv
       └── class_b/
           └── capture1.csv

Each CSV row is one radar frame. Non-numeric header columns such as session, recording, time, track id
and classification columns are dropped automatically. Windows of ``frame_size`` rows are built with
``SimpleWindow`` (``stride_size`` controls the overlap). Details and the optional dataset preparation
script are in :doc:`/byod/audio_radar_format`.

Available Models
----------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Model Name
     - Description
   * - ``Pose_and_Fall_model``
     - Four-layer linear network for radar point-cloud data (``LINEAR_4L_PC`` in
       ``tinyml_modelzoo/models/radar.py``), target device ``IWRL6432``

For your own architecture see :doc:`/BRING_YOUR_OWN_MODEL`.

Metrics
-------

The training log reports accuracy, F1-score and the confusion matrix like the other classification tasks;
see :doc:`/getting_started/understanding_run_output`.

Tips
----

* The example uses ``within_files`` splitting with ``[0.8, 0.1, 0.1]`` because each capture is a long
  continuous recording; see :doc:`/byod/data_splitting`.
* ``quantization: 0`` in the example trains a float model. Check the example page for the quantized
  workflow before changing it.

See Also
--------

* :doc:`/byod/audio_radar_format` - Radar dataset layout
* :doc:`timeseries_classification` - Time series classification
* :doc:`audio_classification` - The other new 1.5.0 task type
