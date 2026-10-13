==============================
Radar Pose and Fall Detection
==============================

Classify the posture of a person (standing, sitting, lying, falling, walking) from mmWave radar point cloud data on
the IWRL6432. This is the radar point cloud classification example. New in 1.5.0.

Example folder: ``examples/radar_pose_and_fall_detection``
(`GitHub <https://github.com/TexasInstruments/tinyml-modelzoo/tree/main/examples/radar_pose_and_fall_detection>`_)

Overview
--------

* **Task**: radar point cloud classification (``radar_classification``, ``target_module: radar``), 5 classes
* **Application**: human pose recognition and fall detection from radar point clouds, for example for elder care
  and in-home monitoring
* **Device**: IWRL6432 (``config.yaml``, the only config in the folder). It is the only target device of the radar
  module; the compiler toolchain is ``ti-cgt-armllvm`` for the Cortex-M4F.
* **Model**: ``Pose_and_Fall_model``, a four-layer fully connected network (BatchNorm and Linear layers
  64, 32, 16 and 5 outputs, implemented as ``LINEAR_4L_PC`` in ``tinyml_modelzoo/models/radar.py``)

The Dataset
-----------

The public dataset ``radar_human_pose_detection`` is downloaded by the config from
``https://software-dl.ti.com/C2000/esd/mcu_ai/datasets/radar_human_pose_detection.zip``. If the download fails on a
corporate network, download the zip manually and set ``dataset.input_data_path`` to the extracted folder. The data
is organised in class folders (``classes/``), one folder per posture:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Index
     - Class
   * - 0
     - ``STANDING``
   * - 1
     - ``SITTING``
   * - 2
     - ``LYING``
   * - 3
     - ``FALLING``
   * - 4
     - ``WALKING``

Each CSV row is a radar point cloud frame with the tracker fields (``posx``, ``posy``, ``posz``, ``velx``, ``vely``,
``velz``, ``accx``, ``accy``, ``accz``).

Preparing Your Own Data (Optional)
----------------------------------

``pose_and_fall_dataset.py`` is an optional preprocessing script for users who do not use the public dataset. Put
your CSV files in class folders under ``dataset/classes/`` in the working directory, set ``DATASET_DIR`` in the
script and run it. It:

* adds missing tracker columns with a default of 0 and labels each row with its class,
* drops rows with fewer than ``MIN_POINTS = 5`` points or an invalid track ID,
* filters points by height (-4 to 3 m) and distance (-4 to 5 m),
* keeps the highest and lowest points of each point cloud frame as the model input,
* writes the cleaned data to ``classes_cleaned/`` and creates ``instances_train_list.txt`` and
  ``instances_test_list.txt`` with an 80/20 split.

Running the Example
-------------------

Install Tiny ML ModelZoo first (:doc:`/installation/index`, :doc:`/getting_started/quickstart`), then run from
the ``tinyml-modelzoo`` folder:

.. tabs::

   .. tab:: Linux / macOS

      .. code-block:: bash

         ./run_tinyml_modelzoo.sh examples/radar_pose_and_fall_detection/config.yaml

   .. tab:: Windows

      See :doc:`/getting_started/running_examples` for all launchers, for example:

      .. code-block:: bat

         run_tinyml_modelzoo.bat examples\radar_pose_and_fall_detection\config.yaml

Configuration
-------------

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

* ``data_dir`` is detected automatically as ``classes`` for radar classification.
* ``split_type: within_files`` splits every file into 80% train, 10% validation and 10% test windows.
* ``SimpleWindow`` with ``frame_size: 8`` groups 8 consecutive point cloud frames per sample.
* ``quantization: 0`` trains a float model (the compilation stage uses the software path for float models).

Results
-------

The example folder has no readme with accuracy numbers. In the automated 1.5.0 config test run (test set of 1999
windows: 380 falling, 324 lying, 491 sitting, 544 standing, 260 walking) the float model reached:

.. list-table::
   :header-rows: 1
   :widths: 60 40

   * - Metric
     - Value
   * - Best epoch (of 160, early stopping enabled)
     - 141
   * - Validation accuracy / F1 score (best epoch)
     - 96.27% / 0.963
   * - Test accuracy
     - 96.10%
   * - Test AUC ROC
     - 1.000

Output Location
---------------

Results are written to ``./data/projects/radar_human_pose_detection/run/<date-time>/Pose_and_Fall_model/``. The trained
model is in ``training/base`` and the compiled model in ``compilation``. See
:doc:`/getting_started/understanding_run_output` and :doc:`/task_types/radar_classification` for the radar task type.

See Also
--------

* :doc:`/task_types/radar_classification`
* :doc:`dynamic_hand_gesture_recognition`
* :doc:`fall_detection_classification`
