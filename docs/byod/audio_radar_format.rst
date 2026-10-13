==============================
Audio and Radar Dataset Format
==============================

Tiny ML ModelZoo 1.5.0 adds two new task types, ``audio_classification`` and
``radar_classification``. Both read datasets with the same ``classes/`` layout used for
time series classification, so the folder name of each class becomes its label.
Read this page together with :doc:`data_splitting`, which applies unchanged.

.. contents::
   :local:
   :depth: 2

Audio Classification
--------------------

Task overview: :doc:`/task_types/audio_classification`.

Directory structure
^^^^^^^^^^^^^^^^^^^

.. code-block:: text

   my_audio_dataset/
   ├── classes/                      # default data_dir for audio
   │   ├── yes/
   │   │   ├── 0001.wav
   │   │   └── 0002.wav
   │   ├── no/
   │   │   └── 0001.wav
   │   └── _silence_/
   │       └── 0001.wav
   └── annotations/                  # optional, generated if missing
       ├── instances_train_list.txt
       ├── instances_val_list.txt
       └── instances_test_list.txt

* Audio files are read from ``classes/<class_name>/*.wav`` or ``*.flac`` (upper-case extensions are
  also accepted). Files nested deeper than one folder below ``classes/`` are not found.
* Every subfolder of ``classes/`` is a class. Files whose path contains ``_background_noise_`` are
  ignored when the split lists are generated.
* Set ``data_processing_feature_extraction.sampling_rate`` and ``audio_duration_ms`` to match your
  recordings. The shipped examples use 16000 Hz / 1000 ms (Google Speech Commands), 8000 Hz / 2000 ms
  (wake word), 44100 Hz / 1000 ms (glass break) and the 8820 Hz LPC preset (cough detection).
* The audio feature-extraction presets set ``mono: True``; ``normalize_audio`` is a preset option.
* Annotation lists use the same names and search order as other tasks (see
  :doc:`classification_format`); each line is a path such as ``yes/0001.wav``.

Preparing public datasets
^^^^^^^^^^^^^^^^^^^^^^^^^

1.5.0 includes helper scripts in the examples folder to turn raw or public data into this layout:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Script
     - Purpose
   * - ``examples/google_speech_command/generate_dataset.py``
     - Arranges the Google Speech Commands data into class folders: ten command words
       (``down, go, left, no, off, on, right, stop, up, yes``) plus ``_unknown_`` and ``_silence_``
       (silence clips are cut from ``_background_noise_``).
   * - ``examples/cough_detection/prepare_dataset.py``
     - Converts raw Edge AI Studio CSV recordings to segmented 8820 Hz WAV clips in class folders.
       Use ``--data-dir`` and ``--output-dir``.

The prepared datasets used by the examples are also available as ZIP files (see
:doc:`/examples/google_speech_command`, :doc:`/examples/wake_word_detection`,
:doc:`/examples/cough_detection` and :doc:`/examples/glass_break_detection`).

Config
^^^^^^

.. code-block:: yaml

   common:
     task_type: audio_classification
     target_device: MSPM0G5187

   dataset:
     dataset_name: my_audio_dataset
     input_data_path: /path/to/my_audio_dataset   # folder, ZIP or URL

   data_processing_feature_extraction:
     feature_extraction_name: GoogleSpeechCommands_MFCC_Default
     sampling_rate: 16000
     audio_duration_ms: 1000

Radar Classification
--------------------

Task overview: :doc:`/task_types/radar_classification`.

Directory structure
^^^^^^^^^^^^^^^^^^^

.. code-block:: text

   my_radar_dataset/
   ├── classes/                      # default data_dir for radar
   │   ├── standing/
   │   │   ├── capture1.csv
   │   │   └── capture2.csv
   │   ├── falling/
   │   │   └── capture1.csv
   │   └── lying/
   │       └── capture1.csv
   └── annotations/                  # optional, generated if missing
       ├── instances_train_list.txt
       ├── instances_val_list.txt
       └── instances_test_list.txt

* Files are ``.csv`` or ``.npy`` placed directly in a class folder.
* Each row is one frame of radar point-cloud features and the data is windowed with
  ``SimpleWindow`` (``frame_size`` rows per window).
* A CSV may have a header row. When the first row is non-numeric it is treated as the header and
  any column whose name contains ``session``, ``recording``, ``time``, ``track``, ``tid`` or
  ``classification`` is dropped. Remaining columns must be numeric.
* If different classes have different numbers of columns, all classes are truncated to the
  narrowest one (a warning is logged).
* Class names are the folder names, sorted alphabetically.

Preparing the public dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``examples/radar_pose_and_fall_detection/pose_and_fall_dataset.py`` is an optional preprocessing step
(point-count and height/distance filtering, train split). You only need it if you do not use the public
``radar_human_pose_detection`` dataset referenced in the example config. See
:doc:`/examples/radar_pose_and_fall_detection`.

Config
^^^^^^

This is the configuration of ``examples/radar_pose_and_fall_detection/config.yaml``:

.. code-block:: yaml

   common:
     task_type: radar_classification
     target_device: IWRL6432

   dataset:
     dataset_name: radar_human_pose_detection
     input_data_path: https://software-dl.ti.com/C2000/esd/mcu_ai/datasets/radar_human_pose_detection.zip
     split_type: within_files
     split_factor: [0.8, 0.1, 0.1]

   data_processing_feature_extraction:
     data_proc_transforms: [SimpleWindow]
     frame_size: 8
     stride_size: 0.5

See Also
--------

* :doc:`data_splitting` - split types and annotation lists
* :doc:`classification_format` - the time series classification layout this builds on
* :doc:`/features/early_stopping` - on by default in 1.5.0
