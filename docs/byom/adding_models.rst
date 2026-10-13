=========================
Adding Models to ModelZoo
=========================

This page is a short reference for registering your own model in Tiny ML ModelZoo. The validated step-by-step
walk-through, with a worked example and troubleshooting, is :doc:`/BRING_YOUR_OWN_MODEL`. The full reference is
`ADDING_NEW_MODELS.md <https://github.com/TexasInstruments/tinyml-modelzoo/blob/main/ADDING_NEW_MODELS.md>`_ in the
repository.

.. note::

   Adding models edits the ModelZoo source, so you need the **clone install** (clone
   `tinyml-modelzoo <https://github.com/TexasInstruments/tinyml-modelzoo>`_ and run ``pip install -e .``). See
   :doc:`/installation/developer_installation`. You only change files in ``tinyml-modelzoo``; nothing in ModelMaker or
   TinyVerse has to change, because models are discovered automatically from ``tinyml_modelzoo``.

The two names
=============

A model has two different names. Mixing them up is the most common cause of "model not found" errors.

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Name
     - Where it is defined
     - Where it is used
   * - Description name (for example ``CLS_1k_NPU``)
     - Key of the model description dict in ``tinyml_modelzoo/model_descriptions/<task>.py``. It must also be listed
       in ``enabled_models_list`` in that file.
     - ``training.model_name`` in the config YAML.
   * - Class name (for example ``CNN_TS_GEN_BASE_1K_NPU``)
     - Python class in ``tinyml_modelzoo/models/<task>.py``. It must be listed in that file's ``__all__``.
     - ``model_training_id`` in the description entry.

.. code-block:: text

   config.yaml: training.model_name --> model_descriptions/<task>.py: description[<name>]  (in enabled_models_list)
                                            `-- training.model_training_id --> models/<task>.py: class <name> (in __all__)

If no enabled description matches ``training.model_name``, the run stops with
``please check if the given model_name is a supported one: <name>``.

Files per task type
===================

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Task type
     - Model classes (``tinyml_modelzoo/models/``)
     - Descriptions (``tinyml_modelzoo/model_descriptions/``)
   * - Time series classification
     - ``classification.py``
     - ``classification.py``
   * - Time series regression
     - ``regression.py``
     - ``regression.py``
   * - Time series anomaly detection
     - ``anomalydetection.py``
     - ``anomalydetection.py``
   * - Time series forecasting
     - ``forecasting.py``
     - ``forecasting.py``
   * - Image, audio, radar classification
     - ``image.py``, ``audio.py``, ``radar.py``
     - no description files in 1.5.0 (see the repository reference)

Writing the model class
=======================

Both styles work:

* **Spec-based**: inherit from ``GenericModelWithSpec`` and describe the network as an ordered dictionary of layers in
  ``gen_model_spec()``. This is the style used by most ModelZoo models.
* **Plain** ``torch.nn.Module``: write the network directly in PyTorch.

Keep the constructor signature ``__init__(self, config, input_features=..., variables=..., num_classes=...)``. The
pipeline fills in these values from your dataset and feature extraction settings, so do not hard-code them. Time
series input reaches the model as ``(batch, variables, input_features, 1)``.

Add the class name to the ``__all__`` list of the model file.

Writing the description
=======================

Add an entry to the descriptions dict, starting from ``template_model_description`` and overriding only what is
specific to your model. The important ``training`` fields are:

* ``model_name``: the description name, equal to the dict key.
* ``model_training_id``: the class name, exactly as in the model file.
* ``target_devices``: the devices the model can be compiled for.

Then append the description name to ``enabled_models_list`` at the bottom of the file. Check that the previous entry ends
with a comma; otherwise Python silently joins the two strings and the model is reported as unsupported.

Using the model
===============

Copy an example config and set the model name:

.. code-block:: yaml

   training:
     model_name: 'MY_CLS_2k_NPU'

Then run it as any other example (see :doc:`/getting_started/running_examples`). For NPU devices, follow the design
guidelines in :doc:`/NPU_CONFIGURATION_GUIDELINES` and check the footprint early with :doc:`/features/memory_estimation`.

Next steps
==========

* :doc:`/BRING_YOUR_OWN_MODEL` for the step-by-step guide
* :doc:`compilation_only` to compile a model that is already trained
* :doc:`/features/quantization` for quantizing your model
