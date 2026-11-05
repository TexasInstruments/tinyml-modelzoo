=====================================
TI CCStudio Tiny ML Workflow (Agentic Skill)
=====================================

An AI agent skill that takes you through the whole tinyml-modelzoo flow, from a dataset
to a model flashed on a device, inside TI Code Composer Studio (CCStudio) or VS Code.

Instead of editing YAML and running scripts by hand, you describe what you want
("Help me create an AI solution for hand gesture recognition using 3-axis acclerometer data, to be deployed on the MSPM0G5187") and the agent asks the questions it needs,
builds the config, runs training and compilation, and deploys the result to a
CCStudio project.

.. note::
  Currently we make the tinyml-workflow skill available via only the TI CCStudio Edge AI Toolkit. 
  Future releases will aim to make a generic version available that is accessible via the terminal as well.

What the Skill Does
-------------------

The skill automates the same steps as :doc:`quickstart` and :doc:`first_example`, and
adds checks at each stage so a mistake is caught before it shows up as a linker error
or a corrupted binary.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Phase
     - What happens
   * - Project definition and import
     - Pick the task type and target device, then import a matching SDK AI example
       template into a new CCStudio project. All configs, training runs and compiled
       artifacts are stored in that project folder itself.
   * - Data analysis and preparation
     - Check the dataset format, analyze its statistics (distribution, normalization,
       class imbalance) and choose feature extraction and data processing transforms.
   * - Model and training configuration
     - Provide user a ranked list of model candidates from the modelzoo to choose from, then
       set the training, testing and compilation sections, including the quantization
       mode.
   * - Build, compile and deploy
     - Run training and compilation with live logs, verify the ``mod.a`` artifact and
       its FLASH and SRAM footprint, then build and flash the device project.

The agent shows the config built so far after each section and waits for your approval
before it continues.

Prerequisites
-------------

* **CCStudio** with the Edge AI Toolkit, or **VS Code** with the CCStudio extension.
  The CCStudio MCP servers (``ccs-project``, ``ccs-debug``) or corresponding project CLI tools (VS Code IDE) are used for project import,
  build and flash when they are available. Without them, the agent asks you to do those
  steps in the IDE and report back.
* The **tinyml-modelzoo** install (see :doc:`/installation/index`) in a virtual
  environment.
* The compiler for your target device (see
  :doc:`/installation/environment_variables`). Without it, training still works but
  compilation does not.
* A **pre-processed dataset** in CSV, TXT (space or comma delimited) or NumPy format.
  The skill does not clean or format raw data. It only validates the format before training.

.. note:: 
  The CCStudio Edge AI Toolkit sets up the skill in your workspace's ``.claude/`` folder and 
  generates the ``.env`` file. This toolkit also takes care of modelzoo installation (as a pip wheel).

Starting the Workflow
---------------------

Invoke the skill with its command, or describe the task in natural language:

.. code-block:: text

   /tinyml-agent-skills:ti-ccstudio-tinyml-workflow

Workflow Steps
--------------

The workflow runs in numbered steps. Each step has a gate, a point the agent must pass
before it continues.

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Step
     - What it does
   * - 1
     - Session setup: load ``.env``, check the virtual environment, detect the IDE and
       its project tools.
   * - 2
     - Requirements: task type and target device.
   * - 3
     - Project import: find the SDK, pick a template, confirm the template's hardware
       and software prerequisites, import the project.
   * - 3.5
     - Bundled example match: use a ModelZoo example as-is, use its template with your
       own dataset, or start from scratch.
   * - 4 to 8
     - Dataset path and task name, common section, dataset validation, dataset analysis,
       feature extraction and data processing.
   * - 8C
     - Model selection from the ranked recommendations.
   * - 9 to 11
     - Training section (including quantization), testing section and compilation
       section.
   * - 12
     - Assemble ``config.yaml`` with absolute output paths in the CCStudio project.
   * - 13
     - Run training and compilation. The compiler path must be set in the same call that
       runs training.
   * - 13A
     - Verify artifacts: ``mod.a`` must exist and FLASH and SRAM usage must fit the
       device.
   * - 13B
     - Review the result of training & check if user is satisfied with the model or wants to retrain. 
       This is required for your own data (BYOD) and skipped for a bundled example used as-is.
   * - 14
     - Deploy: verify, exclude training outputs from the Project build, copy the model
       and golden vectors into the project, build, and flash.

Gates
-----

The gates stop the workflow when a check fails. They are the reason the skill is slower
than running the commands yourself, and the reason a deployment is less likely to fail
at the end.

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Gate
     - Must pass before
   * - G0
     - Importing a template. The template must match the task type, or you must choose
       another template. On an NPU device, you choose between the NPU and the regular
       project.
   * - G0b
     - Importing a template. You confirm you have the template's hardware and software
       prerequisites.
   * - G1
     - Training. The compiler root is set in the same shell call as the training command.
   * - G2
     - Deployment. ``mod.a`` exists and the footprint fits the device.
   * - G3
     - Deployment (CCStudio). The training and compilation outputs are excluded from the
       project build, and you confirm the exclusion list.
   * - G4
     - Build. The placeholder library and header in the template are replaced by the
       trained output, and the copy is verified with ``cmp``.
   * - G4b
     - Build (BYOD case). A golden-vector test is wired into the project and
       passes. See ``golden_vector_test.md`` in the skill references.
   * - G5
     - Flashing. The build finished with no errors.
   * - G6
     - Deployment (BYOD case). You confirm the result is acceptable.

Outputs
-------

Everything the workflow produces is in the project itself, under the run folder
``<project>/run/<date-time>/<model_name>/``. The layout is the same as the one described
in :doc:`understanding_run_output`:

* ``training/base/`` and ``training/quantization/``: the float and quantized models.
* ``compilation/artifacts/``: ``mod.a`` and ``tvmgen_default.h``, the files the device
  project uses.
* ``config.yaml``: the complete config the workflow built. You can rerun it with
  tinyml-modelzoo at any time.

Troubleshooting
---------------

**Training runs but compilation fails**
   Check the compiler root. The variable must be set in the same shell command as the
   training run. The skill natively handles this in normal circumstances, but if required, prompt the agent to do so.

**The model does not fit the device**
   The skill stops at gate G2 and does not deploy. Choose a smaller model in Step 8C
   and run training again.

**The build or flash fails in CCStudio**
   Ask the agent to diagnose the error. It reads the build log and the device
   deployment guide in the skill references.

Next Steps
----------

* :doc:`quickstart` - Run the hello world example from the command line
* :doc:`understanding_run_output` - Understand the files the workflow produces
* :doc:`/CONFIG_YAML_GUIDE` - Edit the config the workflow generates
* :doc:`/DEPLOYING_TO_SDK_EXAMPLE` - Deploy a model to an SDK example by hand
* :doc:`/byod/index` - Bring your own dataset