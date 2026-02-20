# Hello World

  A simple timeseries classification example that classifies sine, square, and sawtooth waveforms.

  ## How to Run

  After completing the repository setup, run the following command from the `tinyml-modelzoo` directory:

  **Windows:**
  ```bash
  .\run_tinyml_modelzoo.bat examples\hello_world\<config_file>
```
** Linux:**
  ```bash
  ./run_tinyml_modelzoo.sh examples/hello_world/<config_file>
```
 ## Available Configurations

  ┌────────────────────┬───────────────────────┐
  │    Config File     │     Target Device     │
  ├────────────────────┼───────────────────────┤
  │ config.yaml        │ F28P55                │
  ├────────────────────┼───────────────────────┤
  │ config_MSPM0.yaml  │ MSPM0                 │
  ├────────────────────┼───────────────────────┤
  │ config_CC1352.yaml │ CC1352                │
  ├────────────────────┼───────────────────────┤
  │ config_CC2755.yaml │ CC2755                │
  └────────────────────┴───────────────────────┘

 
