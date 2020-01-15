# SDC Client

`python asdc/client.py ${EXPERIMENT_DIR}/config.yaml` - start the sdc client.

## configuration

#### input/output files

| Name              | Type      | Description                                                             | Default        |
|-------------------|-----------|-------------------------------------------------------------------------|----------------|
| `target_file`     | `os.path` | csv file containing wafer coordinates, relative to ${EXPERIMENT_DIR}/`. | `map.csv`      |
| `data_dir`        | `os.path` | data directory, relative to `${EXPERIMENT_DIR}`.                        | `data/`        |
| `figure_dir`      | `os.path` | figure directory, relative to `${EXPERIMENT_DIR}`.                      | `figures/`     |
| `db_file`         | `os.path` | sqlite database file, relative to `data_dir`.                           | `sdc.db`       |
| `command_logfile` | `os.path` | log file for slack command history, relative to `data_dir`              | `commands.log` |


#### output configuration

| Name           | Type   | Description                  | Default |
|----------------|--------|------------------------------|---------|
| `notify_slack` | `bool` | post status updates to slack | `False` |
| `plot_cv`      | `bool` | post CV plot to slack        | `False` |
| `plot_current` | `bool` | post current plot to slack   | `False` |


#### debugging options
Here is a collection of debug flags that toggles debug functionality

| Name                 | Type                     | Description                                              | Default    |
|----------------------|--------------------------|----------------------------------------------------------|------------|
| `test`               | `bool`                   |                                                          | `False`    |
| `test_cell`          | `bool`                   | if set, replace potentiostat actions with a no-op.       | `False`    |
| `cell`               | `INTERNAL` or `EXTERNAL` | which cell to use                                        | `INTERNAL` |
| `confirm`            | `bool`                   | ask for permission before performing potentiostat action | `True`     |
| `confirm_experiment` | `bool`                   |                                                          | `False`    |

#### quality checks
| Name                 | Type    | Description                                                                   | Default |
|----------------------|---------|-------------------------------------------------------------------------------|---------|
| `coverage_threshold` | `float` | minimum coverage for a "good" deposit. (default                               | `0.9`)  |
| `current_threshold`  | `float` | average current heuristic for detecting "failed" electrodepositions. (default | `1e-5`) |

#### adafruit configuration
| Name            | Type  | Description                         | Default |
|-----------------|-------|-------------------------------------|---------|
| `adafruit_port` | `str` | serial port for the microcontroller | `COM9`  |


#### solutions handling
| Name              | Type                          | Description                                                  | Default      |
|-------------------|-------------------------------|--------------------------------------------------------------|--------------|
| `solutions`       | `Dict[str, Dict[str, float]]` | dictionary mapping syringe pumps to solution compositions.   |              |
| `pump_array_port` | `str`                         | serial port for the pump array                               | `COM10`      |
| `cleanup_pause`   | `float`                       | time delay for solution cleanup after an e-chem op. (default | `0` seconds) |

#### stage configuration
| Name                    | Type              | Description                                                                          | Default                 |
|-------------------------|-------------------|--------------------------------------------------------------------------------------|-------------------------|
| `initial_combi_position | Dict[str, float]` | manual specification of wafer coordinates for the initial reference frame alignment. | `{"x":  0.0, "y": 0.0}` |
| `frame_orientation`     | (default          | wafer orientation relative to the lab frame.                                         | `-y`                    |
| `stage_ip               | str`              | IP address of stage controller                                                       | `192.168.10.11`         |
| `speed                  | float`            | stage speed in m/s.                                                                  | `1e-3`, i.e. `1 mm/s`)  |
| `step_height            | float`            | step height in meters (default                                                       | `0.0011`                |

#### controller-specific options
| Name             | Type | Description                                      | Default             |
|------------------|------|--------------------------------------------------|---------------------|
| `experiment_file | str` | json file for predefined experimental protocol   | `instructions.json` |
| `domain_file     | str` | json file defining active learning design space. | `domain.json`       |


## SDC

::: asdc.client.SDC