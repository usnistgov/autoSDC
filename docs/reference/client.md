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
* `notify_slack: bool` -- post status updates to slack (`False`)
* `plot_cv: bool` -- post CV plot to slack (`False`)
* `plot_current: bool` -- post current plot to slack (`False`)

#### debugging options
Here is a collection of debug flags that toggles debug functionality

* `test: bool` (`False`)
* `test_cell: bool` -- if set, replace potentiostat actions with a no-op. (`False`)
* `cell: {'INTERNAL', 'EXTERNAL'}` -- which cell to use (`INTERNAL`)
* `confirm: bool` -- ask for permission before performing potentiostat actions (`True`)
* `confirm_experiment: bool` (`False`)

#### quality checks
* `coverage_threshold: float` -- minimum coverage for a "good" deposit. (default: `0.9`)
* `current_threshold: float` -- average current heuristic for detecting "failed" electrodepositions. (default: `1e-5`)

#### adafruit configuration
* `adafruit_port: str` -- serial port for the microcontroller (default: `COM9`)

#### solutions handling
* `solutions: Dict[str, Dict[str, float]]` -- dictionary mapping syringe pumps to solution compositions.
* `pump_array_port`: serial port for the pump array (default: `COM10`)
* `cleanup_pause: Union[int, float]` -- time delay for solution cleanup after an e-chem op. (default: `0` seconds)

#### stage configuration
* `initial_combi_position: Dict[str, float]` -- manual specification of wafer coordinates for the initial reference frame alignment. `{"x": 0.0, "y": 0.0}`
* `frame_orientation` -- wafer orientation relative to the lab frame. (default: `-y`)
* `stage_ip: str` -- IP address of stage controller (default: `192.168.10.11`)
* `speed: float` stage speed in m/s. (default: `1e-3`, i.e. `1 mm/s`)
* `step_height: float` -- step height in meters (default: `0.0011`)

#### controller-specific options
* `experiment_file: str` -- "instructions.json"
* `domain_file: str` -- json file defining active learning design space. (default: `domain.json`)



## SDC

::: asdc.client.SDC