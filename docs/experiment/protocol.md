# Protocol

`asdc` uses a json protocol to define experiments and actions, to support logging, experiment composition, and communication between multiple processes/machines.

## electrochemistry

### potentiostatic

Apply a constant potential for a fixed interval.

**Parameters**

| Name            | Type    | Description                | Default |
|-----------------|---------|----------------------------|---------|
| `potential`     | `float` | applied potential (V)      | None    |
| `duration`      | `float` | operation duration (s)     | 10      |
| `current_range` | `str`   | current range to use       | 'AUTO'  |

Example:

```python
{
    "op": "potentiostatic",
    "potential": 0.5,
    "duration": 30
}
```

### CV

Multi-cycle cyclic voltammetry -- apply a triangular wave potential pattern and measure current.

**Parameters**

| Name                 | Type    | Description                              | Default |
|----------------------|---------|------------------------------------------|---------|
| `initial_potential`  | `float` | starting potential (V)                   | None    |
| `vertex_potential_1` | `float` | first vertex potential in the cycle (V)  | None    |
| `vertex_potential_2` | `float` | second vertex potential in the cycle (V) | None    |
| `initial_potential`  | `float` | final potential (V)                      | None    |
| `scan_rate`          | `float` | operation duration (s)                   | None    |
| `cycles`             | `int`   | operation duration (s)                   | None    |
| `current_range`      | `str`   | post current plot to slack               | 'AUTO'  |

Example:

```python
{
    "op": "cv",
    "initial_potential": 0.0,
    "vertex_potential_1": -1.0,
    "vertex_potential_2": 1.2,
    "final_potential": 0.0,
    "scan_rate": 0.075,
    "cycles": 2
}
```
