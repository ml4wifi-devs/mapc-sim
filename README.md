# IEEE 802.11 MAPC (c-SR) Simulator

`mapc-sim` is a simulation tool for IEEE 802.11 Multi-Access Point Coordination (MAPC) scenarios with coordinated 
spatial reuse (C-SR). It provides a framework for modeling and analyzing the performance of wireless networks under 
various configurations and environmental conditions. A detailed description can be found in:

- Maksymilian Wojnar, Wojciech Ciezobka, Katarzyna Kosek-Szott, Krzysztof Rusek, Szymon Szott, David Nunez, and Boris Bellalta. "IEEE 802.11bn Multi-AP Coordinated Spatial Reuse with Hierarchical Multi-Armed Bandits", $JOURNAL_NAME_TODO, 2024. [[TODO_PREPRINT_INSERT](https://github.com/ml4wifi-devs/mapc-mab/tree/main), [TODO_PUBLICATION_INSERT](https://github.com/ml4wifi-devs/mapc-mab/tree/main)]

## Features

- **Simulation of C-SR network**: Simulate the behavior of nodes in a C-SR network, including the effects of hidden 
nodes, variable transmission power, node positions, and modulation and coding schemes (MCS). Calculate the aggregated 
effective data rate.
- **TGax channel model**: Incorporate the TGax channel model for realistic simulation in enterprise scenarios. The 
simulator also supports the effects of walls attenuation and random noise in the environment.

## Repository Structure

The repository is structured as follows:

- `mapc_sim/`: The main package containing the simulator.
  - `constants.py`: Physical and MAC layer constants used in the simulator.
  - `sim.py`: The main simulator functionality.
  - `utils.py`: Utility functions, including the TGax channel model.
- `test/`: Unit tests and benchmarking scripts.

## Installation

The package can be installed using pip:

```bash
pip install mapc-sim
```

## Usage

The main functionality is provided by the `network_data_rate` function in `mapc_sim/sim.py`. This function calculates 
the effective data rate for a given network configuration. Example usage:

```python
from mapc_sim.sim import network_data_rate

# Define your network configuration
# ...

data_rate = network_data_rate(key, tx, pos, mcs, tx_power, sigma, walls)
```

For more detailed examples, refer to the test cases in `test/test_sim.py`.

## Testing and Benchmarking

Run the unit tests to ensure everything is working correctly:

```bash
python -m unittest
```

You can benchmark the performance of the simulator using the script in `test/sim_benchmark.py`.

## Additional Notes

-   The simulator is written in JAX, an autodiff library for Python. It may require additional dependencies or 
configurations to run properly, especially with GPU acceleration. For more information on JAX, please refer to the 
official [JAX repository](https://jax.readthedocs.io/en/latest/).

## How to reference `mapc-sim`?

```
TODO
```
