# IEEE 802.11 MAPC Coordinated Spatial Reuse (C-SR) Simulator

`mapc-sim` is a simulation tool for IEEE 802.11 Multi-Access Point Coordination (MAPC) scenarios with coordinated 
spatial reuse (C-SR). It provides a framework for modeling and analyzing the performance of wireless networks under 
various configurations and environmental conditions. A detailed description can be found in:

- Maksymilian Wojnar, Wojciech Ciezobka, Katarzyna Kosek-Szott, Krzysztof Rusek, Szymon Szott, David Nunez, and Boris Bellalta. "IEEE 802.11bn Multi-AP Coordinated Spatial Reuse with Hierarchical Multi-Armed Bandits", $JOURNAL_NAME_TODO, 2024. [[TODO_PREPRINT_INSERT](https://github.com/ml4wifi-devs/mapc-mab/tree/main), [TODO_PUBLICATION_INSERT](https://github.com/ml4wifi-devs/mapc-mab/tree/main)]

## Features

- **Simulation of C-SR**: You can simulate the C-SR performance of an 802.11 network, including the effects of hidden 
nodes, variable transmission power, node positions, and modulation and coding schemes (MCS). Calculate the aggregated 
effective data rate.
- **TGax channel model**: The simulator incorporates the TGax channel model for realistic simulation in enterprise scenarios. The 
simulator also supports the effects of wall attenuation and random noise in the environment.
- **JAX JIT compilation**: The simulator is written in JAX, which enables just-in-time (JIT) compilation and hardware acceleration.
- **Reproducibility**: The simulator uses JAX's pseudo random number generator (PRNG) to generate random numbers. This ensures that the
simulator is fully reproducible and you will get the same results for the same input parameters.

## Repository Structure

The repository is structured as follows:

- `mapc_sim/`: Main package containing the simulator.
  - `constants.py`: Physical and MAC layer constants used in the simulator.
  - `sim.py`: Main simulator code.
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
import jax
import jax.numpy as jnp
from mapc_sim.sim import network_data_rate

# Random number generator key
key = jax.random.PRNGKey(42)

# Transmission matrix - 1 if node i transmits to node j, 0 otherwise
tx = jnp.zeros((n_nodes, n_nodes))
tx = tx.at[i_0, j_0].set(1)
tx = tx.at[i_1, j_1].set(1)
...
tx = tx.at[i_n, j_n].set(1)

# Node positions
pos = jnp.array([
    [x_0, y_0],
    [x_1, y_1],
    ...
    [x_n, y_n],
])

# MCS values of transmitting nodes
mcs = jnp.array([mcs_0, mcs_1, ..., mcs_n], dtype=int)

# Transmission power of transmitting nodes
tx_power = jnp.array([tx_power_0, tx_power_1, ..., tx_power_n])

# Standard deviation of the white Gaussian noise
sigma = 2.

# Walls matrix - 1 if there is a wall between node k and node l, 0 otherwise
walls = jnp.zeros((n_nodes, n_nodes))
walls = walls.at[k_0, l_0].set(1)
walls = walls.at[k_1, l_1].set(1)
...
walls = walls.at[k_m, l_m].set(1)

# Calculate the effective data rate with the simulator
data_rate = network_data_rate(key, tx, pos, mcs, tx_power, sigma, walls)
```

For more detailed examples, refer to the test cases in `test/test_sim.py`.

### JAX JIT Compilation

The simulator is written in JAX, which enables just-in-time (JIT) compilation and hardware acceleration. 
The use of JIT is strongly recommended as it can improve the performance of the simulator by orders of magnitude.
To enable JIT, apply the `jax.jit` transformation on the simulator function:
 
```python
import jax
from mapc_sim.sim import network_data_rate

# Define your network configuration
# ...

network_data_rate_jit = jax.jit(network_data_rate)
data_rate = network_data_rate_jit(key, tx, pos, mcs, tx_power, sigma, walls)
```

As the `jax.jit` transformation can be applied to any function, you can also use it to JIT-compile closures. 
For example, you can JIT-compile the `network_data_rate` function with a fixed network configuration as follows:

```python
from functools import partial

import jax
from mapc_sim.sim import network_data_rate

pos = ...
walls = ...

network_data_rate_jit = jax.jit(partial(
    network_data_rate,
    pos=pos,
    walls=walls,
))

# Define the remaining values
# ...

data_rate = network_data_rate_jit(key=key, tx=tx, mcs=mcs, tx_power=tx_power, sigma=sigma)
```

### Reproducibility

The simulator uses JAX's PRNG. This ensures that the simulator is fully reproducible. However, the same key should 
be used at most once for each simulation so that the results are not correlated. For example, you can generate a new 
key and split it into two keys in each step of a simulation:

```python
import jax
from mapc_sim.sim import network_data_rate

# Define your network configuration
# ...

key = jax.random.PRNGKey(42)

for _ in range(n):
    # Generate two new keys, one for the current step and one for the next splits
    key, subkey = jax.random.split(key)
    data_rate = network_data_rate(subkey, tx, pos, mcs, tx_power, sigma, walls)
```

### 64-bit Floating Point Precision

If you want to use 64-bit floating point precision, you can set the appropriate environment variable before running
your script:

```bash
export JAX_ENABLE_X64="True
```

Alternatively, you can set the environment variable in your Python script:

```python
import os
os.environ["JAX_ENABLE_X64"] = "True"
```

## Testing and Benchmarking

Run the unit tests to ensure everything is working correctly:

```bash
python -m unittest
```

You can benchmark the performance of the simulator using `test/sim_benchmark.py`.

## Additional Notes

-   The simulator is written in JAX, an autodiff library for Python. It may require additional dependencies or 
configurations to run properly, especially with hardware acceleration. For more information on JAX, please refer 
to the official [JAX repository](https://jax.readthedocs.io/en/latest/).

## How to reference `mapc-sim`?

```
TODO
```
