# Multi-Agent System (MAS) Consensus Simulation

This project implements a distributed consensus simulation in a multi-agent system. Each agent attempts to reach a common value (the global average) through repeated exchanges with its neighbors.

## Program Overview and Task Description

### Task:

- Simulate a set of agents on a given network (graph).
- Each agent has a state (an initial numerical value).
- Agents communicate locally with their neighbors to approach consensus (the global average).
- The program runs the simulation until convergence or until a maximum number of rounds.
- Provide unit tests validating correct functionality, including extreme cases.

### Language:

- Python 3 (using the standard library).

### Plan:

- A module `consensus.py` containing:
  - The main classes/functions (Agent, Network, ConsensusSimulation).
  - Logic for step-by-step execution, convergence detection, error handling.
- A module `test_consensus.py` containing exhaustive unit tests (using `unittest`).

## Features

- **Agent Class**: Represents an individual agent in the network with a unique identifier, a state value, and a list of neighbors.
- **Network Class**: Represents the network topology, allowing the addition of agents and bidirectional links between them.
- **ConsensusSimulation Class**: Manages the distributed consensus simulation, performing rounds of communication and updates until convergence.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/JacquesGariepy/distributed-consensus-multi-agent-system.git
    cd distributed-consensus-multi-agent-system
    ```

2. Ensure you have Python 3.6+ installed.


## Usage

### Running the Simulation

You can run the standalone example provided in `consensus.py`:

```sh
python consensus.py
```

This will create a simple network, run the simulation, and display the final result.

### Running Unit Tests

Unit tests are provided to verify the functionality of the module. To run the tests, use:

```sh
python -m unittest discover
```

## Project Structure

- `consensus.py`: Contains the implementation of the Agent, Network, and ConsensusSimulation classes.
- `consensus_test.py`: Contains unit tests for the consensus module.
- `README.md`: This file.

## Example

Here is an example of how to create a network, add agents, and run the consensus simulation:

```python
from consensus import Network, ConsensusSimulation

# Create a network
net = Network()

# Add agents
net.add_agent(0, 10.0)
net.add_agent(1, 0.0)
net.add_agent(2, 20.0)
net.add_agent(3, 30.0)

# Add links
net.add_edge(0, 1)
net.add_edge(1, 2)
net.add_edge(2, 3)

# Run the simulation
sim = ConsensusSimulation(net, step_size=0.5)
rounds_done = sim.run(max_rounds=1000, epsilon=1e-4)

# Print results
print(f"Simulation completed in {rounds_done} rounds.")
print("Final values:", net.get_all_values())
```

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## Contact

For any questions or inquiries, please contact [yourname@example.com].
