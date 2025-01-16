#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
consensus.py

This module implements a distributed consensus simulation in a
multi-agent system. Each agent attempts to reach a common value
(the global average) through repeated exchanges with its neighbors.
"""

import math
from typing import List, Dict, Optional

class Agent:
    """
    Represents an individual agent in the network.
    Each agent has an identifier, a state value,
    and a list of neighbors (communication links).

    Attributes:
        agent_id (int)  : the unique identifier of the agent
        value   (float) : the current value of the agent (e.g., temperature, score, etc.)
        neighbors (List[int]) : the list of neighbor IDs for communication
    """

    def __init__(self, agent_id: int, value: float):
        """
        Initializes an agent with an ID and a floating-point value.

        :param agent_id: unique identifier
        :param value: initial value
        """
        self.agent_id: int = agent_id
        self.value: float = value
        self.neighbors: List[int] = []

    def update_value(self, new_value: float):
        """
        Updates the agent's value.
        :param new_value: new value
        """
        self.value = new_value


class Network:
    """
    Represents the network topology (list of agents and connections).
    """

    def __init__(self):
        """
        Initializes an empty network (no agents).
        """
        self.agents: Dict[int, Agent] = {}

    def add_agent(self, agent_id: int, initial_value: float):
        """
        Adds an agent to the network.

        :param agent_id: unique identifier
        :param initial_value: initial value
        :raises ValueError: if an agent with the same ID already exists
        """
        if agent_id in self.agents:
            raise ValueError(f"Agent ID {agent_id} already exists.")
        self.agents[agent_id] = Agent(agent_id, initial_value)

    def add_edge(self, agent_id_a: int, agent_id_b: int):
        """
        Adds a bidirectional link between two agents.

        :param agent_id_a: first agent
        :param agent_id_b: second agent
        :raises KeyError: if one of the agents does not exist
        """
        if agent_id_a not in self.agents or agent_id_b not in self.agents:
            raise KeyError("One or both agent IDs not found in the network.")
        self.agents[agent_id_a].neighbors.append(agent_id_b)
        self.agents[agent_id_b].neighbors.append(agent_id_a)

    def get_all_values(self) -> Dict[int, float]:
        """
        Returns a dict {agent_id: value} for consultation/display.
        """
        return {aid: ag.value for aid, ag in self.agents.items()}


class ConsensusSimulation:
    """
    Manages the distributed consensus simulation.
    """

    def __init__(self, network: Network, step_size: float = 0.5):
        """
        :param network: instance of Network containing agents and links
        :param step_size: alpha coefficient for the update (0 < alpha <= 1)
        """
        self.network = network
        if not (0.0 < step_size <= 1.0):
            raise ValueError("step_size must be in (0,1].")
        self.step_size: float = step_size
        self.current_round: int = 0
        self._previous_values: Dict[int, float] = {}

    def initialize_previous_values(self):
        """
        Saves the initial configuration for convergence detection.
        """
        self._previous_values = {aid: ag.value for aid, ag in self.network.agents.items()}

    def step(self):
        """
        Performs a round of communication and update.
        Each agent calculates the average of its neighbors' values and its own,
        then takes a step towards this average.
        """
        new_values = {}
        for aid, ag in self.network.agents.items():
            if not ag.neighbors:
                # No neighbors => the agent keeps its value
                new_values[aid] = ag.value
                continue

            neighbor_values = [self.network.agents[n].value for n in ag.neighbors]
            average_local = (ag.value + sum(neighbor_values)) / (1 + len(neighbor_values))
            # Update by a simple gradient
            updated_value = ag.value + self.step_size * (average_local - ag.value)
            new_values[aid] = updated_value

        # Apply the new values to all agents
        for aid in new_values:
            self.network.agents[aid].update_value(new_values[aid])

        self.current_round += 1

    def has_converged(self, epsilon: float = 1e-3) -> bool:
        """
        Checks if the values have converged (very small change or tight distribution).

        :param epsilon: tolerance to judge convergence
        :return: True if the system is considered converged
        """
        # 1) Check if all values are close (max-min difference)
        all_values = [ag.value for ag in self.network.agents.values()]
        if not all_values:
            return True  # empty => trivial convergence
        min_val, max_val = min(all_values), max(all_values)
        if (max_val - min_val) < epsilon:
            return True

        # 2) Optionally: check the difference from the previous iteration
        # (to avoid oscillations)
        changed = 0
        for aid, ag in self.network.agents.items():
            old_v = self._previous_values.get(aid, ag.value)
            if abs(ag.value - old_v) > epsilon:
                changed += 1
        self._previous_values = {aid: ag.value for aid, ag in self.network.agents.items()}
        # Consider converged if less than 20% of agents have moved
        if len(self.network.agents) > 0:
            ratio = changed / len(self.network.agents)
            if ratio < 0.2:
                return True

        return False

    def run(self, max_rounds: int = 1000, epsilon: float = 1e-3) -> int:
        """
        Runs the simulation until convergence or the end of iterations.

        :param max_rounds: max number of iterations
        :param epsilon: tolerance threshold
        :return: the number of rounds executed before stopping
        """
        self.initialize_previous_values()
        for _ in range(max_rounds):
            self.step()
            if self.has_converged(epsilon=epsilon):
                break
        return self.current_round


def main_example():
    """
    Standalone usage example: creates a simple network, runs the simulation
    and displays the final result.
    """
    net = Network()
    # Adding 4 agents
    net.add_agent(0, 10.0)
    net.add_agent(1, 0.0)
    net.add_agent(2, 20.0)
    net.add_agent(3, 30.0)

    # Links
    net.add_edge(0, 1)
    net.add_edge(1, 2)
    net.add_edge(2, 3)

    sim = ConsensusSimulation(net, step_size=0.5)
    print("[DEBUG] Initial values:", net.get_all_values())

    rounds_done = sim.run(max_rounds=1000, epsilon=1e-4)
    print(f"Simulation completed in {rounds_done} rounds.")
    print("Final values:", net.get_all_values())


if __name__ == "__main__":
    main_example()
