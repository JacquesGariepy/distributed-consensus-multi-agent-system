#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_consensus.py

Unit tests for the consensus.py module.
Verifies that the required behaviors (adding agents, links,
consensus simulation) work correctly, including under extreme conditions.
"""

import unittest
from consensus import Agent, Network, ConsensusSimulation

class TestAgent(unittest.TestCase):
    def test_agent_creation(self):
        ag = Agent(agent_id=5, value=10.0)
        self.assertEqual(ag.agent_id, 5)
        self.assertEqual(ag.value, 10.0)
        self.assertEqual(ag.neighbors, [])

    def test_agent_update_value(self):
        ag = Agent(agent_id=10, value=0.0)
        ag.update_value(3.14)
        self.assertAlmostEqual(ag.value, 3.14, places=5)

class TestNetwork(unittest.TestCase):
    def setUp(self):
        self.network = Network()

    def test_add_agent(self):
        self.network.add_agent(agent_id=1, initial_value=5.0)
        self.assertIn(1, self.network.agents)
        self.assertEqual(self.network.agents[1].value, 5.0)

    def test_add_agent_duplicate(self):
        self.network.add_agent(agent_id=1, initial_value=5.0)
        with self.assertRaises(ValueError):
            self.network.add_agent(agent_id=1, initial_value=10.0)

    def test_add_edge(self):
        self.network.add_agent(0, 10.0)
        self.network.add_agent(1, 0.0)
        self.network.add_edge(0, 1)
        self.assertIn(1, self.network.agents[0].neighbors)
        self.assertIn(0, self.network.agents[1].neighbors)

    def test_add_edge_invalid(self):
        self.network.add_agent(0, 10.0)
        with self.assertRaises(KeyError):
            self.network.add_edge(0, 5)  # 5 does not exist

    def test_get_all_values(self):
        self.network.add_agent(0, 1.2)
        self.network.add_agent(1, 3.4)
        vals = self.network.get_all_values()
        self.assertEqual(vals[0], 1.2)
        self.assertEqual(vals[1], 3.4)

class TestConsensusSimulation(unittest.TestCase):
    def setUp(self):
        self.net = Network()
        for i in range(4):
            self.net.add_agent(i, float(i*10))  # 0, 10, 20, 30
        self.net.add_edge(0, 1)
        self.net.add_edge(1, 2)
        self.net.add_edge(2, 3)

    def test_init_invalid_step_size(self):
        with self.assertRaises(ValueError):
            ConsensusSimulation(self.net, step_size=1.5)

    def test_run_basic(self):
        sim = ConsensusSimulation(self.net, step_size=0.5)
        self.assertEqual(sim.current_round, 0)
        rounds_done = sim.run(max_rounds=100, epsilon=1e-3)
        # Verify that we did less than 100 rounds (expect to converge before)
        self.assertLess(rounds_done, 100)

    def test_convergence_small_network(self):
        # Line network: 4 agents => the average value is (0 + 10 + 20 + 30)/4 = 15
        sim = ConsensusSimulation(self.net, step_size=0.5)
        sim.run(max_rounds=200, epsilon=1e-5)
        final_vals = [ag.value for ag in self.net.agents.values()]
        for val in final_vals:
            self.assertAlmostEqual(val, 15.0, delta=0.5)  # tolerance of 0.5

    def test_isolated_agent(self):
        # Add an isolated agent (no neighbors). It will not move.
        self.net.add_agent(99, 999.0)
        sim = ConsensusSimulation(self.net, step_size=0.5)
        sim.run(max_rounds=50, epsilon=1e-5)
        val_isolated = self.net.agents[99].value
        self.assertEqual(val_isolated, 999.0)  # No change

    def test_has_converged_no_agents(self):
        empty_net = Network()
        sim = ConsensusSimulation(empty_net)
        sim.run()
        self.assertTrue(True, "No agents => trivial convergence, no error.")

    def test_has_converged_single_agent(self):
        single_net = Network()
        single_net.add_agent(0, 42.0)
        sim = ConsensusSimulation(single_net)
        sim.run()
        self.assertAlmostEqual(single_net.agents[0].value, 42.0)
        # single agent => already converged

if __name__ == '__main__':
    unittest.main()
