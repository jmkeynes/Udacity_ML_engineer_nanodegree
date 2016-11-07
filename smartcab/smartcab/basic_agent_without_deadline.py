import random
import numpy as np
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import sys
import os

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.next_waypoint = None
        self.total_reward = 0
        self.moves = 0
        self.state = None

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.next_waypoint = None
        self.total_reward = 0
        self.moves = 0
        self.state = None

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)
        location = self.env.agent_states[self]['location']
        destination = self.env.agent_states[self]['destination']

        # TODO: Update state
        self.state = (inputs['light'], inputs['oncoming'], inputs['left'], inputs['right'], self.next_waypoint)

        # TODO: Select action according to your policy
        action = random.choice(Environment.valid_actions)

        action_okay = True
        if action == 'right':
            if inputs['light'] == 'red' and inputs['left'] == 'forward':
                action_okay = False
        elif action == 'forward':
            if inputs['light'] == 'red':
                action_okay = False
        elif action == 'left':
            if inputs['light'] == 'red' or (inputs['oncoming'] == 'forward' or inputs['oncoming'] == 'right'):
                action_okay = False
        if not action_okay:
            action = None

        # Execute action and get reward
        reward = self.env.act(self, action)
        self.total_reward += reward
        self.moves = self.moves + 1


        # TODO: Learn policy based on state, action, reward
        ## no policy for basic agent

        ## print out the results
        if (reward > 8) or (deadline == 0):
            print "LearningAgent.update(): total_reward = {}, total_moves = {}, location = {}, destination = {}".\
                format(self.total_reward, self.moves, location, destination)  # debug


def run():
    """Run the agent for a finite number of trials."""
    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=False)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.0001, display=False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line

try:
    os.remove("reports/basic_agent_without_deadline_report.txt")
except:
    pass

sys.stdout = open("reports/basic_agent_without_deadline_report.txt","w")

if __name__ == '__main__':
    run()

sys.stdout.close()
