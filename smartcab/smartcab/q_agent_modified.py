import random
import numpy as np
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import sys
import os


class QLearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    args = []
    for arg in sys.argv[1:]:
        args.append(arg)
    args = [float(x) for x in args]

    def __init__(self, env, epsilon=args[0], alpha=args[1], gamma=args[2], init_q_value = args[3]):
        super(QLearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.next_waypoint = None
        self.total_reward = 0
        self.moves = 0
        self.state = None
        self.new_state = None

        self.init_q_value = init_q_value
        self.epsilon = epsilon  ## probability of doing random move
        self.alpha = alpha  ## learning rate
        self.gamma = gamma

        self.valid_actions = Environment.valid_actions
        self.q_dict = dict()

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.next_waypoint = None
        self.total_reward = 0
        self.moves = 0
        self.state = None
        self.new_state = None

    def get_q(self,state,action):
        # if key is not present in dictionary, self.init_q_value is returned
        return self.q_dict.get((state, action), self.init_q_value)

    def q_learn(self, state, action, next_state, reward):
        ## use q-learning algorithm to update q values
        if (state, action) not in self.q_dict:
            self.q_dict[(state, action)] = self.get_q(state,action)
        else:
            q_next = [self.get_q(next_state, a) for a in self.valid_actions]
            max_q_next = max(q_next)
            self.q_dict[(state, action)] = self.q_dict[(state, action)] +\
                                           self.alpha*(reward + self.gamma*max_q_next - self.q_dict[(state, action)])

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
        ## epsilon greedy approach: epsilon represents a chance of having a random  move (without violating rules)
        if random.random() < self.epsilon:
            action = random.choice(self.valid_actions)

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

        else:  ## choose the best action:
            q = [self.get_q(self.state, a) for a in self.valid_actions]
            max_q = max(q)
            if q.count(max_q) > 1:
                ## if we have multiple actions which correspond to a maximal q, we pick an action randomly:
                best_actions = [i for i in range(len(self.valid_actions)) if q[i] == max_q]
                action_idx = random.choice(best_actions)
            else:
                action_idx = q.index(max_q)
            action = self.valid_actions[action_idx]

        # Execute action and get reward
        reward = self.env.act(self, action)
        self.total_reward += reward
        self.moves = self.moves + 1

        # TODO: Learn policy based on state, action, reward
        next_inputs = self.env.sense(self)
        self.next_state = (next_inputs['light'], next_inputs['oncoming'], next_inputs['left'], next_inputs['right'], self.next_waypoint)
        self.q_learn(self.state, action, self.next_state, reward)

        ## print out the results
        if (reward > 8) or (deadline == 0):
            print "LearningAgent.update(): total_reward = {}, total_moves = {}, location = {}, destination = {}".\
                format(self.total_reward, self.moves, location, destination)  # debug

def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(QLearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.0001, display=False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line

try:
    os.remove("reports/q_agent_"+str(QLearningAgent(Agent).__dict__.get('epsilon'))+"_" +
              str(QLearningAgent(Agent).__dict__.get('alpha'))+"_"+str(QLearningAgent(Agent).__dict__.get('gamma'))+"_" +
              str(QLearningAgent(Agent).__dict__.get('init_q_value'))+".txt")
except:
    pass

sys.stdout = open("reports/q_agent_"+str(QLearningAgent(Agent).__dict__.get('epsilon'))+"_" +
                  str(QLearningAgent(Agent).__dict__.get('alpha'))+"_"+str(QLearningAgent(Agent).__dict__.get('gamma'))+"_" +
                  str(QLearningAgent(Agent).__dict__.get('init_q_value'))+".txt", "w")

if __name__ == '__main__':
    run()

sys.stdout.close()
