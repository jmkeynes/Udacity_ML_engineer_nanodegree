import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class QLearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env, epsilon=.1, alpha=.5, gamma=.7):
        super(QLearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.next_waypoint = None
        self.total_reward = 0
        self.moves = 0
        self.state = None
        self.new_state = None

        self.init_q_value = 15
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

    def pick_q_action(self, state):
        ## epsilon greedy approach: epsilon represents a chance of having a random move
        if random.random() < self.epsilon:
            action = random.choice(self.valid_actions)
        else:  ## choose the best action:
            q = [get_q(state, a) for a in self.valid_actions]
            max_q = max(q)
            if q.count(max_q) > 1:
                ## if we have multiple actions which correspond to a maximal q, we pick an action randomly:
                best_actions = [i for i in range(len(self.valid_actions)) if q[i] == max_q]
                action_idx = random.choice(best_actions)
            else:
                action_idx = q.index(max_q)
            action = self.valid_actions[action_idx]
        return action

    def q_learn(self, state, action, next_state, reward):
        ## use q-learning algorithm to update q values
        if (state, action) not in self.q_dict:
            self.q_dict[(state, action)] = get_q(state,action)
        else:
            q_next = [get_q(next_state, a) for a in self.valid_actions]
            max_q_next = max(q_next)
            self.q_dict[(state, action)] = self.q_dict[(state, action)] +\
                                           self.alpha*(reward + self.gamma*max_q_next - self.q_dict[(state, action)])

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        self.state = (inputs['light'], inputs['oncoming'], inputs['left'], inputs['right'], self.next_waypoint)
        
        # TODO: Select action according to your policy
        action = self.pick_q_action(self.state)

        # Execute action and get reward
        reward = self.env.act(self, action)

        # TODO: Learn policy based on state, action, reward
        next_inputs = self.env.sense(self)
        next_inputs = next_inputs.items()
        next_state = (next_inputs['light'], next_inputs['oncoming'], next_inputs['left'], next_inputs['right'], self.next_waypoint)

        self.q_learn(state, action, next_state, reward)

        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.0001, display=False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line

try:
    os.remove("reports/first_q_agent_report.txt")
except:
    pass

sys.stdout = open("reports/first_q_agent_report.txt","w")

if __name__ == '__main__':
    run()

sys.stdout.close()
