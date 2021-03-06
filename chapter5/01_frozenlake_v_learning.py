import gym
import collections
from tensorboardX import SummaryWriter

#ENV_NAME = "FrozenLake-v0"
ENV_NAME = "FrozenLake8x8-v0"
GAMMA = 0.9
TEST_EPISODES = 20


class Agent:

    def __init__(self):
        self.env = gym.make(ENV_NAME)
        self.state = self.env.reset()
        self.rewards = collections.defaultdict(float)
        self.transits = collections.defaultdict(collections.Counter)
        self.values = collections.defaultdict(float)

# This method is used to explore "understand" the environment
    def play_n_random_steps(self, count):
        for _ in range(count):
            action = self.env.action_space.sample()
            new_state, reward, is_done, _ = self.env.step(action)
            self.rewards[(self.state, action, new_state)] = reward
            self.transits[(self.state, action)][new_state] += 1
            self.state = self.env.reset() if is_done else new_state

# The methods below are used to help on applying the policy to on every given state select
    # the action with highest reward

# For a given state and action calculates the value of applying that action in that state.
# It gets the value of s0, a, s1 transition + the discounted value of s1 based on the probability
    # of ending up on that state
    def calc_action_value(self, state, action):
        target_counts = self.transits[(state, action)]
        total = sum(target_counts.values())
        action_value = 0.0
        for tgt_state, count in target_counts.items():
            reward = self.rewards[(state, action, tgt_state)]
            action_value += (count/total) * (reward + GAMMA * self.values[tgt_state])
        return action_value

# Iterates over all the actions in the state and selects the action with the highest value
    def select_action(self, state):
        best_action, best_value = None, None
        for action in range(self.env.action_space.n):
            action_value = self.calc_action_value(state, action)
            if best_value is None or best_value < action_value:
                best_value = action_value
                best_action = action
        return best_action

# Plays an episode selecting the best action for every given state based on the Value
    def play_episode(self, env):
        total_reward = 0.0
        state = env.reset()
        while True:
            action = self.select_action(state)
            new_state, reward, is_done, _ = env.step(action)
            self.rewards[(state, action, new_state)] = reward
            self.transits[(state, action)][new_state] += 1
            total_reward += reward
            if is_done:
                break
            state = new_state
        return total_reward

    # Calculates the value for each state.
    # The value for each state is the value of of the best action possible from this state.
    # The value of course it will be the transition plus the discounted reward of the target state
    def value_iteration(self):
        for state in range(self.env.observation_space.n):
            state_values = [self.calc_action_value(state, action) for action in range(self.env.action_space.n)]
            self.values[state] = max(state_values)


if __name__ == "__main__":
    test_env = gym.make(ENV_NAME)
    agent = Agent()
    writer = SummaryWriter(comment="-v learning")

    iter_no = 0
    best_reward = 0.0
    while True:
        iter_no += 1
        agent.play_n_random_steps(100)
        agent.value_iteration()

        episode_rewards = 0.0
        for _ in range(TEST_EPISODES):
            episode_rewards += agent.play_episode(test_env)

        episode_rewards /= TEST_EPISODES
        writer.add_scalar("reward", episode_rewards, iter_no)
        if episode_rewards > best_reward:
            print("Best reward updated %.3f -> %.3f" % (best_reward, episode_rewards))
            best_reward = episode_rewards
        if episode_rewards > 0.80:
            print("Solved in % iterations!" % iter_no)
            break

    writer.close()
