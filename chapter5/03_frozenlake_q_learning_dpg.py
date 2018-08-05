import gym
import collections
from tensorboardX import SummaryWriter

ENV_NAME = "FrozenLake-v0"
#ENV_NAME = "FrozenLake8x8-v0"
GAMMA = 0.9
TEST_EPISODES = 20

# Small change to the original one. But instead of setting the reward the reward is also the average.
class Agent:

    def __init__(self):
        self.env = gym.make(ENV_NAME)
        self.state = self.env.reset()
        self.rewards = collections.defaultdict(float)
        self.transits = collections.defaultdict(collections.Counter)
        self.values = collections.defaultdict(float)

    # Same as v learning to explore the environment
    def play_n_random_steps(self, count):
        for _ in range(count):
            action = self.env.action_space.sample()
            new_state, reward, is_done, _ = self.env.step(action)
            self.rewards[(self.state, action, new_state)] += reward
            self.transits[(self.state, action)][new_state] += 1
            self.state = self.env.reset() if is_done else new_state

    # def calc_action_value(self, state, action):
    #     target_counts = self.transits[(state, action)]
    #     total = sum(target_counts.values())
    #     action_value = 0.0
    #     for tgt_state, count in target_counts.items():
    #         reward = self.rewards[(state, action, tgt_state)]
    #         action_value += (count/total) * (reward + GAMMA * self.values[tgt_state])
    #     return action_value

    # Very similar to v learning but in 'values' we already have the s,a value so no need to calculate the value
    # for each action in each state is already there. We have to just select the best action based on Q
    def select_action(self, state):
        best_action, best_value = None, None
        for action in range(self.env.action_space.n):
            # action_value = self.calc_action_value(state, action)
            action_value = self.values[(state, action)]
            if best_value is None or best_value < action_value:
                best_value = action_value
                best_action = action
        return best_action

    # Same as va learning
    def play_episode(self, env):
        total_reward = 0.0
        state = env.reset()
        while True:
            action = self.select_action(state)
            new_state, reward, is_done, _ = env.step(action)
            self.rewards[(state, action, new_state)] = +reward
            self.transits[(state, action)][new_state] += 1
            total_reward += reward
            if is_done:
                break
            state = new_state
        return total_reward

    # Value iteraction is a little bit more complex than v learning because in here we calculate the value of Q
    # and not just V. So we have to iterate over each state and action and calculate the value of (s,a)
    # taking into consideration the probabilities of transitions and the rewards of those transitions
    def value_iteration(self):
        for state in range(self.env.observation_space.n):
            for action in range(self.env.action_space.n):
                action_value = 0.0
                target_counts = self.transits[(state, action)]
                total = sum(target_counts.values())
                for tgt_state, count in target_counts.items():
                    add_reward = self.rewards[(state, action, tgt_state)]
                    # We calculate the reward as the average of the rewards in case the r(s,a,s1) is not constant
                    reward = add_reward / count
                    best_action = self.select_action(tgt_state)
                    action_value += (count/total) * (reward + GAMMA * self.values[(tgt_state, best_action)])

                self.values[(state, action)] = action_value


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
            print("Solved in %d iterations!" % iter_no)
            break

    writer.close()
