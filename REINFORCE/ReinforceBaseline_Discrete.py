import numpy as np
import tensorflow as tf
import gym
import pandas as pd
import matplotlib.pyplot as plt
from os import makedirs
from os.path import isdir

# Configuration parameters for the whole setup
GYM_ENVIRONMENT = "LunarLander-v2"

GAMMA = 0.98  # Discount factor for past returns (G_t)
LEARNING_RATE = 1e-4
HIDDEN_POLICY = [256, 128]
HIDDEN_BASELINE = [256, 128]

EPISODES = 15
STEPS_PER_EPISODE = 400
RENDER_AFTER = 0

MODEL_SAVING_DIR = "models"
MODEL_LOADING_DIR = "models/" + GYM_ENVIRONMENT + "/reinforce_bl_256x128_256x128/450_steps_per_episode/5200_episodes"
SAVE_EVERY = 500    # save the model every N steps


class Agent:
    def __init__(self, state_size, actions_size, lr=0.01, hidden_policy=HIDDEN_POLICY, hidden_baseline=HIDDEN_BASELINE, gamma=0.95,
                 render_after_n_episodes=50, loading_dir=None):
        self.state_size = state_size
        self.n_actions = actions_size
        self.gamma = gamma
        self.render_after_n_steps = render_after_n_episodes

        self.optimizer = tf.keras.optimizers.Adam(lr=lr)

        ''' Load or Build the networks '''
        if loading_dir is not None:
            self.policy_net, self.baseline_net = self.load_models(loading_dir)
            print("\nModels loaded from: " + loading_dir + "\n")
        else:
            ''' POLICY NETWORK '''
            self.policy_net = tf.keras.models.Sequential([
                tf.keras.layers.Dense(hidden_policy[0], activation='relu', input_shape=[state_size])
            ])
            for units in hidden_policy[1:]:
                self.policy_net.add(tf.keras.layers.Dense(units, activation='relu'))
            self.policy_net.add(tf.keras.layers.Dense(actions_size, activation='softmax'))

            ''' BASELINE NETWORK '''
            self.baseline_net = tf.keras.models.Sequential([
                tf.keras.layers.Dense(hidden_baseline[0], activation='relu', input_shape=[state_size])
            ])
            for units in hidden_baseline[1:]:
                self.baseline_net.add(tf.keras.layers.Dense(units, activation='relu'))
            self.baseline_net.add(tf.keras.layers.Dense(1))

    def play_episode_and_train(self, env, episode, max_steps_per_episode=200):
        state = env.reset()
        episode_reward = 0
        action_probs_history = []
        baseline_history = []
        rewards_history = []
        last_step = 0

        with tf.GradientTape(persistent=True) as tape:
            """ Generation of the episode's steps """
            for t in range(max_steps_per_episode):
                last_step += 1
                if episode >= self.render_after_n_steps:
                    env.render()

                state = tf.convert_to_tensor(state, dtype=np.float32)
                state = tf.expand_dims(state, 0)

                action_probs = self.policy_net(state)       # these are π(A_t | S_t, theta) in the reinforce update formula
                baseline = self.baseline_net(state)         # these are v^(S_t, w) in the formula
                baseline_history.append(baseline[0][0])

                # sample action from action probability distribution
                action = np.random.choice(num_actions, p=np.reshape(action_probs, self.n_actions))
                action_probs_history.append(tf.math.log(action_probs[0][action]))   # list of ln(π(A_t | S_t, theta)) for each time step in the episode

                # take the chosen action in the environment
                state, reward, done, info = env.step(action)
                rewards_history.append(reward)                                      # list of the rewards R_t+1 for each time step in the episode
                episode_reward += reward

                if done or episode_reward >= 200.0:
                    break

            """ REINFORCE update """
            G_t = 0
            returns = []
            for r in rewards_history[::-1]:     # with [::-1] we obtain the reverse (full) list
                G_t = self.gamma * G_t + r   # for each time step t, this is the total (discounted) reward received after that step
                returns.append(G_t)
            returns.reverse()

            # computing the loss values '''
            policy_losses = []
            baseline_losses = []
            for ln_pi, state_value, ret in zip(action_probs_history, baseline_history, returns):
                policy_loss = - (ret - state_value) * ln_pi
                policy_losses.append(policy_loss)
                baseline_loss = 0.5 * (ret - state_value)**2  # squared error
                baseline_losses.append(baseline_loss)

            total_policy_loss = sum(policy_losses)
            total_baseline_loss = sum(baseline_losses)

        """ Backpropagation """
        grads_policy = tape.gradient(total_policy_loss, self.policy_net.trainable_variables)
        grads_baseline = tape.gradient(total_baseline_loss, self.baseline_net.trainable_variables)
        self.optimizer.apply_gradients(zip(grads_policy, self.policy_net.trainable_variables))
        self.optimizer.apply_gradients(zip(grads_baseline, self.baseline_net.trainable_variables))
        return last_step, episode_reward

    def play_episode_no_train(self, env, episode, max_steps_per_episode=200):
        pass

    def folder_path(self, dir_path):
        path = dir_path + "/" + GYM_ENVIRONMENT + "/reinforce_bl_"
        for layer in self.policy_net.layers[:-1]:
            path += str(layer.units) + "x"
        path = path[:-1] + "_"
        for layer in self.baseline_net.layers[:-1]:
            path += str(layer.units) + "x"
        path = path[:-1] + "/" + str(STEPS_PER_EPISODE) + "_steps_per_episode"
        return path

    def save_models(self, dir_path, episodes):
        path = self.folder_path(dir_path)
        path += "/" + str(episodes) + "_episodes/"
        if not isdir(path):
            makedirs(path, exist_ok=True)
        self.policy_net.save(path + "policy_network.h5")
        self.baseline_net.save(path + "baseline_network.h5")
        print("-" * 25 + " Models successfully saved " + "-" * 25)

    @staticmethod
    def load_models(load_dir_path):
        policy = tf.keras.models.load_model(load_dir_path + "/policy_network.h5")
        baseline = tf.keras.models.load_model(load_dir_path + "/baseline_network.h5")
        return policy, baseline


""" #################################
    #                               # 
    #           M A I N             #
    #                               #
    #################################
"""

env = gym.make(GYM_ENVIRONMENT)  # Create the gym environment

num_inputs = env.observation_space.shape[0]
num_actions = env.action_space.n

df_data = []
agent = Agent(num_inputs, num_actions, gamma=GAMMA, lr=LEARNING_RATE,
              hidden_policy=HIDDEN_POLICY, hidden_baseline=HIDDEN_BASELINE,
              loading_dir=MODEL_LOADING_DIR, render_after_n_episodes=RENDER_AFTER)

for ep in range(EPISODES):
    steps, ep_reward = agent.play_episode_and_train(env, ep, STEPS_PER_EPISODE)
    print("Episode " + str(ep))
    print("\t steps: {};\treward: {}".format(steps, ep_reward))
    df_data.append([steps, ep_reward])
    if (ep + 1) % SAVE_EVERY == 0:
        agent.save_models(MODEL_SAVING_DIR, ep+1)

df = pd.DataFrame(df_data, columns=['steps', 'reward'])
csv_dir = agent.folder_path(MODEL_SAVING_DIR)
if not isdir(csv_dir):
    makedirs(csv_dir, exist_ok=True)
df.to_csv(agent.folder_path(MODEL_SAVING_DIR)+"/summary.csv")

x = df.index.values
m, b = np.polyfit(x, df['reward'], 1)
df.plot(y='reward', kind='line', figsize=(12, 12), title='Reward curve')
plt.plot(x, (m * x + b), '--')
plt.ylim((-100, 400))
plt.grid()
plt.show()
