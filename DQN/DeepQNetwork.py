import numpy as np
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.optimizers import Adam
import gym
from collections import deque
import pandas as pd
import matplotlib.pyplot as plt

GYM_ENVIRONMENT = 'CartPole-v1'

BUFFER_SIZE = 5000
BATCH_SIZE = 32

MODEL_LOADING_DIR = None
LEARNING_RATE = 0.001
HIDDEN_UNITS = [32, 32]

GAMMA = 0.95
EPSILON_MAX = 0.95
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.00005

EPISODES = 1500
STEPS_PER_EPISODE = 200
PRETRAIN_STEPS = BATCH_SIZE * 2
RENDER_AFTER = 1400


class ReplayBuffer:
    def __init__(self, size=10000):
        self.memory = deque(maxlen=size)

    def remember(self, experience):     # experience is a tuple  (s, a, r, s', done)
        self.memory.append(experience)

    def sample(self, batch_size=32):
        indexes = np.random.randint(len(self.memory), size=batch_size)
        return [self.memory[idx] for idx in indexes]


class Agent:
    '''
    state_size: (e.g. 4 in CartPole-v1)
    action_size: number of possible actions (e.g. 2 in CartPole-v1)
    '''
    def __init__(self, state_size, actions_size, lr=0.01, hidden=[32, 32],
                 gamma=0.95, epsilon_i=0.9, epsilon_f=0.01, epsilon_dec=0.005, model_dir=MODEL_LOADING_DIR):
        self.state_size = state_size
        self.num_actions = actions_size
        self.gamma = gamma
        self.epsilon_i = epsilon_i          # initial value for epsilon
        self.epsilon_dec = epsilon_dec      # epsilon decay parameter
        self.epsilon_f = epsilon_f          # minimum epsilon value
        self.epsilon = epsilon_i            # current epsilon value

        self.model = Sequential()
        if model_dir is not None:
            self.model = self.load_model(model_dir)
        else:
            self.model.add(Dense(hidden[0], activation='relu', input_dim=state_size))
            for units in hidden[1:]:
                self.model.add(Dense(units, activation='relu'))
            self.model.add(Dense(actions_size, activation='linear'))

        self.optimizer = Adam(lr=lr)
        self.model.compile(loss='mse', optimizer=self.optimizer)

    def decay_epsilon(self, n_steps):
        self.epsilon = self.epsilon_f + \
                       (self.epsilon_i - self.epsilon_f) * np.exp(-self.epsilon_dec * n_steps)

    def epsilon_greedy_selection(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions)
        else:
            Q_values = self.model.predict(state)[0]
            return np.argmax(Q_values)

    ''' replay '''
    def update_network(self, batch):
        inputs = np.zeros((len(batch), self.state_size))
        targets = np.zeros((len(batch), self.num_actions))

        ''' this workss very fast: the model works on the entire batch of states at once'''
        #_states, _actions, _rewards, _next_states, _dones = [np.array([experience[field_index] for experience in batch])
        #                                                     for field_index in range(5)]
        #_states = np.reshape(_states, [len(batch), self.state_size])
        #_next_states = np.reshape(_next_states, [len(batch), self.state_size])
        #
        #_target_Qs = self.model.predict(_next_states)
        #_targets = self.model.predict(_states)
        #for i in range(len(_dones)):
        #    _inputs[i:i+1] = _states[i]
        #    _new_target = _rewards[i]
        #    if not _dones[i]:
        #        _new_target += self.gamma * np.amax(_target_Qs[i])
        #    _targets[i][_actions[i]] = _new_target
        #self.model.fit(_inputs, _targets, epochs=1, verbose=0)

        ''' this works, but more slowly : the network is called once per batch element '''
        for i, (state, action, reward, next_state, done) in enumerate(batch):
            inputs[i:i+1] = state
            target = reward
            if not done:
                target_Q = self.model.predict(next_state)[0]
                target += self.gamma * np.amax(target_Q)
            targets[i] = self.model.predict(state)
            targets[i][action] = target
        self.model.fit(inputs, targets, epochs=1, verbose=0)
        ''' ******* '''

    '''
        Pretrain to fill the replay memory 
    
            env: gym environment
            buffer: ReplayBuffer object
            pretrain_steps: number of experiences to collect
            
            At the end, the environment must be reset in order to use it properly for the training phase
    '''
    def generate_experiences(self, env, buffer, pretrain_steps):
        state = env.reset()
        state = np.reshape(state, [1, self.state_size])
        print("=" * 50)
        print("COLLECTING EXPERIENCES...")

        for _ in range(pretrain_steps):
            action = env.action_space.sample()
            next_state, reward, done, info = env.step(action)

            if not done:
                next_state = np.reshape(next_state, [1, self.state_size])
                buffer.remember((state, action, reward, next_state, done))
            else:
                next_state = np.zeros(state.shape)  # failed simulation: set s'=[ 0.0 0.0 0.0 0.0 ]
                buffer.remember((state, action, reward, next_state, done))
                next_state = env.reset()
                next_state = np.reshape(next_state, [1, self.state_size])
            state = next_state

        print("Done.")

    ''' 
        Train the network using past experiences from the replay buffer
        
    '''
    def play(self, env, buffer, batch_size=32, episodes=1000, steps_per_episode=200,
             render_after_n_episodes=100):
        print("=" * 50)
        print("PLAYING {} EPISODES\n".format(episodes))
        total_steps = 0
        state = env.reset()
        state = np.reshape(state, [1, self.state_size])
        df_data = []

        for episode in range(episodes):
            episode_reward = 0
            last_step = steps_per_episode
            print("Episode {}".format(episode+1))

            for t in range(steps_per_episode):
                total_steps += 1

                if episode+1 >= render_after_n_episodes:
                    env.render()

                ''' agent's on-policy action '''
                self.decay_epsilon(total_steps)
                action = self.epsilon_greedy_selection(state)
                next_state, reward, done, info = env.step(action)
                next_state = np.reshape(next_state, [1, self.state_size])
                episode_reward += reward

                if not done:
                    buffer.remember((state, action, reward, next_state, done))
                    state = next_state
                else:
                    last_step = t
                    next_state = np.zeros(state.shape)
                    buffer.remember((state, action, reward, next_state, done))
                    state = env.reset()
                    state = np.reshape(state, [1, self.state_size])

                batch = buffer.sample(batch_size)
                self.update_network(batch)
                if done:
                    break

            print("\t steps: {}; episode reward: {}".format(last_step+1, episode_reward))
            print("\t last epsilon: {}\n".format(self.epsilon))
            df_data.append([last_step+1, episode_reward])
        return pd.DataFrame(df_data, columns=['steps', 'reward'])


    def save_model(self, dir_path):
        path = dir_path + "/" + GYM_ENVIRONMENT + "_"
        for h in HIDDEN_UNITS:
            path += str(h) + "x"
        path = path[:-1]
        #path += "_" + str(EPISODES) +"eps"
        path += ".h5"
        self.model.save(path)

    @staticmethod
    def load_model(dir_path):
        path = dir_path + "/" + GYM_ENVIRONMENT + "_"
        for h in HIDDEN_UNITS:
            path += str(h) + "x"
        path = path[:-1]
        #path += "_" + str(EPISODES) +"eps"
        path += ".h5"
        return load_model(path)


''' initialize the simulation '''
env = gym.make(GYM_ENVIRONMENT)
state_size = env.observation_space.shape
actions = env.action_space.n

buffer = ReplayBuffer(size=BUFFER_SIZE)
agent = Agent(state_size[0], actions,
              lr=LEARNING_RATE,
              hidden=HIDDEN_UNITS,
              gamma=GAMMA,
              epsilon_i=EPSILON_MAX,
              epsilon_f=EPSILON_MIN,
              epsilon_dec=EPSILON_DECAY)

agent.generate_experiences(env, buffer, PRETRAIN_STEPS)

summary = agent.play(env, buffer,
                     batch_size=BATCH_SIZE, episodes=EPISODES,
                     steps_per_episode=STEPS_PER_EPISODE,
                     render_after_n_episodes=RENDER_AFTER)

summary.plot(y='reward', kind='line', figsize=(20, 20), title='Reward curve')
plt.show()
