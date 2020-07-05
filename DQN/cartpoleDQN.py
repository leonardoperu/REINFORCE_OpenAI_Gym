import numpy as np
import tensorflow as tf
import gym
from matplotlib import pyplot as plt
from collections import deque

""" Cartpole initialization """
env = gym.make('CartPole-v0')
print("Action space: ", env.action_space)
print("Observation space: ", env.observation_space)
print("\thigh: ", env.observation_space.high)
print("\tlow: ", env.observation_space.low)


""" Random actions for 1000 steps """
#env.reset()
#for _ in range(1000):
#    env.render()
#    env.step(env.action_space.sample())
#env.close()


""" Model building """
input_shape = [4]   #equal to env.observation_space.shape
n_outputs = 2       #equal to env.action_space.n

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=input_shape),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(n_outputs)  # activation=None by default
])


""" Definition of the epsilon-greedy policy"""
def epsilon_greedy_policy(state, epsilon=0):
    if np.random.rand() < epsilon:
        return np.random.randint(2)
    else:
        Q_values = model.predict(state[np.newaxis])  # np.newaxis adds a second dimension to the state: [[-0.0116 -0.3969 00213 0.5915]] -> shape (1, 4)
        return np.argmax(Q_values[0])   # Q values for actions 0 and 1 (push cart to left/right) : [[0.10328 0.05753]]
#                                       # np.argmax returns the index of the highest Q value


""" creation of the replay buffer and sampling experience"""
replay_buffer = deque(maxlen=10000)


def sample_experiences(batch_size):
    indices = np.random.randint(len(replay_buffer), size=batch_size)
    batch = [replay_buffer[index] for index in indices]
    states, actions, rewards, next_states, dones = [np.array([experience[field_index] for experience in batch])
                                                    for field_index in range(5)]
    return states, actions, rewards, next_states, dones


""" Network initialization """
batch_size = 32
discount_factor = 0.95  # gamma
optimizer = tf.keras.optimizers.Adam(lr=1e-2)
loss_fn = tf.keras.losses.mean_squared_error


""" Training """
def training_step(batch_size):
    experiences = sample_experiences(batch_size)
    states, actions, rewards, next_states, dones = experiences
#    prova_nextQ = model(next_states)
    next_Q_values = model.predict(next_states)  # ARRAY w/ shape (32, 2): 32 couples of Q values for the two actions
    max_next_Q_values = np.max(next_Q_values, axis=1)   # array w/ shape (32,) : for each couple of Q values, pick the highest value (not index)
    target_Q_values = rewards + (1 - dones) * discount_factor * max_next_Q_values
    mask = tf.one_hot(actions, n_outputs) # one-hot version of the actions array (action chosen by the model): [[1. 0.] [0. 1.] [0. 1.] .... ]
    with tf.GradientTape() as tape:
        all_Q_values = model(states)    # EAGERTENSOR network's prediction of the Q values for the current state -> 32 couples Q(S_t, A_t)
#                                       # tf.Tensor( [[ 0.11906283  0.05771903] [ 0.07355762  0.03493534] [ 0.22810857  0.06732281] .... ]
        Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)    # (all_Q_values * mask): # tf.Tensor( [[ 0.11906283  0.] [ 0.  0.03493534] [ 0.  0.06732281] .... ]
#                                                                               # Q_values: tf.Tensor( [[0.11906283] [0.03493534] [0.06732281] .... ]
        loss = tf.reduce_mean(loss_fn(target_Q_values, Q_values))
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))


""" Evaluation """
def play_one_step(env, state, epsilon):
    action = epsilon_greedy_policy(state, epsilon)
    next_state, reward, done, info = env.step(action)
    replay_buffer.append((state, action, reward, next_state, done))
    return next_state, reward, done, info

max_step = 0
steps_scores = []
for episode in range(1000):
    print("Episode ", episode)
    obs = env.reset()
    last_step = 0
    for step in range(150):
        last_step = step
        epsilon = max(1-episode/500, 0.01)
        obs, reward, done, info = play_one_step(env, obs, epsilon)
        if episode >= 900:
            env.render()
        if done:
            break
        if episode > 49:
            training_step(batch_size)
        if step >= max_step:
            max_step = step
            print("\t" + str(step+1) + " steps reached")
    steps_scores.append(last_step+1)

plt.figure(figsize=(64, 40))
plt.plot(range(1, 1000+1), steps_scores, '-o', linewidth=4, markersize=8)
plt.xlabel('episode')
plt.ylabel('steps')
plt.title("Steps per episode", fontsize=24)
plt.grid()
plt.show();

""" Atari environment: Breakout - random actions"""
#env2 = gym.make('Breakout-v0')
#for i_episode in range(10):
#    print("EPISODE " + str(i_episode))
#    observation = env2.reset()
#    for t in range(1000):
#        env2.render()
#        print(observation)
#        action = env2.action_space.sample()
#        observation, reward, done, info = env2.step(action)
#        if done:
#            print("Episode finished after {} timesteps".format(t + 1))
#            print("=" * 40)
#            break
#env2.close()
