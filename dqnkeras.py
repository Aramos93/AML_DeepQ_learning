# -*- coding: utf-8 -*-
"""DQNKeras.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1gYzr5a0TzWCoiMAkvsaSO9WUBoudTyV2
"""

# Libraries 
import gym
import numpy as np
import random
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Conv2D, Dense, Flatten
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.losses import Huber

import tracemalloc
import os
import linecache
def display_top(snapshot, key_type='lineno', limit=10):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        # replace "/path/to/module/file.py" with "module/file.py"
        filename = os.sep.join(frame.filename.split(os.sep)[-2:])
        print("#%s: %s:%s: %.1f KiB"
              % (index, filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))


tracemalloc.start()  # save upto 5 stack frames

# Hyper parameters
PROBLEM = 'BreakoutDeterministic-v4'
FRAME_SKIP = 4
MEMORY_BATCH_SIZE = 32
REPLAY_START_SIZE = 50000
REPLAY_MEMORY_SIZE = 1000000  # RMSProp train updates sampled from this number of recent frames
NUMBER_OF_EPISODES = 20  # TODO: save and restore model with infinite episodes
EXPLORATION_RATE = 1
MIN_EXPLORATION_RATE = 0.1
MAX_FRAMES_DECAYED = REPLAY_MEMORY_SIZE / FRAME_SKIP  # TODO: correct? 1 million in paper
IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 84, 84, 1
CONV1_NUM_FILTERS, CONV1_FILTER_SIZE, CONV1_FILTER_STRIDES = 32, 8, 4
CONV2_NUM_FILTERS, CONV2_FILTER_SIZE, CONV2_FILTER_STRIDES = 64, 4, 2
CONV3_NUM_FILTERS, CONV3_FILTER_SIZE, CONV3_FILTER_STRIDES = 64, 3, 1
DENSE_NUM_UNITS, OUTPUT_NUM_UNITS = 512, 4  # TODO: GET Action count from constructor
LEARNING_RATE, GRADIENT_MOMENTUM, MIN_SQUARED_GRADIENT = 0.00025, 0.95, 0.01
HUBER_LOSS_DELTA, DISCOUNT_FACTOR = 2.0, 0.99  # TODO: is value 1 or 2 in paper for Huber?
RANDOM_WEIGHT_INITIALIZER = tf.initializers.RandomNormal()
HIDDEN_ACTIVATION, OUTPUT_ACTIVATION, PADDING = 'relu', 'linear', "SAME"  # TODO: remove?
LEAKY_RELU_ALPHA, DROPOUT_RATE = 0.2, 0.5  # TODO: remove or use to improve paper
optimizer = tf.optimizers.RMSprop(learning_rate=LEARNING_RATE, rho=GRADIENT_MOMENTUM, epsilon=MIN_SQUARED_GRADIENT)


class FramePreprocessor:
    """
    FramePreprocessor re-sizes, normalizes and converts RGB atari frames to gray scale frames.
    """

    def __init__(self, state_space):
        self.state_space = state_space

    def convert_rgb_to_grayscale(self, tf_frame):
        return tf.image.rgb_to_grayscale(tf_frame)

    def resize_frame(self, tf_frame, frame_height, frame_width):
        return tf.image.resize(tf_frame, [frame_height, frame_width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    def normalize_frame(self, tf_frame):
        return tf_frame / 255

    def plot_frame_from_greyscale_values(self, image):
        height, width, _ = image.shape
        grey_image = np.array([[(image[i, j].numpy()[0], image[i, j].numpy()[0], image[i, j].numpy()[0])
                                for i in range(height)]
                               for j in range(width)])
        grey_image = np.transpose(grey_image, (1, 0, 2))  # Switch height and width
        plt.imshow(grey_image)
        plt.show()

    def preprocess_frame(self, frame):
        tf_frame = tf.Variable(frame, shape=self.state_space, dtype=tf.uint8)
        image = self.convert_rgb_to_grayscale(tf_frame)
        image = self.resize_frame(image, IMAGE_HEIGHT, IMAGE_WIDTH)
        return image


# Todo use experience: (state, action, reward, next_state, is_done)
from typing import NamedTuple, Tuple


class Experience(NamedTuple):
    state: Tuple[int, int, int]  # y, x, c
    action: int
    reward: float
    next_state: Tuple[int, int, int]
    is_done: bool


class ReplayMemory:
    """
    Memory class holds a list of game plays stored as experiences (s,a,r,s', d) = (state, action, reward, next_state, is_done)
    Credits: https://stackoverflow.com/questions/40181284/how-to-get-random-sample-from-deque-in-python-3 
    """

    def __init__(self, capacity):  # Initialize memory with given capacity
        self.experiences = [None] * capacity
        self.capacity = capacity
        self.index = 0
        self.size = 0

    def add(self, experience):  # Add a sample to the memory, removing the earliest entry if memeory capacity is reached
        self.experiences[self.index] = experience
        self.size = min(self.size + 1, self.capacity)
        self.index = (self.index + 1) % self.capacity  # Overwrites earliest entry if memory capacity reached

    def sample(self, size):
        indices = random.sample(range(self.size), size)
        return [self.experiences[index] for index in indices]  # Efficient random access


class ConvolutionalNeuralNetwork:
    """
    CNN CLASS
    Architecture of DQN has 4 hidden layers:

    Input:  84 X 84 X 1 image (4 in paper due to frame skipping) (PREPROCESSED image), Game-score, Life count, Actions_count (4)
    1st Hidden layer: Convolves 32 filters of 8 X 8 with stride 4 (relu)
    2nd hidden layer: Convolves 64 filters of 4 X 4 with stride 2 (relu)
    3rd hidden layer: Convolves 64 filters of 3 X 3 with stride 1 (Relu)
    4th hidden layer: Fully connected, (512 relu units)
    Output: Fully connected linear layer, Separate output unit for each action, outputs are predicted Q-values
    """

    def __init__(self, number_of_states, number_of_actions, model=None):
        self.number_of_states = number_of_states
        self.number_of_actions = number_of_actions
        self.model = self.model() if model is None else load_model(model)

    def model(self):
        model = Sequential()
        model.add(Conv2D(filters=32, kernel_size=8, strides=4, activation='relu',
                         input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)))
        model.add(Conv2D(filters=64, kernel_size=4, strides=2, activation='relu'))
        model.add(Conv2D(filters=64, kernel_size=3, strides=1, activation='relu'))
        model.add(Flatten())
        model.add(Dense(units=512, activation='relu'))
        model.add(Dense(units=number_of_actions, activation='linear'))
        opt = RMSprop(learning_rate=LEARNING_RATE, rho=GRADIENT_MOMENTUM, epsilon=MIN_SQUARED_GRADIENT)
        model.compile(loss=Huber(), optimizer=opt)
        print(model.summary())
        return model

    def train(self, x, y, batch_size=MEMORY_BATCH_SIZE, epochs=1, verbose=0):
        self.model.fit(x, y, batch_size=batch_size, epochs=epochs, verbose=verbose)  # batch_size removed

    def normalize_images(self, images):
        return tf.cast(images / 255, dtype=tf.float32)

    def predict(self, states):
        normalized_states = self.normalize_images(states)
        return self.model.predict(normalized_states)

    def predictOne(self, state):
        state = tf.reshape(state, shape=(1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS))
        return self.predict(state).flatten()


class Agent:
    """
    Agent takes actions and saves them to its memory, which is initialized with a given capacity
    """
    steps = 0
    exploration_rate = EXPLORATION_RATE

    def decay_exploration_rate(self):
        decay_rate = (self.exploration_rate - MIN_EXPLORATION_RATE) / MAX_FRAMES_DECAYED
        return decay_rate

    # Initialize agent with a given memory capacity, and a state, and action space
    def __init__(self, number_of_states, number_of_actions, model=None):
        self.experiences = ReplayMemory(REPLAY_MEMORY_SIZE)
        self.model = ConvolutionalNeuralNetwork(number_of_states, number_of_actions,
                                                model) if model else ConvolutionalNeuralNetwork(number_of_states,
                                                                                                number_of_actions)
        self.number_of_states = number_of_states
        self.number_of_actions = number_of_actions
        self.decay_rate = self.decay_exploration_rate()

    # The behaviour policy during training was e-greedy with e annealed linearly
    # from 1.0 to 0.1 over the first million frames, and fixed at 0.1 thereafter
    def e_greedy_policy(self, state):
        exploration_rate_threshold = random.uniform(0, 1)
        if exploration_rate_threshold > self.exploration_rate:
            next_q_values = self.model.predictOne(state)  # TODO: return only 4 actions not 784 x 4 !!!
            best_action = tf.argmax(next_q_values)
        else:
            best_action = self.random_policy()
        return best_action

    def random_policy(self):
        return random.randint(0, self.number_of_actions - 1)

    def act(self, state):
        return self.random_policy() if self.experiences.size <= REPLAY_START_SIZE else self.e_greedy_policy(state)

    def observe(self, experience):
        self.experiences.add(experience)
        self.steps += 1
        self.exploration_rate = (MIN_EXPLORATION_RATE
                                 if self.exploration_rate <= MIN_EXPLORATION_RATE
                                 else self.exploration_rate - self.decay_rate)

    def replay(self):  # Experience: (state, action, reward, next_state, is_done) # Train neural net with experiences
        memory_batch = self.experiences.sample(MEMORY_BATCH_SIZE)
        memory_batch = [(tf.reshape(state, (1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)), action, reward,
                         np.zeros(shape=(1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)), is_done) if is_done
                        else (tf.reshape(state, (1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)), action, reward,
                              tf.reshape(next_state, (1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)), is_done)
                        for (state, action, reward, next_state, is_done) in memory_batch]

        states = tf.reshape([state for (state, _, _, _, _) in memory_batch],
                            shape=(MEMORY_BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS))
        next_states = tf.reshape([next_state for (_, _, _, next_state, _) in memory_batch],
                                 shape=(MEMORY_BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS))

        state_predictions = self.model.predict(states)
        next_state_predictions = self.model.predict(next_states)

        inputs = np.zeros(shape=(MEMORY_BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS))
        outputs = np.zeros(shape=(MEMORY_BATCH_SIZE, number_of_actions))

        for i, (state, action, reward, next_state, is_done) in enumerate(memory_batch):
            state_target = state_predictions[i]  # target Q(s,a) for state and action [Q1 Q2 Q3 Q4]
            state_target[action] = reward if is_done else reward + DISCOUNT_FACTOR * np.amax(
                next_state_predictions[i])  # TODO: use tf.reduce_max
            inputs[i] = state
            outputs[i] = state_target
        self.model.train(inputs, outputs, MEMORY_BATCH_SIZE)


REPLAY_START_SIZE = 100  # TODO: remove after fix, currently ensures training starts after only 500 random experiences 


class Environment:
    """
    Creates a game environment which an agent can play using certain actions.
    Run takes an agent as argument that plays the game, until the agent 'dies' (no more lives)
    """

    def __init__(self, problem):
        self.gym = gym.make(problem)
        self.state_space = self.gym.observation_space.shape
        self.frame_preprocessor = FramePreprocessor(self.state_space)
        self.best_reward = 0

    def clip_reward(self, reward):  # Clip positive rewards to 1 and negative rewards to -1
        return np.sign(reward)

    def run(self, agent, shouldPrint):
        state = self.gym.reset()
        state = self.frame_preprocessor.preprocess_frame(state)
        total_reward, step = 0, 0

        while True:
            action = agent.act(state)
            next_state, reward, is_done, _ = self.gym.step(action)
            next_state = self.frame_preprocessor.preprocess_frame(next_state)
            reward = self.clip_reward(reward)

            if is_done: next_state = None

            experience = (state, action, reward, next_state, is_done)  # Experience(experience)
            agent.observe(experience)

            if agent.experiences.size > REPLAY_START_SIZE:
                agent.replay()  # Train on states in mini batches

            state = next_state
            total_reward += reward
            step += 1

            if is_done: break

        self.best_reward = total_reward if total_reward > self.best_reward else self.best_reward
        self.gym.close()
        if shouldPrint:
            print(
                f"Total reward: {total_reward} memory: {agent.experiences.size} exploration rate: {agent.exploration_rate} \n")


HAS_MODEL = False
environment = Environment(PROBLEM)
number_of_states = environment.gym.observation_space.shape
number_of_actions = environment.gym.action_space.n
dqn_agent = Agent(number_of_states, number_of_actions, model="Breakout-4.h5") if HAS_MODEL else Agent(number_of_states,
                                                                                                      number_of_actions)

for episode in range(NUMBER_OF_EPISODES):
    shouldPrint = (episode + 1) % 10 == 0
    environment.run(dqn_agent, shouldPrint)
    snapshot = tracemalloc.take_snapshot()
    display_top(snapshot)
    if shouldPrint:
        print(f"Episode: {episode+1} with best reward: {environment.best_reward}")
    if not HAS_MODEL:
        dqn_agent.model.model.save("models\Breakout-4.h5")