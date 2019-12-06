# Libraries 
import gym
import numpy as np
import random
import matplotlib.pyplot as plt
import tensorflow as tf
import copy
import sys

# Install TF 2 and enable GPU
# if "2." not in tf.__version__ or not tf.test.is_gpu_available(): 
#   !pip uninstall tensorflow
#   !pip install tensorflow-gpu
# print(f"Tensorflow version: {tf.__version__}")
# print(f"Python version: {sys.version}")
# device_name = tf.test.gpu_device_name()
# if device_name != '/device:GPU:0':
#   raise SystemError('GPU device not found')
# print('Found GPU at: {}'.format(device_name))
# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Hyper parameters 
PROBLEM = 'BreakoutDeterministic-v4'
FRAME_SKIP = 4
MEMORY_BATCH_SIZE = 32
REPLAY_START_SIZE = 50000
REPLAY_MEMORY_SIZE = 1000000  # RMSProp train updates sampled from this number of recent frames
NUMBER_OF_EPISODES = 1000000  # TODO: save and restore model with infinite episodes
EXPLORATION_RATE = 1
MIN_EXPLORATION_RATE = 0.1
MAX_FRAMES_DECAYED = REPLAY_MEMORY_SIZE / FRAME_SKIP  # TODO: correct? 1 million in paper
IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 84, 84, 1  
IMAGE_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS) 
CONV1_NUM_FILTERS, CONV1_FILTER_SIZE, CONV1_FILTER_STRIDES = 32, 8, 4
CONV2_NUM_FILTERS, CONV2_FILTER_SIZE, CONV2_FILTER_STRIDES = 64, 4, 2
CONV3_NUM_FILTERS, CONV3_FILTER_SIZE, CONV3_FILTER_STRIDES = 64, 3, 1
DENSE_NUM_UNITS, OUTPUT_NUM_UNITS = 512, 4  # TODO: GET Action count from constructor
LEARNING_RATE, GRADIENT_MOMENTUM, MIN_SQUARED_GRADIENT = 0.00025, 0.95, 0.01
HUBER_LOSS_DELTA, DISCOUNT_FACTOR = 1.0, 0.99  
RANDOM_WEIGHT_INITIALIZER = tf.initializers.RandomNormal()
HIDDEN_ACTIVATION, OUTPUT_ACTIVATION, PADDING = 'relu', 'linear', "SAME"  # TODO: remove?
TARGET_MODEL_UPDATE_FREQUENCY = 10000
optimizer = tf.optimizers.RMSprop(learning_rate=LEARNING_RATE, rho=GRADIENT_MOMENTUM, epsilon=MIN_SQUARED_GRADIENT)
optimizer = tf.optimizers.RMSprop(learning_rate=LEARNING_RATE, rho=GRADIENT_MOMENTUM, epsilon=MIN_SQUARED_GRADIENT)
# LEAKY_RELU_ALPHA, DROPOUT_RATE = 0.2, 0.5  # TODO: remove or use to improve paper

class FramePreprocessor:
    """
    FramePreprocessor re-sizes, normalizes and converts RGB atari frames to gray scale frames.
    """

    def __init__(self, state_space):
        self.state_space = state_space

    def convert_rgb_to_grayscale(self, tf_frame):
        return tf.image.rgb_to_grayscale(tf_frame)
    
    def resize_frame(self, tf_frame, frame_height, frame_width):
        return tf.image.resize(tf_frame, [frame_height,frame_width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

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
  state: Tuple[int, int, int] # y, x, c
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
    
    def add(self, experience): # Add a sample to the memory, removing the earliest entry if memeory capacity is reached
      self.experiences[self.index] = experience 
      self.size = min(self.size + 1, self.capacity)
      self.index = (self.index + 1) % self.capacity  # Overwrites earliest entry if memory capacity reached

    def sample(self, size): 
      indices = random.sample(range(self.size), size)
      return [self.experiences[index] for index in indices]  # Efficient random access

class ConvolutionalNeuralNetwork:
    """
    CNN Architecture for DQN has 4 hidden layers:
    Input:  84 X 84 X 1 image (4 in paper due to frame skipping) (PREPROCESSED image), Game-score, Life count, Actions_count (4)
    1st Hidden layer: Convolves 32 filters of 8 X 8 with stride 4 (relu)
    2nd hidden layer: Convolves 64 filters of 4 X 4 with stride 2 (relu)
    3rd hidden layer: Convolves 64 filters of 3 X 3 with stride 1 (Relu)
    4th hidden layer: Fully connected, (512 relu units)
    Output: Fully connected linear layer, Separate output unit for each action, outputs are predicted Q-values
    """

    weights = { # 4D: Filter Width, Filter Height, In Channel, Out Channel 
        # Conv Layer 1: 8x8 conv, 1 input (preprocessed image has 1 color channel), 32 output filters
        'conv1_weights': tf.Variable(RANDOM_WEIGHT_INITIALIZER([CONV1_FILTER_SIZE, CONV1_FILTER_SIZE, IMAGE_CHANNELS, CONV1_NUM_FILTERS])), 
        # Conv Layer 2: 4x4 conv, 32 input filters, 64 output filters
        'conv2_weights': tf.Variable(RANDOM_WEIGHT_INITIALIZER([CONV2_FILTER_SIZE, CONV2_FILTER_SIZE, CONV1_NUM_FILTERS, CONV2_NUM_FILTERS])),
        # Conv Layer 3: 3x3 conv, 64 input filters, 64 output filters
        'conv3_weights': tf.Variable(RANDOM_WEIGHT_INITIALIZER([CONV3_FILTER_SIZE, CONV3_FILTER_SIZE, CONV2_NUM_FILTERS, CONV3_NUM_FILTERS])),
        # Fully Connected (Dense) Layer: 3x3x64 inputs (64 filters of size 3x3), 512 output units
        'dense_weights': tf.Variable(RANDOM_WEIGHT_INITIALIZER([IMAGE_HEIGHT * IMAGE_WIDTH * CONV3_NUM_FILTERS, DENSE_NUM_UNITS])),
        # Output layer: 512 input units, 4 output units (actions)
        'output_weights': tf.Variable(RANDOM_WEIGHT_INITIALIZER([DENSE_NUM_UNITS, OUTPUT_NUM_UNITS]))
    }

    biases = {
        'conv1_biases': tf.Variable(tf.zeros([CONV1_NUM_FILTERS])),  # 32
        'conv2_biases': tf.Variable(tf.zeros([CONV2_NUM_FILTERS])),  # 64
        'conv3_biases': tf.Variable(tf.zeros([CONV3_NUM_FILTERS])),  # 64
        'dense_biases': tf.Variable(tf.zeros([DENSE_NUM_UNITS])),  # 512
        'output_biases': tf.Variable(tf.zeros([OUTPUT_NUM_UNITS]))  # 4
    }

    target_weights = { # 4D: Filter Height, Filter Width, In Channel, Out Channel 
        # Conv Layer 1: 8x8 conv, 1 input (preprocessed image has 1 color channel), 32 output filters
        'conv1_target_weights': tf.Variable(RANDOM_WEIGHT_INITIALIZER([CONV1_FILTER_SIZE, CONV1_FILTER_SIZE, IMAGE_CHANNELS, CONV1_NUM_FILTERS])),  # Out Channel
        # Conv Layer 2: 4x4 conv, 32 input filters, 64 output filters
        'conv2_target_weights': tf.Variable(RANDOM_WEIGHT_INITIALIZER([CONV2_FILTER_SIZE, CONV2_FILTER_SIZE, CONV1_NUM_FILTERS, CONV2_NUM_FILTERS])),
        # Conv Layer 3: 3x3 conv, 64 input filters, 64 output filters
        'conv3_target_weights': tf.Variable(RANDOM_WEIGHT_INITIALIZER([CONV3_FILTER_SIZE, CONV3_FILTER_SIZE, CONV2_NUM_FILTERS, CONV3_NUM_FILTERS])),
        # Fully Connected (Dense) Layer: 3x3x64 inputs (64 filters of size 3x3), 512 output units
        'dense_target_weights': tf.Variable(RANDOM_WEIGHT_INITIALIZER([IMAGE_HEIGHT * IMAGE_WIDTH * CONV3_NUM_FILTERS, DENSE_NUM_UNITS])),
        # Output layer: 512 input units, 4 output units (actions)
        'output_target_weights': tf.Variable(RANDOM_WEIGHT_INITIALIZER([DENSE_NUM_UNITS, OUTPUT_NUM_UNITS]))
    }

    target_biases = {
        'conv1_target_biases': tf.Variable(tf.zeros([CONV1_NUM_FILTERS])),  # 32
        'conv2_target_biases': tf.Variable(tf.zeros([CONV2_NUM_FILTERS])),  # 64
        'conv3_target_biases': tf.Variable(tf.zeros([CONV3_NUM_FILTERS])),  # 64
        'dense_target_biases': tf.Variable(tf.zeros([DENSE_NUM_UNITS])),  # 512
        'output_target_biases': tf.Variable(tf.zeros([OUTPUT_NUM_UNITS]))  # 4
    }

    def __init__(self, number_of_states, number_of_actions):  #, model=None):
      self.number_of_states = number_of_states
      self.number_of_actions = number_of_actions

    def overwrite_model_params(self): # Assume same order and length 
      for weight, target_weight_key in zip(self.weights.values(), self.target_weights.keys()): 
        self.target_weights[target_weight_key].assign(tf.identity(weight))

      for bias, target_bias_key in zip(self.biases.values(), self.target_biases.keys()): 
        self.target_biases[target_bias_key].assign(tf.identity(bias)) 
        
    @tf.function
    def normalize_images(self, images):
        return tf.cast(images / 255, dtype=tf.float32)

    @tf.function
    def convolutional_2d_layer(self, inputs, filter_weights, biases, strides=1):
        output = tf.nn.conv2d(inputs, filter_weights, strides, padding=PADDING)  # TODO: padding in paper?
        output_with_bias = tf.nn.bias_add(output, biases)
        activation = tf.nn.relu(output_with_bias)  # non-linearity TODO: improve paper with leaky relu?
        return activation

    @tf.function
    def flatten_layer(self, layer):  # output shape: [32, 64*84*84]
        # Shape: Minibatches: 32, Num of Filters * Img Height, Image width: 64*84*84 = 451584
        memory_batch_size, image_height, image_width, num_filters = layer.get_shape()
        flattened_layer = tf.reshape(layer, (memory_batch_size, num_filters * image_height * image_width))
        return flattened_layer

    @tf.function
    def dense_layer(self, inputs, weights, biases):
        output = tf.nn.bias_add(tf.matmul(inputs, weights), biases)
        dense_activation = tf.nn.relu(output)  # non-linearity
        # dropout = tf.nn.dropout(dense_activation, rate=DROPOUT_RATE)  # TODO: does paper dropout?
        return dense_activation

    @tf.function
    def output_layer(self, input, weights, biases):
        linear_output = tf.nn.bias_add(tf.matmul(input, weights), biases)
        return linear_output

    @tf.function
    def huber_error_loss(self, y_true, y_predictions, delta=1.0):
            y_predictions = tf.cast(y_predictions, dtype=tf.float32)
            errors = y_true - y_predictions
            condition = tf.abs(errors) <= delta
            l2_squared_loss = 0.5 * tf.square(errors)
            l1_absolute_loss = delta * (tf.abs(errors) - 0.5 * delta)
            loss = tf.where(condition, l2_squared_loss, l1_absolute_loss)
            return loss

    @tf.function
    def train(self, inputs, outputs):  # Optimization
        # Wrap computation inside a GradientTape for automatic differentiation
        with tf.GradientTape() as tape:
            predictions = self.predict(inputs)
            current_loss = self.huber_error_loss(predictions, outputs)

        # Trainable variables to update
        trainable_variables = list(self.weights.values()) + list(self.biases.values())

        gradients = tape.gradient(current_loss, trainable_variables)

        # Update weights and biases following gradients
        optimizer.apply_gradients(zip(gradients, trainable_variables))
        # tf.print(tf.reduce_mean(current_loss))
      
    @tf.function
    def predict(self, inputs, is_target = False): # 4D input for CNN: (batch_size, height, width, depth) 
        # Input shape: [32, 84, 84, 1]. A batch of 84x84x1 (gray scale) images.        
        inputs = self.normalize_images(inputs)

        # Convolution Layer 1 with output shape [32, 84, 84, 32]
        conv1_weights = self.target_weights['conv1_target_weights'] if is_target else self.weights['conv1_weights']
        conv1_biases = self.target_biases['conv1_target_biases'] if is_target else self.biases['conv1_biases']
        conv1 = self.convolutional_2d_layer(inputs,conv1_weights,conv1_biases)

        # Convolutional Layer 2 with output shape [32, 84, 84, 64]
        conv2_weights = self.target_weights['conv2_target_weights'] if is_target else self.weights['conv2_weights']
        conv2_biases = self.target_biases['conv2_target_biases'] if is_target else self.biases['conv2_biases']
        conv2 = self.convolutional_2d_layer(conv1, conv2_weights, conv2_biases)

        # Convolutional Layer 3 with output shape [1, 84, 84, 64]
        conv3_weights = self.target_weights['conv3_target_weights'] if is_target else self.weights['conv3_weights']
        conv3_biases = self.target_biases['conv3_target_biases'] if is_target else self.biases['conv3_biases']
        conv3 = self.convolutional_2d_layer(conv2, conv3_weights, conv3_biases)

        # Flatten output of 2nd conv. layer to fit dense layer input, output shape [32, 64*84*84]
        flattened_layer = self.flatten_layer(layer=conv3) 

        # Dense fully connected layer with output shape [1, 512]
        dense_weights = self.target_weights['dense_target_weights'] if is_target else self.weights['dense_weights']
        dense_biases = self.target_biases['dense_target_biases'] if is_target else self.biases['dense_biases']

        dense_layer = self.dense_layer(flattened_layer, dense_weights, dense_biases)

        # Fully connected output of shape [1, 4]
        output_weights = self.target_weights['output_target_weights'] if is_target else self.weights['output_weights']
        output_biases = self.target_biases['output_target_biases'] if is_target else self.biases['output_biases']
        output_layer = self.output_layer(dense_layer, output_weights, output_biases)

        return output_layer

    @tf.function
    def predict_one(self, state, is_target = False):
        state = tf.reshape(state, shape=(1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)) # Reshape 
        prediction = self.predict(state, is_target)
        return prediction

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
    def __init__(self, number_of_states, number_of_actions):
        self.experiences = ReplayMemory(REPLAY_MEMORY_SIZE)
        self.model = ConvolutionalNeuralNetwork(number_of_states, number_of_actions)  
        self.number_of_states = number_of_states
        self.number_of_actions = number_of_actions
        self.decay_rate = self.decay_exploration_rate()

    # The behaviour policy during training was e-greedy with e annealed linearly
    # from 1.0 to 0.1 over the first million frames, and fixed at 0.1 thereafter
    def e_greedy_policy(self, state):
        exploration_rate_threshold = random.uniform(0, 1)
        if exploration_rate_threshold > self.exploration_rate:
            next_q_values = self.model.predict_one(state)
            best_action = np.argmax(next_q_values) # tf.argmax fails on tie 
        else:
            best_action = self.random_policy() 
        return best_action

    def random_policy(self):
        return random.randint(0, self.number_of_actions - 1)

    def act(self, state):
        return self.random_policy() if self.experiences.size <= REPLAY_START_SIZE else self.e_greedy_policy(state)

    def update_target_model(self):
      self.model.overwrite_model_params()
      
    @tf.function 
    def reshape_image(self, images, batch_size=1): 
      return tf.reshape(images, shape=(batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS))

    def observe(self, experience):
        self.experiences.add(experience)
        self.steps += 1
        self.exploration_rate = (MIN_EXPLORATION_RATE if self.exploration_rate <= MIN_EXPLORATION_RATE
                                 else self.exploration_rate - self.decay_rate)
        if self.steps % TARGET_MODEL_UPDATE_FREQUENCY == 0:
          self.update_target_model()

    def replay(self):  # Experience: (state, action, reward, next_state, is_done) # Train neural net with experiences
        memory_batch = self.experiences.sample(MEMORY_BATCH_SIZE)
        memory_batch = [(self.reshape_image(state), action, reward, np.zeros(shape=(1, *IMAGE_SHAPE), dtype=np.uint8), done) if done
                        else (self.reshape_image(state), action, reward, self.reshape_image(next_state), done)
                        for (state, action, reward, next_state, done) in memory_batch]

        states = self.reshape_image([state for (state, *rest) in memory_batch], batch_size=MEMORY_BATCH_SIZE)
        next_states = self.reshape_image([next_state for (_, _, _, next_state, _) in memory_batch], batch_size=MEMORY_BATCH_SIZE)

        state_predictions = self.model.predict(states)
        next_state_predictions = self.model.predict(next_states)
        target_next_state_predictions = self.model.predict(next_states, is_target = True)

        inputs = np.zeros(shape=(MEMORY_BATCH_SIZE, *IMAGE_SHAPE))
        outputs = np.zeros(shape=(MEMORY_BATCH_SIZE, number_of_actions))

        for i, (state, action, reward, next_state, is_done) in enumerate(memory_batch):
            state_target = state_predictions[i].numpy() # Target Q(s,a) for state and action of sample i: [Q1 Q2 Q3 Q4] 
            next_state_target = target_next_state_predictions[i] 
            future_discounted_reward = target_next_state_predictions[i][tf.argmax(next_state_predictions[i])] # QTarget[nextstate][action]
            state_target[action] = reward if is_done else reward + DISCOUNT_FACTOR * future_discounted_reward 
            inputs[i], outputs[i] = state, state_target

        self.model.train(inputs, outputs)

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

    def run(self, agent, should_print):
        state = self.gym.reset()
        state = self.frame_preprocessor.preprocess_frame(state)
        total_reward, step = 0, 0

        while True:
            action = agent.act(state)
            next_state, reward, is_done, _ = self.gym.step(action)
            next_state = self.frame_preprocessor.preprocess_frame(next_state)
            # reward = self.clip_reward(reward) # Only for generalization to other Atari games 

            if is_done: next_state = None

            experience = (state, action, reward, next_state, is_done)  # Experience(experience)
            agent.observe(experience)

            if agent.experiences.size > REPLAY_START_SIZE: # SPEED UP BY TRAINING ONLY EVERY 50th STEP and step < 50:
                agent.replay()  # Train on states in mini batches

            state = next_state
            total_reward += reward
            step += 1

            if is_done: break

        self.best_reward = total_reward if total_reward > self.best_reward else self.best_reward
        self.gym.close()
        if should_print:
            print(f"Total reward: {total_reward} memory: {agent.experiences.size} exploration rate: {agent.exploration_rate} \n")

environment = Environment(PROBLEM)
number_of_states = environment.gym.observation_space.shape
number_of_actions = environment.gym.action_space.n
dqn_agent = Agent(number_of_states, number_of_actions)  

for episode in range(NUMBER_OF_EPISODES):
    should_print = (episode + 1) % 1 == 0
    environment.run(dqn_agent, should_print)
    if should_print:
        print(f"Episode: {episode+1} with best reward: {environment.best_reward}")

# TODO: 3) Save and restore model parameters 2) Convert NP to Tensors 3) Run experiments!!! 
# Report: What did you implement. The experiments, difficulties (local machines, scalability, less episodes and memory) and results. Last 2-3 hours with less experiments. 14 pages 
# Images of architecture, Breakout, convolutions, preprocessed images, Tables of results (time, reward, exploration rate, episodes, memory, hyperparams)
# Intro: Paper 1-2 page Objective, Theory behind CNN and Reinforcement Q Learning and Deep Q Learning 3 pages, Implementation 2 pages, Experiments and Results 2 pages, Discuss Improvements/Conclusion 1 page    
# Improvements: faster machine, scalable optimizations, run with more games, generalize to other games? we run for Breakout but not for generalization 
# Technical: CNN architecture, experience replay, Q target network, 
# 500 episodes play randomly, train 300 episodes, env.render every100th episode and repeat training after,

