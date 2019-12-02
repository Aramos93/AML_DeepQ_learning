import gym
import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt

# Uncomment line below to play the game as a human
# from gym.utils import play
# play.play(env, zoom=3)

# Agent and memory constants
MEMORY_CAPACITY = 10000  # TODO: 1000000 in paper divided by 4 in our case due to frame skip
PROBLEM = 'BreakoutDeterministic-v4'
NUMBER_OF_EPISODES = 10
FRAME_SKIP = 4
MIN_EXPLORATION_RATE = 0.1
EXPLORATION_RATE = 1
MAX_FRAMES_DECAYED = 1000/FRAME_SKIP  # 1 million in paper
MEMORY_BATCH_SIZE = 32

# CNN Constants
IMAGE_INPUT_HEIGHT, IMAGE_INPUT_WIDTH, IMAGE_INPUT_CHANNELS = 84, 84, 1
CONV1_NUM_FILTERS, CONV1_FILTER_SIZE, CONV1_FILTER_STRIDES = 32, 8, 4
CONV2_NUM_FILTERS, CONV2_FILTER_SIZE, CONV2_FILTER_STRIDES = 64, 4, 2
CONV3_NUM_FILTERS, CONV3_FILTER_SIZE, CONV3_FILTER_STRIDES = 64, 3, 1
DENSE_NUM_UNITS, OUTPUT_NUM_UNITS = 512, 4  # TODO: GET Action count from constructor
LEARNING_RATE, GRADIENT_MOMENTUM, MIN_SQUARED_GRADIENT = 0.00025, 0.95, 0.01
HUBER_LOSS_DELTA, DISCOUNT_FACTOR = 2.0, 0.99  # TODO: is value 1 or 2 in paper for Huber?
RANDOM_WEIGHT_INITIALIZER = tf.initializers.RandomNormal()
HIDDEN_ACTIVATION, OUTPUT_ACTIVATION, PADDING = 'relu', 'linear', "SAME"  # TODO: remove?
LEAKY_RELU_ALPHA, DROPOUT_RATE = 0.2, 0.5  # TODO: remove or use to improve paper
optimizer = tf.optimizers.RMSprop(learning_rate=LEARNING_RATE, rho=0.9,
                                  momentum=GRADIENT_MOMENTUM, epsilon=MIN_SQUARED_GRADIENT)


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

    def normalize_frame(self, tf_frame):
        return tf_frame/255

    def plot_frame_from_greyscale_values(self, image):
        height, width, _ = image.shape 
        grey_image = np.array([[(image[i, j].numpy()[0], image[i, j].numpy()[0], image[i, j].numpy()[0]) 
                               for i in range(height)] 
                               for j in range(width)])                    
        grey_image = np.transpose(grey_image, (1, 0, 2)) # Switch height and width 
        plt.imshow(grey_image) 
        plt.show()

    def preprocess_frame(self, frame):
        tf_frame = tf.Variable(frame, shape=self.state_space, dtype=tf.uint8)
        image = self.convert_rgb_to_grayscale(tf_frame)
        image = self.resize_frame(image, IMAGE_INPUT_HEIGHT, IMAGE_INPUT_WIDTH)
        image = self.normalize_frame(image)
        image = tf.cast(image, dtype=tf.uint8)

        return image


class Memory:
    """
    Memory class holds a list of game plays stored as experiences (s,a,r,s')
    """
    samples = []
    
    def __init__(self, capacity):  # Initialize memory with given capacity
        self.capacity = capacity
    
    def add(self, sample):  # Add a sample to the memory, removing the earliest entry if memeory capacity is reached
        self.samples.append(sample)
        
        if len(self.samples) > self.capacity:
            self.samples.pop(0)
         
    def get_samples(self, sample_size):  # Return n samples from the memory
        sample_size = min(sample_size, len(self.samples))
        return random.sample(self.samples, sample_size)


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

    weights = {
        # Conv Layer 1: 8x8 conv, 1 input (preprocessed image has 1 color channel), 32 output filters
        'conv1_weights': tf.Variable(RANDOM_WEIGHT_INITIALIZER([CONV1_FILTER_SIZE,  # Filter width
                                                                CONV1_FILTER_SIZE,  # Filter height
                                                                IMAGE_INPUT_CHANNELS,  # In Channel
                                                                CONV1_NUM_FILTERS])),  # Out Channel
        # Conv Layer 2: 4x4 conv, 32 input filters, 64 output filters
        'conv2_weights': tf.Variable(RANDOM_WEIGHT_INITIALIZER([CONV2_FILTER_SIZE,
                                                                CONV2_FILTER_SIZE,
                                                                CONV1_NUM_FILTERS,
                                                                CONV2_NUM_FILTERS])),
        # Conv Layer 3: 3x3 conv, 64 input filters, 64 output filters
        'conv3_weights': tf.Variable(RANDOM_WEIGHT_INITIALIZER([CONV3_FILTER_SIZE,
                                                                CONV3_FILTER_SIZE,
                                                                CONV2_NUM_FILTERS,
                                                                CONV3_NUM_FILTERS])),
        # Fully Connected (Dense) Layer: 3x3x64 inputs (64 filters of size 3x3), 512 output units
        'dense_weights': tf.Variable(
            RANDOM_WEIGHT_INITIALIZER([CONV3_FILTER_SIZE * CONV3_FILTER_SIZE * CONV3_NUM_FILTERS,
                                       DENSE_NUM_UNITS])),

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

    def __init__(self, number_of_states, number_of_actions):
        self.number_of_states = number_of_states
        self.number_of_actions = number_of_actions

    @tf.function
    def convolutional_2d_layer(self, inputs, filter_weights, biases, strides=1):
        strides = [1, strides, strides, 1]
        output = tf.nn.conv2d(inputs, filter_weights, strides, padding=PADDING)  # TODO: padding in paper?
        output_with_bias = tf.nn.bias_add(output, biases)
        activation = tf.nn.relu(output_with_bias)  # non-linearity TODO: improve paper with leaky relu?
        return activation

    # TODO: consider removing since not used
    @tf.function
    def maxpool_layer(self, inputs, pools_dim, strides_dim):
        return tf.nn.max_pool2d(inputs, pools_dim, strides_dim, padding=PADDING)

    @tf.function
    def flatten_layer(self, layer, weights_name='dense_weights'):  # output shape: [-1, 3*3*64]
        dimensions = self.weights[weights_name].get_shape().as_list()[0]
        flattened_layer = tf.reshape(layer, shape=(-1, dimensions))  # -1 flattens into 1-D
        return flattened_layer

    @tf.function
    def dense_layer(self, inputs, weights, biases):
        output = tf.nn.bias_add(tf.matmul(inputs, weights), biases)
        dense_activation = tf.nn.leaky_relu(output, LEAKY_RELU_ALPHA)  # non-linearity
        dropout = tf.nn.dropout(dense_activation, rate=DROPOUT_RATE)  # TODO: does paper dropout?
        return dropout

    @tf.function
    def output_layer(self, input, weights, biases):
        linear_output = tf.nn.bias_add(tf.matmul(input, weights), biases)
        return linear_output

    @tf.function
    def huber_error_loss(self, y_predictions, y_true, delta=1.0):
            errors = y_true - y_predictions

            condition = tf.abs(errors) < HUBER_LOSS_DELTA

            l2_squared_loss = 0.5 * tf.square(errors)
            l1_absolute_loss = HUBER_LOSS_DELTA * (tf.abs(errors) - 0.5 * HUBER_LOSS_DELTA)

            loss = tf.where(condition, l2_squared_loss, l1_absolute_loss)

            return tf.reduce_mean(loss)

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

        print(tf.reduce_mean(current_loss))

    @tf.function
    def predict(self, inputs):

        # Input shape: [1, 84, 84, 1]. A batch of 84x84x1 (grayscale) images.
        inputs = tf.reshape(tf.cast(inputs, dtype=tf.float32), shape=[-1, IMAGE_INPUT_HEIGHT, IMAGE_INPUT_WIDTH, IMAGE_INPUT_CHANNELS])

        print(inputs.shape)

        # Convolution Layer 1 with output shape [-1, 84, 84, 32]
        conv1 = self.convolutional_2d_layer(inputs, self.weights['conv1_weights'], self.biases['conv1_biases'])

        # Convolutional Layer 2 with output shape [-1, 84, 84, 64]
        conv2 = self.convolutional_2d_layer(conv1, self.weights['conv2_weights'], self.biases['conv2_biases'])

        # Flatten output of 2nd conv. layer to fit dense layer input, output shape [-1, 3x3x64]
        flattened_layer = self.flatten_layer(layer=conv2, weights_name='dense_weights')

        # Dense fully connected layer with output shape [-1, 512]
        dense_layer = self.dense_layer(flattened_layer, self.weights['dense_weights'], biases=self.biases['dense_biases'])

        # Fully connected output of shape [-1, 4]
        output_layer = self.output_layer(dense_layer, self.weights['output_weights'], biases=self.biases['output_biases'])

        return output_layer


class Agent:
    """
    Agent takes actions and saves them to its memory, which is initialized with a given capacity
    """
    steps = 0
    exploration_rate = EXPLORATION_RATE
     
    def decay_exploration_rate(self):
        decay_rate = (self.exploration_rate - MIN_EXPLORATION_RATE)/MAX_FRAMES_DECAYED
        return decay_rate

    # Initialize agent with a given memory capacity, and a state, and action space
    def __init__(self, number_of_states, number_of_actions):
        self.replay_memory_buffer = Memory(MEMORY_CAPACITY)
        self.model = ConvolutionalNeuralNetwork(number_of_states, number_of_actions)  # TODO parameters
        self.number_of_states = number_of_states
        self.number_of_actions = number_of_actions
        self.decay_rate = self.decay_exploration_rate()
   
    # The behaviour policy during training was e-greedy with e annealed linearly
    # from1.0 to 0.1 over the first million frames, and fixed at 0.1 thereafter
    def e_greedy_policy(self, state):
        exploration_rate_threshold = random.uniform(0, 1)

        if exploration_rate_threshold > self.exploration_rate:
            next_q_values = self.model.predict(state)  # TODO parameters
            best_action = tf.argmax(next_q_values, 1)
        else:
            best_action = self.random_policy() 

        return best_action
    
    def random_policy(self): 
        return random.randint(0, self.number_of_actions-1)

    def choose_action(self, state):
        return self.e_greedy_policy(state)

    def observe(self, sample):
        self.replay_memory_buffer.add(sample)
        self.steps += 1
        self.exploration_rate = (MIN_EXPLORATION_RATE
                                 if self.exploration_rate <= MIN_EXPLORATION_RATE
                                 else self.exploration_rate - self.decay_rate)

    def experience_replay(self):
        memory_batch = self.replay_memory_buffer.get_samples(MEMORY_BATCH_SIZE)
        for (state, action, reward, next_state, is_done) in memory_batch: 
            self.model.train(next_state, outputs=self.model.predict(next_state))  # TODO: initial state not preprocessed


class Environment:
    """
    Creates a game environment which an agent can play using certain actions.
    Run takes an agent as argument that plays the game, until the agent 'dies' (no more lives)
    """
    def __init__(self, problem):
        self.gym = gym.make(problem)
        self.state_space = self.gym.observation_space.shape
        self.frame_preprocessor = FramePreprocessor(self.state_space)

    def run(self, agent):
        state = self.gym.reset()
        total_reward = 0

        while True:
            self.gym.render()
            action = agent.choose_action(state)
            next_state, reward, is_done, _ = self.gym.step(action)
            preprocessed_next_state = self.frame_preprocessor.preprocess_frame(next_state)
            # self.frame_preprocessor.plot_frame_from_greyscale_values(preprocessed_next_state)

            if is_done:
                next_state = None
            
            experience = (state, action, reward, preprocessed_next_state, is_done)
            agent.observe(experience)
            agent.experience_replay()

            state = next_state
            total_reward += reward

            if is_done:
                break

        self.gym.close()
        print(f"Total reward: {total_reward}")


environment = Environment(PROBLEM)

number_of_states = environment.gym.observation_space.shape
number_of_actions = environment.gym.action_space.n
dqn_agent = Agent(number_of_states, number_of_actions)

for episode in range(NUMBER_OF_EPISODES):
    environment.run(dqn_agent)

