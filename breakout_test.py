import gym
import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt

# Uncomment line below to play the game as a human
# from gym.utils import play
# play.play(env, zoom=3)
print(tf.__version__)  # for Python 2

# Constants
MEMORY_CAPACITY = 10000 # TODO: 1000000 in paper divided by 4 in our case due to frame skip
PROBLEM = 'BreakoutDeterministic-v4'
NUMBER_OF_EPISODES = 10

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
        self.plot_frame_from_greyscale_values(image)
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
IMAGE_INPUT_HEIGHT, IMAGE_INPUT_WIDTH, IMAGE_INPUT_CHANNELS = 84, 84, 1
CONV1_NUM_FILTERS, CONV1_FILTER_SIZE, CONV1_FILTER_STRIDES = 32, 8, 4
CONV2_NUM_FILTERS, CONV2_FILTER_SIZE, CONV2_FILTER_STRIDES = 64, 4, 2
CONV3_NUM_FILTERS, CONV3_FILTER_SIZE, CONV3_FILTER_STRIDES = 64, 3, 1
DENSE_NUM_UNITS, OUTPUT_NUM_UNITS = 512, 4  # TODO: GET Action count from constructor
LEARNING_RATE, GRADIENT_MOMENTUM, MIN_SQUARED_GRADIENT = 0.00025, 0.95, 0.01
HUBER_LOSS_DELTA, DISCOUNT_FACTOR = 2.0, 0.99  # TODO: is value 1 or 2 in paper?
RANDOM_WEIGHT_INITIALIZER = tf.initializers.RandomNormal()
HIDDEN_ACTIVATION, OUTPUT_ACTIVATION, PADDING = 'relu', 'linear', "SAME"  # TODO: remove?
LEAKY_RELU_ALPHA, DROPOUT_RATE = 0.2, 0.5  # TODO: remove or use to improve paper
optimizer = tf.optimizers.RMSprop(learning_rate=LEARNING_RATE, rho=0.9,
                                  momentum=GRADIENT_MOMENTUM, epsilon=MIN_SQUARED_GRADIENT)

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
    'dense_weights': tf.Variable(RANDOM_WEIGHT_INITIALIZER([CONV3_FILTER_SIZE * CONV3_FILTER_SIZE * CONV3_NUM_FILTERS,
                                                            DENSE_NUM_UNITS])),

    # Output layer: 512 input units, 4 output units (actions)
    'output_weights': tf.Variable(RANDOM_WEIGHT_INITIALIZER([DENSE_NUM_UNITS, OUTPUT_NUM_UNITS]))
}

biases = {
    'conv1_biases': tf.Variable(tf.zeros([CONV1_NUM_FILTERS])),  # 32
    'conv2_biases': tf.Variable(tf.zeros([CONV2_NUM_FILTERS])),  # 64
    'conv3_biases': tf.Variable(tf.zeros([CONV3_NUM_FILTERS])),  # 64
    'dense_biases': tf.Variable(tf.zeros([DENSE_NUM_UNITS])),    # 512
    'output_biases': tf.Variable(tf.zeros([OUTPUT_NUM_UNITS]))   # 4
}


class ConvolutionalNeuralNetwork:

    def __init__(self, number_of_states, number_of_actions):
        self.number_of_states = number_of_states
        self.number_of_actions = number_of_actions

    @tf.function
    def convolutional_2d_layer(self, inputs, filter_weights, biases, strides=1):
        # strides = [1, strides, strides, 1]
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
        dimensions = weights[weights_name].get_shape().as_list()[0]
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

    def train_step_optimization(self, inputs, outputs):
        # Wrap computation inside a GradientTape for automatic differentiation
        with tf.GradientTape() as tape:
            predictions = self.convolutional_neural_network(inputs)
            loss = self.huber_error_loss(predictions, outputs)

        # Trainable variables to update
        trainable_variables = weights.values() + biases.values()

        gradients = tape.gradient(loss, trainable_variables)

        # Update weights and biases following gradients
        optimizer.apply_gradients(zip(gradients, trainable_variables))

        print(tf.reduce_mean(loss))

    # TODO: is prediction correct? 
    def predict(self, states, actions, rewards, is_done):
        next_q_values = self.model([states, np.ones(actions.shape)])
        next_q_values[is_done] = 0  # reset all Q values to 0 if game is done
        q_values = rewards + DISCOUNT_FACTOR * tf.maximum(next_q_values, axis=1)
        return q_values

    def convolutional_neural_network(self, inputs):

        # Input shape: [1, 84, 84, 1]. A batch of 84x84x1 (grayscale) images.
        inputs = tf.reshape(inputs, shape=[-1, IMAGE_INPUT_HEIGHT, IMAGE_INPUT_WIDTH, IMAGE_INPUT_CHANNELS])

        # Convolution Layer 1 with output shape [-1, 84, 84, 32]
        conv1 = self.convolutional_2d_layer(inputs, weights['conv1_weights'], biases['conv1_biases'])

        # Convolutional Layer 2 with output shape [-1, 84, 84, 64]
        conv2 = self.convolutional_2d_layer(conv1, weights['conv2_weights'], biases['conv2_biases'])

        # Flatten output of 2nd conv. layer to fit dense layer input, output shape [-1, 3x3x64]
        flattened_layer = self.flatten_layer(layer=conv2, weights_name='dense_weights')

        # Dense fully connected layer with output shape [-1, 512]
        dense_layer = self.dense_layer(flattened_layer, weights['dense_weights'], biases=biases['dense_biases'])

        # Fully connected output of shape [-1, 4]
        output_layer = self.output_layer(dense_layer, weights['output_weights'], biases=biases['output_biases'])

        return output_layer


FRAME_SKIP = 4
MIN_EXPLORATION_RATE = 0.1
EXPLORATION_RATE = 1
MAX_FRAMES_DECAYED = 1000/FRAME_SKIP  # 1 million in paper
MEMORY_BATCH_SIZE = 32


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
            self.model.train(current_state=state, next_state=next_state, action=action, reward=reward, is_done=is_done)
        # self.model.train(memory_batch)


class Environment:
    """
    Creates a game environment which an agent can play using certain actions.
    Run takes an agent as argument that plays the game, until the agent 'dies' (no more lives)
    """
    def __init__(self, problem):
        self.env = gym.make(problem)
        self.state_space = self.env.observation_space.shape
        self.frame_preprocessor = FramePreprocessor(self.state_space)

    def run(self, agent):
        state = self.env.reset()
        total_reward = 0

        # need to  be while True
        for i in range(1):
            # self.env.render()
            action = agent.choose_action(state)
            next_state, reward, is_done, _ = self.env.step(action)
            preprocessed_next_state = self.frame_preprocessor.preprocess_frame(next_state)

            break

            if is_done:
                next_state = None
            
            experience = (state, action, reward, next_state)
            agent.observe(experience)
            agent.experience_replay()

            state = next_state
            total_reward += reward

            if is_done:
                break

        self.env.close()
        print(f"Total reward: {total_reward}")


game = Environment(PROBLEM)

number_of_states = game.env.observation_space.shape
number_of_actions = game.env.action_space.n

print(number_of_states)
print(number_of_actions)
agent = Agent(number_of_states, number_of_actions)

# for episode in range(NUMBER_OF_EPISODES):
#     env.run(agent)

# need to be while true
for i in range(1):
    game.run(agent)
