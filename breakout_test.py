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
IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH = 84, 84, 1


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
        image = self.resize_frame(image, IMAGE_HEIGHT, IMAGE_WIDTH)
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

Input:  84 X 84 X 1 image (4 in paper due to fram skipping) (PREPROCESSED image), Game-score, Life count, Actions_count (4)
1st Hidden layer: Convolves 32 filters of 8 X 8 with stride 4 (relu)
2nd hidden layer: Convolves 64 filters of 4 X 4 with stride 2 (relu)
3rd hidden layer: Convolves 64 filters of 3 X 3 with stride 1 (Relu)
4th hidden layer: Fully connected, (512 relu units)
Output: Fully connected linear layer, Separate output unit for each action, outputs are predicted Q-values
"""
FILTER_COUNT_1, FILTER_SIZE_1, FILTER_STRIDE_1 = 32, 8, 4
FILTER_COUNT_2, FILTER_SIZE_2, FILTER_STRIDE_2 = 64, 4, 2
FILTER_COUNT_3, FILTER_SIZE_3, FILTER_STRIDE_3 = 64, 3, 1
DENSE_UNIT_COUNT = 512
OUTPUT_UNIT_COUNT = 4  # TODO: GET Action count from constructor
HIDDEN_ACTIVATION = 'relu'
OUTPUT_ACTIVATION = 'linear'
PADDING = "SAME"
LEAKY_RELU_ALPHA = 0.2
DROPOUT_RATE = 0.5
WEIGHT_INITIALIZOR = tf.initializers.TruncatedNormal()
LEARNING_RATE = 0.00025
GRADIENT_MOMENTUM = 0.95
MIN_SQUARED_GRADIENT = 0.01
optimizer = tf.optimizers.RMSprop(learning_rate=LEARNING_RATE, rho=0.9,
                                  momentum=GRADIENT_MOMENTUM, epsilon=MIN_SQUARED_GRADIENT)
HUBER_LOSS_DELTA = 2.0  # TODO: is value 1 or 2 in paper?
DISCOUNT_FACTOR = 0.99


class ConvolutionalNeuralNetwork:

    def __init__(self, number_of_states, number_of_actions):
        self.number_of_states = number_of_states
        self.number_of_actions = number_of_actions

    @tf.function
    def convolutional_layer(self, input, filters, stride_size):
        output = tf.nn.conv2d(input, filters, stride_size, padding=PADDING)  # TODO: what is padding in paper?
        activation = tf.nn.leaky_relu(output, LEAKY_RELU_ALPHA)  # TODO: paper uses relu
        return activation

    @tf.function
    def maxpool_layer(self, input, pool_size, stride_size):
        return tf.nn.max_pool2d(input, pool_size, stride_size, padding=PADDING) #TODO: size ok?

    @tf.function
    def dense_layer(self, input, weights):
        output = tf.matmul(input, weights)
        dense_activation = tf.nn.leaky_relu(output, LEAKY_RELU_ALPHA)
        dropout = tf.nn.dropout(dense_activation, rate=DROPOUT_RATE)  # TODO: does paper dropout?
        return dropout

    @tf.function
    def output_layer(self, input, weights, bias):
        output = tf.matmul(input, weights) + bias
        return output

    @tf.function
    def huber_error_loss(self, y_predictions, y_true, delta=1.0):
            errors = y_true - y_predictions

            condition = tf.abs(errors) < HUBER_LOSS_DELTA

            L2_squared_loss = 0.5 * tf.square(errors)
            L1_absolute_loss = HUBER_LOSS_DELTA * (tf.abs(errors) - 0.5 * HUBER_LOSS_DELTA)

            loss = tf.where(condition, L2_squared_loss, L1_absolute_loss)

            return tf.reduce_mean(loss)

    def train_step(self, model, inputs, outputs):
        with tf.GradientTape() as tape:
            current_loss = self.huber_error_loss(model(inputs), outputs)
        grads = tape.gradient(current_loss, weights) # TODO: fix weights
        optimizer.apply_gradients(zip(grads, weights))
        print(tf.reduce_mean(current_loss))

    def predict(self, states, actions, rewards, is_done):
        next_q_values = self.model([states, np.ones(actions.shape)])
        next_q_values[is_done] = 0  # reset all Q values to 0 if game is done
        q_values = rewards + DISCOUNT_FACTOR * tf.maximum(next_q_values, axis=1)
        return q_values

    # TODO: separate model creation from prediction
    def model(self, input):
        """
        FILTER_COUNT_1, FILTER_SIZE_1, FILTER_STRIDE_1 = 32, 8, 4
        FILTER_COUNT_2, FILTER_SIZE_2, FILTER_STRIDE_2 = 64, 4, 2
        FILTER_COUNT_3, FILTER_SIZE_3, FILTER_STRIDE_3 = 64, 3, 1
        """
        input = tf.cast(input, dtype=tf.uint8, shape=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH))

        # 4D: filter_height, filter_width, in_channels, out_channels
        filter1 = [FILTER_SIZE_1, FILTER_SIZE_1, FILTER_COUNT_1, FILTER_COUNT_1]
        filter2 = [FILTER_SIZE_2, FILTER_SIZE_2, FILTER_COUNT_1, FILTER_COUNT_2]
        filter3 = [FILTER_SIZE_3, FILTER_SIZE_3, FILTER_COUNT_2, FILTER_COUNT_3] 

        # TODO: Max pool and flatten before dense layer
        filter_weights_1 = tf.Variable(WEIGHT_INITIALIZOR(shape=filter1), dtype=tf.float32)
        filter_weights_2 = tf.Variable(WEIGHT_INITIALIZOR(shape=filter2), dtype=tf.float32)
        filter_weights_3 = tf.Variable(WEIGHT_INITIALIZOR(shape=filter3), dtype=tf.float32)

        # TODO: Consider maxpool after each conv layer
        convolutional_layer_1 = self.convolutional_layer(input, filter_weights_1, FILTER_STRIDE_1)
        convolutional_layer_2 = self.convolutional_layer(convolutional_layer_1, filter_weights_2, FILTER_STRIDE_2)
        convolutional_layer_3 = self.convolutional_layer(convolutional_layer_2, filter_weights_3, FILTER_STRIDE_3)
        flattened_layer = tf.compat.v1.layers.flatten(convolutional_layer_3)
        # flattened_layer = tf.reshape(conv3, shape=(tf.shape(conv3)[0], -1)) TODO: Consider using this flatten?

        dense_weights_shape = tf.shape(flattened_layer)
        dense_weights = tf.Variable(WEIGHT_INITIALIZOR(dense_weights_shape, dtype=tf.float32))

        dense_layer = self.dense_layer(flattened_layer, dense_weights)

        output_weights_shape = tf.shape(dense_layer)  # TODO: DENSE UNIT COUNT 512
        output_weights = tf.Variable(WEIGHT_INITIALIZOR(output_weights_shape, dtype=tf.float32))
        bias = tf.zeros(OUTPUT_UNIT_COUNT)
        output_layer = self.output_layer(dense_layer, output_weights, bias)

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
