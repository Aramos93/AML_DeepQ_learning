import gym
import numpy as np
import tensorflow as tf
import random
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
#from gym.utils import play

# Uncomment line below to play the game as a human
#play.play(env, zoom=3)


print(tf.__version__)  # for Python 2

#Constants
MEMORY_CAPACITY = 10000
PROBLEM = 'BreakoutDeterministic-v4'
NUMBER_OF_EPISODES = 10
FRAME_HEIGHT, FRAME_WIDTH = 84, 84 #TODO downsample to half?


"""
FramePreprocessor resizes, normalizes and converts rgb atari frames
to grayscale frames
"""
class FramePreprocessor:

    def __init__(self, state_space):
        self.state_space = state_space

    #TODO Do we get 1 grayscale?
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
        image = self.resize_frame(image, FRAME_HEIGHT, FRAME_WIDTH)
        self.plot_frame_from_greyscale_values(image)
        image = self.normalize_frame(image)
        image = tf.cast(image, dtype=tf.uint8)

        return image



"""
Memory class holds a list of gameplays stored as (s,a,r,s')
"""
class Memory:   
    samples = []
    
    def __init__(self, capacity): # Initialize memory with given capacity
        self.capacity = capacity
    
    def add(self, sample):  # Add a sample to the memory, removing the earliest entry if memeory capacity is reached
        self.samples.append(sample)
        
        if len(self.samples) > self.capacity:
            self.samples.pop(0)
         
    def get_samples(self, N): # Return n samples from the memory
        N = min(N,len(self.samples))
        return random.sample(self.samples, N)


"""
CNN CLASS
Architecture of DQN has 4 hidden layers:

Input:  84 X 84 X 1 image (4 in paper due to fram skipping) (PREPROCESSED image), Game-score, Life count, Actions_count (4)
1st Hidden layer: Convolves 32 filters of 8 X 8 with stride 4 (relu)
2nd hidden layer: Convolves 64 filters of 4 X 4 with stride 2 (relu)
3rd hidden layer: Convolves 64 filters of 3 X 3 with stride 1 (Relu)
4th hidden layer: Fully connected, (512 relu units)
Output: Fully connected linear layer, Seperate output unit for each action, outputs are predicted Q-values
"""
(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH) =  84, 84, 1 #TODO: Do we need the last dimension?
FILTER_COUNT_1, FILTER_SIZE_1, FILTER_STRIDE_1 = 32, 8, 4
FILTER_COUNT_2, FILTER_SIZE_2, FILTER_STRIDE_2 = 64, 4, 2
FILTER_COUNT_3, FILTER_SIZE_3, FILTER_STRIDE_3 = 64, 3, 1
DENSE_UNIT_COUNT = 512
OUTPUT_UNIT_COUNT = 4 #TODO: GET Action count from constructor
HIDDEN_ACTIVATION = 'relu'
OUTPUT_ACTIVATION = 'linear'
PADDING = "SAME"
LEAKY_RELU_ALPHA = 0.2
WEIGHT_INITIALIZOR = tf.initializers.TruncatedNormal()

class CNN:

    def __init__(self, number_of_states, number_of_actions):
        self.number_of_states = number_of_states
        self.number_of_actions = number_of_actions
    
    def convolutional_layer(self, input, filters, stride_size):
        output = tf.nn.conv2d(input, filters, stride_size, padding=PADDING) #TODO: what is padding in paper?
        activation = tf.nn.leaky_relu(output, LEAKY_RELU_ALPHA) #TODO: paper uses relu
        return activation

    def maxpool_layer(self, input, pool_size, stride_size):
        return tf.nn.max_pool2d(input, pool_size, stride_size, padding=PADDING) #TODO: size ok?
    
    #TODO: Consider dropout: return tf.nn.dropout(dense_activation, rate=0.5)
    def dense_layer(self, input, weights):
        output = tf.matmul(input, weights)
        dense_activation = tf.nn.leaky_relu(output, LEAKY_RELU_ALPHA)
        return dense_activation

    def output_layer(self, input, weights, bias):
        output = tf.matmul(input, weights) + bias
        return output

    """
    FILTER_COUNT_1, FILTER_SIZE_1, FILTER_STRIDE_1 = 32, 8, 4
    FILTER_COUNT_2, FILTER_SIZE_2, FILTER_STRIDE_2 = 64, 4, 2
    FILTER_COUNT_3, FILTER_SIZE_3, FILTER_STRIDE_3 = 64, 3, 1
    """
    def create_model(self, input):
        #4D: filter_height, filter_width, in_channels, out_channels 
        filter1 = [FILTER_SIZE_1, FILTER_SIZE_1, FILTER_COUNT_1, FILTER_COUNT_1]
        filter2 = [FILTER_SIZE_2, FILTER_SIZE_2, FILTER_COUNT_1, FILTER_COUNT_2]
        filter3 = [FILTER_SIZE_3, FILTER_SIZE_3, FILTER_COUNT_2, FILTER_COUNT_3] 

        #TODO: Maxpool and flatten before dense layer
        filter_weights_1 = tf.Variable(WEIGHT_INITIALIZOR(shape=filter1), dtype=tf.float32)
        filter_weights_2 = tf.Variable(WEIGHT_INITIALIZOR(shape=filter2), dtype=tf.float32)
        filter_weights_3 = tf.Variable(WEIGHT_INITIALIZOR(shape=filter3), dtype=tf.float32)
        
        input = tf.cast(input, dtype=tf.uint8)

        #TODO: Consider maxpool after each conv layer
        conv1 = self.convolutional_layer(input, filter_weights_1, FILTER_STRIDE_1)
        conv2 = self.convolutional_layer(conv1, filter_weights_2, FILTER_STRIDE_2)
        conv3 = self.convolutional_layer(conv2, filter_weights_3, FILTER_STRIDE_3)
        flattened_layer = tf.compat.v1.layers.flatten(conv3)
        #flattened_layer = tf.reshape(conv3, shape=(tf.shape(conv3)[0], -1)) TODO: Consider using this flatten?

        dense_weights_shape = tf.shape(flattened_layer)
        dense_weights = tf.Variable(WEIGHT_INITIALIZOR(dense_weights_shape, dtype=tf.float32))

        dense_layer = self.dense_layer(flattened_layer, dense_weights)

        output_weights_shape = tf.shape(dense_layer) #TODO: DENSE UNIT COUNT 512
        output_weights = tf.Variable(WEIGHT_INITIALIZOR(output_weights_shape, dtype=tf.float32))
        bias = tf.zeros(OUTPUT_UNIT_COUNT)
        output_layer = self.output_layer(dense_layer, output_weights, bias)

        return output_layer





"""
Agent takes actions and saves them to its memory, which is initialized with a given capacity
"""
FRAME_SKIP = 4
MIN_EXPLORATION_RATE = 0.1
EXPLORATION_RATE = 1
MAX_FRAMES_DECAYED = 1000/FRAME_SKIP #1 million in paper
MEMORY_BATCH_SIZE = 32

class Agent:
    steps = 0
    exploration_rate = EXPLORATION_RATE
     
    def decay_exploration_rate(self):
        decay_rate = (self.exploration_rate - MIN_EXPLORATION_RATE)/MAX_FRAMES_DECAYED
        return decay_rate

    def __init__(self, number_of_states, number_of_actions): #Initialize agent with a given memory capacity, and a state, and action space
        self.replay_memory_buffer = Memory(MEMORY_CAPACITY)
        self.model = CNN(number_of_states, number_of_actions) #TODO parameters
        self.number_of_states = number_of_states
        self.number_of_actions = number_of_actions
        self.decay_rate = self.decay_exploration_rate()
    
    
   
    # The behaviour policy during training was e-greedy with e annealed linearly
    # from1.0 to 0.1 over the firstmillion frames, and fixed at 0.1 thereafter
    def e_greedy_policy(self, state):
        exploration_rate_threshold = random.uniform(0,1)

        return random.randint(0, self.number_of_actions-1)
        #TODO replace random_action with e_greedy
        # if exploration_rate_threshold > self.exploration_rate:
        #     next_q_values = self.model.predict(state) #TODO parameters
        #     best_action = tf.argmax(next_q_values,1)
        # else:
        #     best_action = random.randint(0, self.number_of_actions-1)

        # return best_action
    
    def choose_action(self, state): #choose an action. At the moment, a random action.
        return self.e_greedy_policy(state)
    

    def observe(self, sample):
        self.replay_memory_buffer.add(sample)
        self.steps += 1
        self.exploration_rate = MIN_EXPLORATION_RATE if self.exploration_rate <= MIN_EXPLORATION_RATE else self.exploration_rate - self.decay_rate
        
    
    def experience_replay(self):
        memory_batch = self.replay_memory_buffer.get_samples(MEMORY_BATCH_SIZE)
        #self.model.train(memory_batch)




"""
Creates a game environment which an agent can play using certain actions.
Run takes an agent as argument that plays the game, until the agent 'dies' (no more lives)
"""
class Environment:
    
    def __init__(self,problem):
        self.env = gym.make(problem)
        self.state_space = self.env.observation_space.shape
        self.frame_preprocessor = FramePreprocessor(self.state_space)


    def run(self, agent):
        state = self.env.reset()
        total_reward = 0

        #need to  be while Trie
        for i in range(1):
            #self.env.render()

            action = agent.choose_action(state)
            next_state, reward, is_done, _ = self.env.step(action)
            preprocessed_next_state = self.frame_preprocessor.preprocess_frame(next_state)
    
            #print(preprocessed_next_state)
            #greyscale_image = np.array([(g, g, g) for g in np.array(preprocessed_next_state)])
            #plt.imshow(next_state)
            #lt.show()
            break
            # if is_done:
            #     next_state = None
            
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

#need to be while true
for i in range(1):
    game.run(agent)
