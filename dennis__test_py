import gym
import numpy as np
import tensorflow as tf
import random
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

#Keras
from keras.models import Model
from keras.layers import Dense, Conv2D, Flatten, Input, multiply
from keras.optimizers import RMSprop

#from gym.utils import play

# Uncomment line below to play the game as a human
#play.play(env, zoom=3)


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
        image = self.normalize_frame(image)
        # self.plot_frame_from_greyscale_values(image)

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
"""
class CNN:
    def __init__(self, input_dimensions, n_actions):
        self.model = self.init_keras_CNN(input_dimensions,n_actions)

    def init_keras_CNN(self, input_dimensions, n_actions):
        # INPUTS
        frames_input = Input(shape = input_dimensions, name='frames')
        actions_input = Input(shape = (n_actions,), name='action_mask')

        # CONVOLUTION LAYER - kernel_size = denotes the size of a filter 8x8
        conv_1 = Conv2D(filters = 32, kernel_size = (8,8), strides = 4,activation= 'relu')(inputs = frames_input) 
        conv_2 = Conv2D(filters = 64, kernel_size = (4,4), strides = 2,activation= 'relu')(inputs = conv_1)
        conv_3 = Conv2D(filters = 64, kernel_size = (3,3), strides = 1,activation= 'relu')(inputs = conv_2)
        
        flatten = Flatten()(inputs = conv_3) # Flatten serves as a connection between the convolution and dense layers.

        # HIDDEN LAYER
        hidden = Dense(units = 512, activation='relu')(flatten) #  LAYER FULLY CONNECTED
        
        # Output layer connected to each action
        output = Dense(units = n_actions, activation='linear')(hidden)
        filtered_output = multiply(inputs = [output, actions_input]) # multiply the output by the mask
             
        # Set model
        model = Model(input=[frames_input,actions_input], output=filtered_output)

        #Setup optimizer
        learning_rate = 0.00025
        gradient_momentum = 0.95
        min_sqrd_gradient = 0.01
        loss_func = 'mse'

        optimizer = RMSprop(lr = learning_rate, rho = gradient_momentum, epsilon=min_sqrd_gradient)
        model.compile(optimizer = optimizer, loss=loss_func)

        return model


    def DQN_train(self, start_states, next_states,actions,rewards, is_done):
        q_values = self.DQN_predict(next_states,actions,rewards, is_done)
        
        #Fit only one iteration
        self.model.fit([start_states,actions], actions * q_values[:,None], nb_epoch = 1, batch_size = len(start_states),verbose=0)

    def DQN_predict(self, state,actions,rewards, is_done, discount_factor = 0.99):
        next_q_values = self.model.predict([state, np.ones(actions.shape)])
        next_q_values[is_done] = 0 # if game is done override all Q values to 0
        
        q_values = rewards + discount_factor * np.max(next_q_values, axis=1)
        return q_values

    def train(self, start_states, next_states, actions, rewards, is_done):
        self.DQN_train(start_states = start_states, next_states = next_states, actions = actions, rewards = rewards, is_done = is_done)

    def predict(self,state, actions, rewards, is_done):
        return self.DQN_predict(state, actions, rewards, is_done)
    



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

    def __init__(self, number_of_states, actions): #Initialize agent with a given memory capacity, and a state, and action space
        self.replay_memory_buffer = Memory(MEMORY_CAPACITY)
        self.model = CNN(number_of_states, number_of_actions) #TODO parameters
        self.number_of_states = number_of_states
        self.actions = actions
        self.number_of_actions = actions.n
        self.decay_rate = self.decay_exploration_rate()
    
    
   
    # The behaviour policy during training was e-greedy with e annealed linearly
    # from1.0 to 0.1 over the firstmillion frames, and fixed at 0.1 thereafter
    def e_greedy_policy(self, state):
        exploration_rate_threshold = random.uniform(0,1)

        if exploration_rate_threshold > self.exploration_rate:
            next_q_values = self.model.predict(state, self.actions) #TODO parameters
            best_action = tf.argmax(next_q_values,1)
        else:
            best_action = random.randint(0, self.number_of_actions-1)

        return best_action

    def random_policy(self):
        return random.randint(0, self.number_of_actions-1)

    def choose_action(self, state): #choose an action. At the moment, a random action.
        if self.model is None:
            return self.random_policy()
        else:
            return self.e_greedy_policy(state)
    
    def observe(self, sample):
        self.replay_memory_buffer.add(sample)
        self.steps += 1

        #Decay exploration rate until min threshold reached
        self.exploration_rate = MIN_EXPLORATION_RATE if self.exploration_rate <= MIN_EXPLORATION_RATE else self.exploration_rate - self.decay_rate
        
    def experience_replay(self):
        memory_batch = self.replay_memory_buffer.get_samples(MEMORY_BATCH_SIZE)
        if self.model is not None:
            for (state, action, reward, next_state, is_done) in memory_batch:
                self.model.train(start_states = state, next_states = next_state,actions = action ,rewards = reward, is_done = is_done)




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
        is_done = False

        #need to  be while Trie
        while True:
            self.env.render()
            action = agent.choose_action(state,)
            next_state, reward, is_done, _ = self.env.step(action)
            preprocessed_next_state = self.frame_preprocessor.preprocess_frame(next_state)

            if is_done:
                next_state = None
            
            experience = (state, action, reward, preprocessed_next_state, is_done)
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

agent = Agent(number_of_states, game.env.action_space)

# for episode in range(NUMBER_OF_EPISODES):
#     env.run(agent)

#need to be while true
# for i in range(1):
while True:
    game.run(agent)
