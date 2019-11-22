import gym
import numpy as np
import random
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
#from gym.utils import play

# Uncomment line below to play the game as a human
#play.play(env, zoom=3)


#Constants
MEMORY_CAPACITY = 10000
PROBLEM = 'BreakoutDeterministic-v4'
NUMBER_OF_EPISODES = 10




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
Agent takes actions and saves them to its memory, which is initialized with a given capacity
"""

class Agent:
    steps = 0

    def __init__(self, number_of_states, number_of_actions): #Initialize agent with a given memory capacity, and a state, and action space
        self.memory = Memory(MEMORY_CAPACITY)
        self.number_of_states = number_of_states
        self.number_of_actions = number_of_actions
    
    def choose_action(self, state): #choose an action. At the moment, a random action.
        return random.randint(0, self.number_of_actions-1)
    

    def observe(self, sample):
        self.memory.add(sample)
        self.steps += 1
    



"""
Creates a game environment which an agent can play using certain actions.
Run takes an agent as argument that plays the game, until the agent 'dies' (no more lives)
"""
class Environment:
    
    def __init__(self,problem):
        self.env = gym.make(problem)


    def run(self, agent):
        state = self.env.reset()
        total_reward = 0

        while True:
            self.env.render()

            action = agent.choose_action(state)
            next_state, reward, is_done, _ = self.env.step(action)

            if is_done:
                next_state = None
            
            agent.observe((state, action, reward, next_state))
            state = next_state
            total_reward += reward

            if is_done:
                break

        self.env.close()
        print(f"Total reward: {total_reward}")


env = Environment(PROBLEM)

number_of_states = env.env.observation_space.shape[0]
number_of_actions = env.env.action_space.n

agent = Agent(number_of_states, number_of_actions)

# for episode in range(NUMBER_OF_EPISODES):
#     env.run(agent)

while True:
    env.run(agent)
