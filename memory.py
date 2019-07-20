from collections import deque
import random
import numpy as np

class ReplayMemory:

    def __init__(self, buffer_size, batch_size):
        self.memory = deque(maxlen=buffer_size)
        self.BATCH_SIZE = batch_size
        
    
    def add(self, state, action, reward, next_state, done):
        '''Add a new experience to memory'''
        e = tuple((state, action, reward, next_state, done))
        self.memory.append(e)
        
    
    def sample(self, state_shape):
        '''Randomly sample a batch of experiences from memory'''
        
        experiences = random.sample(self.memory, k=self.BATCH_SIZE)
        #extracting the SARSA
        states, actions, rewards, next_states, dones = zip(*experiences)
 
        #converting them to numpy arrays for easy operations
        states = np.array(states).reshape(self.BATCH_SIZE, state_shape)
        actions = np.array(actions, dtype='int').reshape(self.BATCH_SIZE)
        rewards = np.array(rewards).reshape(self.BATCH_SIZE)
        next_states = np.array(next_states).reshape(self.BATCH_SIZE,state_shape)
        dones = np.array(dones).reshape(self.BATCH_SIZE)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):     
        '''overriding the __len___ method'''
        return len(self.memory)