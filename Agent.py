import time
import random
import numpy as np
from collections import deque

from q_network import QNetwork 
from memory import ReplayMemory

class DeepQ_agent:

    def __init__(self, env, hidden_units = None, network_LR = 0.001, batch_size = 64, update_every=4, gamma=1.0):
        self.env = env
        self.BATCH_SIZE = batch_size
        self.GAMMA = gamma
        self.NETWORK_LR = network_LR
        self.MEMORY_CAPACITY = int(1e5)   #this is pythonic
        
        self.nA = env.ACTION_SPACE              #number of actions agent can perform
        self.HIDDEN_UNITS = hidden_units
        self.UPDATE_EVERY = update_every
       
        #let's give it some brains
        self.qnetwork_local = QNetwork(input_shape = self.env.STATE_SPACE,
                                        hidden_units = self.HIDDEN_UNITS,
                                        output_size = self.nA,
                                        learning_rate = self.NETWORK_LR)
        print(self.qnetwork_local.model.summary())
        
        #I call the target network as the PC
        # Where our agent stores all the concrete and important stuff
        self.qnetwork_target = QNetwork(input_shape = self.env.STATE_SPACE,
                                        hidden_units = self.HIDDEN_UNITS,
                                        output_size = self.nA,
                                        learning_rate = self.NETWORK_LR)

        #and the memory of course
        self.memory = ReplayMemory(self.MEMORY_CAPACITY, self.BATCH_SIZE) 

        #handy temp variable
        self.t = 0

#----------------------Learn from experience-----------------------------------#

    def learn(self):
        '''
            hell yeah   
        '''

        if self.memory.__len__() > self.BATCH_SIZE:
            states, actions, rewards, next_states, dones = self.memory.sample(self.env.STATE_SPACE)
            
            #calculating action-values using local network
            target = self.qnetwork_local.predict(states, self.BATCH_SIZE)
            
            #future action-values using target network
            target_val = self.qnetwork_target.predict(next_states, self.BATCH_SIZE)
            
            #future action-values using local network
            target_next = self.qnetwork_local.predict(next_states, self.BATCH_SIZE)
            
            #The main point of Double DQN is selection of action from local network
            #while the update si from target network
            max_action_values = np.argmax(target_next, axis=1)   #action selection
            
            for i in range(self.BATCH_SIZE):
                if dones[i]:
                    target[i][actions[i]] = rewards[i]
                else:
                    target[i][actions[i]] = rewards[i] + self.GAMMA*target_val[i][max_action_values[i]]   #action evaluation
            
            self.qnetwork_local.train(states, target, batch_size = self.BATCH_SIZE)

            if self.t == self.UPDATE_EVERY:
                self.update_target_weights()
                self.t = 0
            else:
                self.t += 1


#-----------------------Time to act-----------------------------------------------#

    def act(self, state, epsilon = 0):                 #set to NO exploration by default
        state = state.reshape((1,)+state.shape)
        action_values = self.qnetwork_local.predict(state)    #returns a vector of size = self.nA
        if random.random() > epsilon:
            action = np.argmax(action_values)      #choose best action - Exploitation
        else:
            action = random.randint(0, self.nA-1)  #choose random action - Exploration
        
        return action

#-----------------------------Add experience to agent's memory------------------------#

    def add_experience(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)


#----------------------Updates values of Target network----------------------------#
    
    def update_target_weights(self):
        #well now we are doing hard update, but we can do soft update also
        self.qnetwork_target.model.set_weights(self.qnetwork_local.model.get_weights())


#---------------------helpful save function-------------------------------------#
    
    def save(self,model_num, directory):
        self.qnetwork_local.model.save(f'{directory}/snake_dqn_{model_num}_{time.asctime()}.h5')
