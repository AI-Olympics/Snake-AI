import sys
from tensorflow.keras.models import load_model

from Agent import DeepQ_agent
from Environment import Snake_Env
from collections import deque
import numpy as np

#----------------------------------colours/colors--------------------------------------------#

WHITE = (255,255,255)
GREEN = (0,0,255)

#-------------------------------------Initialise the environment & agent-----------------------#

#creating the environment
max_env_width, max_env_height = 40, 40               # max size of environment, basically the coordinates where food and snake reside
env_width, env_height = 5, 5                       # starting size of environment, this is kind of related to Curriculum Learning
display_width, display_height = 600, 600            #size of display
env = Snake_Env(max_env_width, max_env_height, env_width, env_height, display_width, display_height)

#----------------------------------Hyperparams ! all the magic happens here----------------------#

HIDDEN_UNITS = (32, 16)
NETWORK_LR = 0.01
BATCH_SIZE = 64
UPDATE_EVERY = 5
GAMMA = 0.95
epsilon, eps_min, eps_decay = 1, 0.05, 0.9997
NUM_EPISODES = 10000    #number of episodes to train
directory = 'Check Training'


#--------------------------------Initialise the DQN agent----------------------------------------#

#def __init__(self, env, hidden_units = None, network_LR = 0.001, batch_size = 64, update_every=4, gamma=1.0):
agent = DeepQ_agent(env, hidden_units = HIDDEN_UNITS, network_LR = NETWORK_LR, batch_size = BATCH_SIZE, update_every = UPDATE_EVERY, gamma = GAMMA)

#---------------------------Let's Train the agent---------------------------------------------------#

'''
#If continuing Training
#set the agent's network to previous state
agent.qnetwork_local.model = load_model('Training 30-30/snake_dqn_final_Wed Jul 17 19:46:05 2019.h5')
agent.qnetwork_target.model = load_model('Training 30-30/snake_dqn_final_Wed Jul 17 19:46:05 2019.h5')
#decrease epsilon to value where it stopped
for i in range(10000):
    epsilon = max(epsilon*eps_decay, eps_min)
print('epsilon', epsilon)
'''

scores, avg_scores = [], []                        # list containing scores from each episode
INCREASE_EVERY, SAVE_EVERY = 500, 500
scores_window = deque(maxlen=INCREASE_EVERY) 
stats = [0,0,0,0,0]


#loop over episodes
for i_episode in range(1, NUM_EPISODES+1):

    epsilon = max(epsilon*eps_decay, eps_min)
    state = env.reset()
    action = agent.act(state, epsilon)

    #render the environment         
    env.render(action, stats, i_episode, epsilon, GAMMA)
    score = 0

    while True:
        
        next_state, reward, done, info = env.step(action)
        
        stats[info] += 1   #collecting some stats
        #add the experience to agent's memory
        agent.add_experience(state, action, reward, next_state, done)
        #let's teach our agent to do something! hopefully they learn.
        agent.learn()
        
        #render the environment
        env.render(action, stats,i_episode, epsilon, GAMMA)

        if done:
            #finish this episode    
            break
        
        #update state and action
        state = next_state
        action = agent.act(state, epsilon)
        score += reward
   
    scores_window.append(score)       # save most recent score
    scores.append(score)              # save most recent score                
    avg_scores.append(np.mean(scores_window))
  
    
    # monitor progress, very important to keep me patient    
    if (i_episode + 1)% SAVE_EVERY == 0:
        agent.save(i_episode+1, directory)       #save the model
        print('\rEpisode {}\t Score {}\tAverage Score: {:.2f}'.format(i_episode+1, score, np.mean(scores_window))) 
        sys.stdout.flush()
    
    #increase environment size
    if (i_episode +1)% INCREASE_EVERY == 0:
        env.change_size(1, 1)  #increase the env size by 1
    
    #after 6k episodes increase up the training process
    if (i_episode + 1) == 6000:
        INCREASE_EVERY = 100  #That is increase the size by 1, every 100 episodes
        SAVE_EVERY = 100      

#save the agent's q-network for testing
agent.save('final', directory)
        