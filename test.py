import cv2
from tensorflow.keras.models import load_model
import time

from Agent import DeepQ_agent
from Environment import Snake_Env


#creating the environment
max_env_width, max_env_height = 37, 37
env_width, env_height = 37-2, 37-2
display_width, display_height = 37*18, 37*18
env = Snake_Env(max_env_width, max_env_height, env_width, env_height, display_width, display_height)

agent = DeepQ_agent(env, hidden_units=(32, 16, 10))
agent.qnetwork_local.model = load_model('Easy Training/Training 11/snake_dqn_final_Sat Jul 20 11_54_02 2019.h5')
NUM_TIMES = 20

stats = [0,0,0,0]
#testing the agent
for i in range(NUM_TIMES):  #running for 10 times
    state = env.reset()
    env.render(0, stats, train = False)
    total_reward = 0
    while True:
        
        #decide action for present state
        action = agent.act(state)
        state, reward, done, _ = env.step(action)
        #rendering the environment
        env.render(action, stats, train = False)
        time.sleep(0.02)
        total_reward += reward
        if done:
            time.sleep(2)
            print(total_reward)
            break 