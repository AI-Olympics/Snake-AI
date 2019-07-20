import random
from PIL import Image
import cv2
import numpy as np
import time

from snake import Snake
from collections import deque
from display_utils import add_info

#colours - b, g, r
WHITE = (255,255,255)
BLUE = (255,0,0)
BLACK = (0,0,0)
GREEN = (0,255,0)
RED = (0,0,255)
FOOD_COLOR = RED
BOUNDARY_COLOR = BLACK

class Snake_Env():

    def __init__(self, max_width, max_height, init_width, init_height, display_width, display_height):

        self.MAX_WIDTH = max_width
        self.MAX_HEIGHT = max_height
        self.WIDTH = init_width
        self.HEIGHT = init_height
        self.DISPLAY_WIDTH, self.DISPLAY_HEIGHT = display_width, display_height
        
        self.STATE_SPACE = 18  #no of inputs to be fed to the neural network
        self.ACTION_SPACE = 4
                
#---------------------------------------Increase the size of environment--------------------------------------#
    def change_size(self, width_change, height_change):
        self.WIDTH = min(self.WIDTH + width_change, self.MAX_WIDTH-2)
        self.HEIGHT = min(self.HEIGHT + height_change, self.MAX_HEIGHT-2)

#--------------------------------helpful function to get boundaries---------------------------------------------#        
            
    def get_boundaries(self):
        odd_w, odd_h = self.WIDTH%2, self.HEIGHT%2
        mid_width, mid_height = int(self.MAX_WIDTH/2), int(self.MAX_HEIGHT/2)
        x1, x2 = (mid_width-int(self.WIDTH/2), mid_height-1+int(self.WIDTH/2)+odd_w) 
        y1, y2 = (mid_width-int(self.HEIGHT/2), mid_height-1+int(self.HEIGHT/2)+odd_h)
        return x1, x2, y1, y2

#-------------------------------Returns random positions within the boundary-----------------------------------#

    def get_randoms(self, length = 1):
        x1, x2, y1, y2 = self.get_boundaries()
        
        #logic to keep the snake AI inside the boundary
        if length>1:
            max_w, max_h = self.MAX_WIDTH-length, self.MAX_HEIGHT-length
            if x1<length-1: x1 = length-1
            if x2>max_w: x2 = max_w
            if y1<length-1: y1 = length-1
            if y2>max_h: y2 = max_h    
        a = random.randint(x1, x2)
        b = random.randint(y1, y2)
        return a,b  

    def play_region(self, env):
        x1, x2, y1, y2 = self.get_boundaries()
        env[y1:y2+1, x1:x2+1, :] = WHITE
        return env

#------------------------------------------------Reset the environment---------------------------------------#

    def reset(self):

        #initialising the snake and food
        snake_x, snake_y = self.get_randoms(length=4)  
        self.SNAKE = Snake(snake_x, snake_y , self.WIDTH, self.HEIGHT)
        self.VELOCITY = self.SNAKE.INITIAL_DIRECTION
        self.FOOD_X, self.FOOD_Y = self.get_randoms()
        self.STATE =  self.SNAKE.look(self.FOOD_X, self.FOOD_Y, self.get_boundaries())
    
        return self.STATE

#-------------------------------------------------Render the environment-----------------------------------------#
    
    def render(self, action, stats, episode=-1, epsilon=-1, gamma=-1, train = True):
        
        env = np.zeros((self.MAX_HEIGHT, self.MAX_WIDTH, 3), dtype = np.uint8) #basically an image with 3 channels RGB
        env = self.play_region(env)                    #get the allowed boundary
        env[self.FOOD_Y, self.FOOD_X] = FOOD_COLOR     #add the food
        env = self.SNAKE.draw(env)                     #add the snake 
        
        #sprinkle some info to the display
        img = add_info(self.DISPLAY_WIDTH, self.DISPLAY_HEIGHT, env, action, 
                                    self.SNAKE.LENGTH, stats, episode, epsilon, gamma, train)    #from the display_utils
        cv2.imshow("Snake Game", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):  #when Q is pressed
            print('Stop Execution')
            cv2.destroyAllWindows()
            quit()

#-----------------------------------Agent takes an action and environment changes----------------------------------------#

    def step(self,action):

        reward, info = -1, 0  # Should we give positive reward for each move. what do u think. write in the comments??
        #info = 0:nothing, bite:1, eat:2, out_boundary:3, moves_done:4 
        done, food_eaten = False, False
        update = False

        x_change, y_change = 0, 0
        # logic to decide the action by the environment
        # Note that if the snake is going up, and decides to move down straight away
        # it's not allowed to do that, by this block of code
        if action == 0:
            if self.VELOCITY == 1: #if moving right
                x_change = 1 
            else:
                x_change = -1      #move left
                update = True
        elif action == 1:
            if self.VELOCITY == 0: #if moving right
                x_change = -1 
            else:
                x_change = 1       #move right
                update = True
        elif action == 2:
            if self.VELOCITY == 3: #if moving down
                y_change = 1 
            else:
                y_change = -1      #move up
                update = True
        elif action ==3:
            if self.VELOCITY == 1: #if moving up
                y_change = -1 
            else:
                y_change = 1        #move down
                update = True
        
        if update:
            self.VELOCITY = action
        head_x, head_y = self.SNAKE.head_pos()
        new_head_x, new_head_y = head_x + x_change, head_y + y_change   #new head position
        
        # check if the snake bit itself
        if self.SNAKE.is_on_body(new_head_x, new_head_y) and self.SNAKE.LENGTH > 2:   
            reward = -50
            done = True
            self.SNAKE.kill()
            info = 1
        
        # check if it ate food
        if new_head_x == self.FOOD_X and new_head_y == self.FOOD_Y:
            self.SNAKE.eat_food(self.FOOD_X, self.FOOD_Y)  #make it eat food
            reward = 50                                                         #give him a reward for eating food                             
            food_eaten = True
            self.FOOD_X, self.FOOD_Y = self.get_randoms()
            info = 2
    
        x1, x2, y1, y2 = self.get_boundaries()
        if new_head_x < x1 or new_head_x > x2 or new_head_y < y1 or new_head_y > y2:
            reward = -50
            done = True
            self.SNAKE.kill()
            info = 3
        
        # update the snake's position
        if not done and not food_eaten:
            self.SNAKE.update(x_change, y_change)
          
        if self.SNAKE.MOVES == 0:   #natural death, so no indication by coloring the head red
            done = True
            info = 4
        
        boundaries = [x1, x2, y1, y2]
        self.STATE =  self.SNAKE.look(self.FOOD_X, self.FOOD_Y, boundaries)

        return self.STATE, reward, done, info
        
         
