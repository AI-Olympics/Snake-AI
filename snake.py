from collections import deque
import numpy as np 
import time
import random

#colours - b, g, r
WHITE = (255,255,255)
BLUE = (255,0,0)
RED = (0,0,255)
YELLOW = (255,255,0)
BODY_COLOR = (100,100,100)

class Snake:

    def __init__(self, x_start, y_start, env_width, env_height):

        self.ENV_WIDTH = env_width
        self.ENV_HEIGHT = env_height
        
        #generating random directions for initiating the snake
        a = random.randint(0,3)
        init_dir = 0  #short for initial direction
        if a == 0:
            self.X = deque([x_start, x_start-1, x_start -2, x_start-3])
            self.Y = deque([y_start, y_start, y_start, y_start])
            init_dir = 1
        elif a == 1:
            self.X = deque([x_start, x_start+1, x_start+2, x_start+3])
            self.Y = deque([y_start, y_start, y_start, y_start])
            init_dir = 0
        elif a == 2:
            self.Y = deque([y_start, y_start-1, y_start-2, y_start-3])
            self.X = deque([x_start, x_start, x_start, x_start])
            init_dir = 3
        else:
            self.Y = deque([y_start, y_start+1, y_start+2, y_start+3])
            self.X = deque([x_start, x_start, x_start, x_start])
            init_dir = 2
        
        self.is_alive = True 
        self.MOVES = max(100, env_width*env_height*2)
        self.LENGTH = 4
        self.INITIAL_DIRECTION = init_dir

#----------------------------------------------Draw the snake on the matrix-----------------------------------#
    def draw(self, env):

        for body_y, body_x in zip(self.Y, self.X):
            env[body_y, body_x] = BODY_COLOR

        #coloring the head diff color
        head_x, head_y = self.head_pos()
        env[head_y, head_x] = BLUE if self.is_alive else YELLOW 
        return env 
            
#---------------------------------------------snake eat's food-----------------------------------------------#
    
    def eat_food(self, x_food, y_food):
        self.X.appendleft(x_food) 
        self.Y.appendleft(y_food)
        self.LENGTH += 1
        self.MOVES += 100

#--------------------------------------------move the snake forward--------------------------------------------#
        
    def update(self, x_change, y_change):
        # updating snake's body position
        for i in range(self.LENGTH-1,0,-1):
            self.X[i] = self.X[i-1]
            self.Y[i] = self.Y[i-1]

        #updating head position
        self.X[0] = self.X[0] + x_change
        self.Y[0] = self.Y[0] + y_change

        self.MOVES -= 1

#-------------------------------------------method to check if snake bit itself---------------------------------------#

    def bit_itself(self):
        for i in range(1,self.LENGTH):
            if self.X[0] == self.X[i] and self.Y[0] == self.Y[i]:
                return True
        return False

#-------------------------------------------check if given position is on snake's body---------------------------------#
    
    def is_on_body(self,check_x, check_y, remove_last = True):
        X, Y = self.X.copy(), self.Y.copy()
        
        if remove_last:    #don't consider the last tail part
            X.pop()
            Y.pop()
        for x,y in zip(X,Y):
            if x == check_x and y == check_y:
                return True
        return False

#---------------------------------------------------some helpful methods--------------------------------------#

    def head_pos(self):
        return self.X[0], self.Y[0]

    def kill(self):
        self.is_alive = False
    
    def set_length(self, length):
        self.LENGTH = length
    

#----------------------------------------------look in all directions--------------------------------------------#
    
    def look(self, x_food, y_food, boundaries):
        #we will look in all directions in a clockwise manner
        #look up
        up         = self.lookInDirection(boundaries, y=-1, x=0)
        #look up/right
        up_right   = self.lookInDirection(boundaries, y=-1, x=1)
        #look right
        right      = self.lookInDirection(boundaries, y=0, x=1)
        #look down/right
        down_right = self.lookInDirection(boundaries, y=1, x=1)
        #look down
        down       = self.lookInDirection(boundaries, y=1, x=0)
        #look down/left
        down_left  = self.lookInDirection(boundaries, y=1, x=-1)
        #look left
        left       = self.lookInDirection(boundaries, y=0, x=-1)
        #look up/lefts
        up_left    = self.lookInDirection(boundaries, y=-1, x=-1)
        
        head_x, head_y = self.head_pos()
        food_position = np.array([head_x - x_food, head_y - y_food])
        return np.hstack((food_position, up,up_right,right, down_right, down, down_left, left, up_left))
        #hence an array of size 2*8 + 2 = 18
	

#-----------------------------------------------look in specific direction--------------------------------------------#

    def lookInDirection(self,boundaries, y, x):
        tail_distance = 0   # 0 if tail is not in this direction
        distance = 1 #starting with 1 as it is the min distance of wall

        curr_x, curr_y = self.head_pos()
        check_x, check_y = curr_x + x, curr_y + y #look one step furthe in the direction
        x1, x2, y1, y2 = boundaries

        while check_y >= y1 and check_y <= y2 and check_x >= x1 and check_x <= x2:

            if tail_distance==0 and (self.is_on_body(check_x, check_y, remove_last = False)):
                tail_distance = (1/distance)

            #continue looking
            check_y += y
            check_x += x
            
            distance += 1
		
        return np.array([1/distance, tail_distance])
    
    