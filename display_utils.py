'''
 Collection of functions which help to show hyperparameters and other important info
'''

import numpy as np
from cv2 import putText, FONT_HERSHEY_COMPLEX
from PIL import Image


# colors - b, g, r

#colours - b, g, r
WHITE = (255,255,255)
BLUE = (255,0,0)
BLACK = (0,0,0)
GREEN = (0,255,0)
RED = (0,0,255)

# text properties
font = FONT_HERSHEY_COMPLEX


def add_info(width, height, env, action, length, stats, episode, epsilon, gamma, train = False):
    '''
        Damn helpful function to show 
        all the stuff going on while Training
    '''
    extra_cols = 12         #some extra columns to show all the stuff
    rows, cols, channels = env.shape
    display_matrix = np.ones((rows, cols+extra_cols, channels), dtype=np.uint8)*255    #white background
    display_matrix[:rows, 1:cols+1, :] = env 
    display_matrix = display_action(display_matrix, action, rows)
    
    img = Image.fromarray(display_matrix, 'RGB')
    scale = int(width/cols)
    img = np.array(img.resize((width + extra_cols*scale, height)))

    putText(img, f'Length : {length}', (width + scale, 70), font , 0.6, BLACK  , 1)  #show the length    
    if train:                                                                #show other info
        putText(img, f'Episode {episode}', (width + scale, 30), font , 0.8, BLUE, 2)
        putText(img, f'Total food : {stats[2]}', (width + scale, 100), font , 0.6, BLACK , 1)
        putText(img, 'Game Over -', (width + scale, 170), font, 0.6, BLACK, 1)
        putText(img, f'Self Bite : {stats[1]}', (width + scale, 200), font , 0.6, BLACK , 1)
        putText(img, f'boundary : {stats[3]}', (width + scale, 230), font , 0.6, BLACK  , 1)
        putText(img, 'Epsilon : {:.3f}'.format(epsilon), (width + scale, height - 60), font , 0.6, BLUE, 1)
        putText(img, f'Gamma : {gamma}', (width + scale, height-30), font , 0.6, BLUE, 1)
        
    return img

def display_action(matrix, action, max_height):
    '''
        Helpful function to display the keys
        which display the action taken by the agent
    '''
    mid_height = int(max_height/2)
    matrix[mid_height,-7,:] = BLUE   #left button
    matrix[mid_height,-9,:] = BLUE   #right button
    matrix[mid_height-1,-8,:] = BLUE  #up button
    matrix[mid_height+1,-8,:] = BLUE  #down button
    
    if action == 0:
        matrix[mid_height,-7,:] = RED   #left 
    elif action == 1:
        matrix[mid_height,-9,:] = RED   #right
    elif action == 2:
        matrix[mid_height-1,-8,:] = RED  #up 
    elif action ==3:
        matrix[mid_height+1,-8,:] = RED  #down 

    return matrix

