#for gif making
import imageio 
import numpy as np
import os
from PIL import Image
from math import floor

def makeGif(savePath, outputPath, frame_every_X_steps = 1, repeat_ending = 15, start_frame = 0):
    number_files = len(os.listdir(savePath))-1 - start_frame - 1 #-1 for parms :)
    frame_every_X_steps = frame_every_X_steps
    repeat_ending = repeat_ending
    steps = np.arange(floor(number_files/frame_every_X_steps)) * frame_every_X_steps
    steps = steps + (number_files - np.max(steps))

    images = []
    for f in steps:
        filename = savePath + str(f+start_frame) + '.jpg'
        images.append(imageio.imread(filename))

    #repeat ending
    for _ in range(repeat_ending):
        filename = savePath + str(number_files+start_frame) + '.jpg'
        images.append(imageio.imread(filename))  
        
    imageio.mimsave(outputPath, images)