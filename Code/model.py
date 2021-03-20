# 
import sys
import os 
import math
import numpy as np 
import global_variables as gv

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Activation
from keras.optimizers import SGD
from keras.layers.advanced_activations import LeakyReLU


class SoundSeparationModel():

    def __init__(self) -> None:
        pass

    def model(self):
        model = Sequential()

        bins = math.ceil(gv.window_size/2) + 1

        model.add(Conv2D(32, (3,3), padding="same", input_saphe=(bins, gv.sample_length,1), activation="relu"))
               
        return model

if __name__=="__main__":
    pass