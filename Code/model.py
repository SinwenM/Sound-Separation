# 
import sys
import os 
import math
import numpy as np

from librosa.core.fft import set_fftlib


import constant as cs
from data_processing import DataProcessing 

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Activation

#from numpy.core.numeric import NaN 
#from keras.backend.cntk_backend import ConvertToBatch


class SoundSeparationModel():

    def __init__(self):
        """
        CNN-2D model for sound separation.


        """

        model = Sequential()
        bins = math.ceil(cs.window_size/2)+1
    
        model.add(Conv2D(32, (3,3), padding= "same", input_shape=(bins, cs.sample_length,1),activation='relu'))
        model.add(Conv2D(32, (3,3), padding="same", activation='relu'))
    
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.5))
    
        model.add(Conv2D(64, (3,3), padding="same", activation='relu'))
        model.add(Conv2D(64, (3,3), padding="same", activation='relu'))
    
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.5))
    
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(bins, activation='sigmoid'))
    

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


        self.model =  model


    def train(self, mixture_training, vocals_training, epochs, validation_split, batch_size):
        """
        This function train the model on the training set and saves the model's weights

        Args:
            mixture_training ([Numpy array]): Processed mixture's data training part
            vocals_training ([Numpy array]): Processed vocals's data training part
            epochs ([Integer]): number of epochs
            validation_split ([Float]): the ratio of train and validation
            batch_size ([Integer]): size of the batch during the training phase
        """
        
        self.model.fit(mixture_training, vocals_training, validation_split=validation_split,  epochs=epochs, batch_size=batch_size)
        self.model.save_weights(cs.model_weights, overwrite=True)
    
    def evaluate(self, mixture_test, vocals_test):
        """
        This function evaluate our model on the test set

        Args:
            mixture_test ([Numpy array]): Processed mixture's data testing part
            vocals_test ([Numpy array]): Processed vocals's data testing part

        Returns:
            [type]: [description]
        """

        return self.model.evaluate(mixture_test, vocals_test)
    
    def load_weights(self, path=cs.model_weights):
        """

        Args:
            path ([string]): Path to the model weights. Defaults to cs.model_weights.

        Returns:
            
        """

        self.model.load_weights(path)
        

    def isolate(self, mixture_path):
        """
        Given a song this function will apply the model and retunr the vocals

        Args:
            mixture_path ([String]): Path to the song that we won't to separate the vocals
                                    from the instrumeents.

        Returns:
            [np.array]: Time series of the vocals
        """
        dp =  DataProcessing()

        mixture_wa = dp.get_raw_wave(song_path=mixture_path)
        mixture_st = dp.compute_stft(mixture_wa)
        mixture_amplitude = dp.compute_amplitude(mixture_st)
    
        split_x = np.array(dp.sliding_window(mixture_amplitude,length=cs.sample_length))
        split_x = split_x.reshape(len(split_x), len(split_x[0]), len(split_x[0][0]), 1)

        prediction = self.model.predict(split_x)
        prediction = np.transpose(prediction)
    
        
        for x in range(0, len(prediction)):
            for y in range(0, len(prediction[x])):
                prediction[x][y] = 1 if prediction[x][y] > 0.65 else 0 


        vocal = dp.apply_binary_mask(mixture_amplitude, prediction)
        vocal = dp.reverse_stft(vocal)
    
    
        return vocal

    

if __name__=="__main__":
    pass 