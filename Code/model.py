# 
import sys
import os 
import math
from keras.backend.cntk_backend import ConvertToBatch
from librosa.core.fft import set_fftlib
import numpy as np
from numpy.core.numeric import NaN 
import global_variables as gv
from data_processing import DataProcessing as dp

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Activation
from keras.optimizers import SGD
from keras.layers.advanced_activations import LeakyReLU


class SoundSeparationModel():

    def __init__(self, model=NaN):
        self.model = model

    def model(self):
        """[summary]

        Returns:
            [CNN Model]: []
        """
        model = Sequential()

        bins = math.ceil(gv.window_size/2) + 1

        model.add(Conv2D(32, (3,3), padding="same", input_shape=(bins, gv.sample_length,1), activation="relu"))
        model.add(Conv2D(16, (3,3), padding="same", activation="relu"))   

        model.add(MaxPooling2D(pool_size=(2,2))) 

        model.add(Conv2D(64, (3,3), padding="same", activation='relu'))
        model.add(Conv2D(32, (3,3), padding="same", activation='relu'))

        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(bins, activation='sigmoid'))


        model.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])

        model.summary()

        self.model= model


    def train(self, mixture_training, vocals_training, mixture_test, vocals_test, epochs, validation_split, batch_size):
        """


        Args:
            mixture_training ([Numpy array]): [description]
            vocals_training ([Numpy array]): [description]
            mixture_test ([Numpy array]): [description]
            vocals_test ([Numpy array]): [description]
            epochs ([Integer]): [description]
            validation_split ([Float]): [description]
            batch_size ([Integer]): [description]
        """
        
        self.model.fit(mixture_training, vocals_training, validation_split=validation_split,  epochs=epochs, batch_size=batch_size)
        print(self.model.evaluate(mixture_test, vocals_test))
        self.model.save_weights(gv.model_weights, overwrite=True)
    
    def load_weights(self, path=gv.model_weights):
        """[summary]

        Args:
            path ([type], optional): [description]. Defaults to gv.model_weights.

        Returns:
            [type]: [description]
        """

        self.model.load_weights(path)
        return m

    def isolate(self, mixture_path):
        """[summary]

        Args:
            mixture_path ([type]): [description]

        Returns:
            [type]: [description]
        """
        mixture_wa = dp.get_raw_wav(mixture_path)
        mixture_st = dp.compute_stft(mixture_wa)
        mixture_amplitude = dp.compute_amplitude(mixture_st)
    
        split_x = np.array(dp.sliding_window(mixture_amplitude,length=gv.sample_length))
        split_x = split_x.reshape(len(split_x), len(split_x[0]), len(split_x[0][0]), 1)

        prediction = self.model.predict(split_x)
        prediction = np.transpose(prediction)
    
        accompaniment = np.zeros(np.shape(prediction))
        for x in range(0, len(prediction)):
            for y in range(0, len(prediction[x])):
                prediction[x][y] = 1 if prediction[x][y] > 0.45 else 0 
                accompaniment[x][y] = 0 if prediction[x][y] > 0.4 else 1

        vocal = dp.apply_binary_mask(mixture_amplitude, prediction)
        vocal = dp.reverse_stft(vocal)
    
        melody = dp.apply_binary_mask(mixture_amplitude, accompaniment)
        melody = dp.reverse_stft(melody)
    
    
        return vocal, melody

    

if __name__=="__main__":
    pass 