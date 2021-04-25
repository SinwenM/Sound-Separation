import os 
import sys 
import numpy as np 
import  math

import librosa
import constant as cs 
from data_processing import DataProcessing 
from model import SoundSeparationModel 

def process_data():
    """
    This  function will be called only when the user needs to process the data for the neural  network
    """
    dp  = DataProcessing()
    mix, voc =dp.clean_paths(mixtures_paths=dp.vocals_paths(path=cs.data_vocals_path),
                             vocals_paths=dp.mixture_paths(path=cs.data_mixtures_path))
    
    dp.make_mixture_data_cnn(paths=mix[1:3],name="spectograms")
    dp.make_vocal_data_cnn(paths=voc[1:3],name="spectograms")


def train_model():
    """
    This function trains the model and print the evaluation on the test set. 
    """
    ssm = SoundSeparationModel()

    mixture_Data = np.load(cs.mixtures_path_processed)
    vocals_Data = np.load(cs.vocals_path_processed)

    
    
    split_index = math.ceil(cs.split_ratio * len(mixture_Data))

    mixture_training = mixture_Data[:split_index]
    vocals_training = vocals_Data[:split_index]

    mixture_test = mixture_Data[split_index:]
    vocals_test = vocals_Data[split_index:]

    ssm.train(mixture_training, vocals_training, cs.validation_split, cs.epochs, cs.batch_size)

    print(ssm.evaluate(mixture_test=mixture_test, vocals_test=vocals_test))

def separate_sound():
    """
    This function will be called if the user wants to apply the model,
    beforehand, they will need to specify the song path in the constant file
    The result will be saved in a vocals.wav file   
    """

    ssm = SoundSeparationModel()

    ssm.load_weights(cs.model_weights)
    vocal = ssm.isolate(cs.song)

    librosa.output.write_wav("vocals.wav", vocal, cs.sample_rate, norm=False)





if __name__ == "__main__":
   
    print("############################################")
    print("################# Main #####################")
    print("############################################\n\n")

    print("Select the task you want to run \n"+
           "\t 1: Process Data for CNN\n" +
           "\t 2: Train The model\n" +
           "\t 3: Separate Vocals from Instruments\n" +
           "\t 4: Quit\n\n")

    arg = int(input("Enter your value: "))

    while True:
        if arg  == 1:
            process_data()
            break

        elif arg == 2:
            train_model()
            break

        elif  arg  == 3:
            separate_sound()
            break

        elif arg == 4:
            quit()
        else:
            print("Invalid Value, Try again\n")

            print("Select the task you want to run \n"+
                "\t 1: Process Data for CNN\n" +
                "\t 2: Train The model\n" +
                "\t 3: Separate Vocals from Instrument\n" +
                "\t 4: Quit\n\n")

            arg = int(input("Enter your value: "))
