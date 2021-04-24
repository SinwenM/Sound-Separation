import os 
import sys 
import numpy as np 


import global_variables as gv 
from data_processing import DataProcessing 
from model import SoundSeparationModel 

def process_data():
    dp  = DataProcessing()
    mix, voc =dp.clean_paths(mixtures_paths=dp.vocals_paths(path=cs.data_vocals_path),
                             vocals_paths=dp.mixture_paths(path=cs.data_mixtures_path))
    
    dp.make_mixture_data_cnn(paths=mix[1:3],name="spectograms")
    dp.make_vocal_data_cnn(paths=voc[1:3],name="spectograms")


def train_model():
    mixture_Data = np.load(gv.mixtures_path_processed)
    vocals_Data = np.load(gv.vocals_path_processed)

    # We split data into training and test sample, we can use shuffle_set on it
    
    split_index = math.ceil(gv.split_ratio * len(mixture_Data))
    mixture_training = mixture_Data[:split_index]
    vocals_training = vocals_Data[:split_index]

    mixture_test = mixture_Data[split_index:]
    vocals_test = vocals_Data[split_index:]

    sm.train(mixture_training, vocals_training, mixture_test, vocals_test, gv.validation_split, gv.epochs, gv.batch_size)
    print("train_model_h")

def separate_sound():
    print("separate_sound_h")




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
