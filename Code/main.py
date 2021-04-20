import os 
import sys 
import numpy as np 


# import global_variables as gv 
# from data_processing import DataProcessing as dp 
# from model import SoundSeparationModel as sm 

def process_data(a=0):
    print("process_data_H")


def train_model():
    print("train_model_h")

def separate_sound():
    print("separate_sound_h")

def main(arg):

    def invalid(): 
         print("Invalid Value")

    switcher = {
        1: process_data,
        2: train_model,
        3: separate_sound,
    }
    return switcher.get(arg, invalid())()


if __name__ == "__main__":
   
    print("############################################")
    print("################# Main #####################")
    print("############################################\n\n")

    print("Select the task you want to run \n"+
           "\t 1: Process Data for CNN\n" +
           "\t 2: Train The model\n" +
           "\t 3: Separate Vocals from Instrument\n" +
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
            print("Invalid Value, Try again")
            arg = int(input("Enter your value: "))
