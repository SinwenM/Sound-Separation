import sys
import os 
import librosa
import numpy as np
import math 
import global_variables as gv


class DataProcessing():
    """
    This Class contain all functions necessary to process sound files 
    for the neural network in addition to a collection of functions for 
    """
    def __init__(self):
        pass

    def mixture_paths(self, path=gv.data_mixtures_path):
        """[summary]

        Args:
            path ([type], optional): [description]. Defaults to gv.data_mixtures_path.

        Returns:
            [type]: [description]
        """
        paths = []
        for p, dir, files in os.walk(path):
            for name in [f for f in files if f.endswith(".wav")]:
                paths.append(os.path.join(path, name))
        return tuple(paths)
    
    def vocals_path(self, path=gv.data_vocals_path):
        """[summary]

        Args:
            path ([type], optional): [description]. Defaults to gv.data_vocals_path.

        Returns:
            [type]: [description]
        """

        paths = []
        for p, dir, files in os.walk(path):
            for subdir in dir:
                paths.append(os.path.join(path, subdir), "vocals.wav")
        return tuple(path)


    def get_raw_wave(self, filename):
        """[summary]

        Args:
            filename ([type]): [description]

        Returns:
            [type]: [description]
        """

        data, _ = librosa.load(filename, sr=gv.sample_rate, mono=True)
        return data

    def compute_stft(self, raw_wave):
        """[summary]

        Args:
            raw_wave ([type]): [description]

        Returns:
            [type]: [description]
        """
        return librosa.stft(raw_wave, gv.window_size, hop_length=gv.hop_length)
    
    def compute_amplitude(self, stft):
        """[summary]

        Args:
            stft ([type]): [description]
        """
        return librosa.power_to_db(np.abs(stft)**2)
    




        


if __name__ == "__main__":
    print("import succes")
    pass

