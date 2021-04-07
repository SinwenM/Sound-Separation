import sys
import os 
import librosa
import IPython.display
import numpy as np
import math 
import global_variables as gv


class DataProcessing():
    """
    This Class contain all functions necessary to process sound files 
    for the neural network in addition to a collection of functions for p
    """
    def __init__(self):
        pass

    def mixture_paths(self, path=gv.data_mixtures_path):
        """
        Each song is inside a file and , this function returns all the paths to 

        Args:
            path ([String]): [description]. Defaults to gv.data_mixtures_path.

        Returns:
            [tuple]: [A tuple of all mixtures wav files path in the mixtures file]
        """
        paths = []
        for p, dir, files in os.walk(path):
            for name in [f for f in files if f.endswith(".wav")]:
                paths.append(os.path.join(path, name))
        return tuple(paths)
    
    def vocals_path(self, path=gv.data_vocals_path):
        """[summary]

        Args:
            path ([String]): [path for the vocals files]. Defaults to gv.data_vocals_path.

        Returns:
            [tuple]: [A tuple of all vocals wav files path in the song file ]
        """

        paths = []
        for p, dir, files in os.walk(path):
            for subdir in dir:
                paths.append(os.path.join(path, subdir), "vocals.wav")
        return tuple(path)


    def get_raw_wave(self, song_path):
        """[summary]

        Args:
            song_path ([String]): [Path to a wav file]

        Returns:
            [Numpy array, Int]: [this function will return an audio timeseries and a Sampling rate of it]
        """

        data, _ = librosa.load(song_path, sr=gv.sample_rate, mono=True)
        return data

    def compute_stft(self, raw_wave):
        """
        Compute tje Short-time Fourier transform

        Args:
            raw_wave ([type]): [description]

        Returns:
            np.array: [A complex-valued matrix D such as np.abs(D[f, t]) is the magnitude of frequency bin f at frame t]
        """
        return librosa.stft(raw_wave, gv.window_size, hop_length=gv.hop_length)
    
    def compute_amplitude(self, stft):
        """

        Args:
            stft ([type]): [description]
        """
        return librosa.power_to_db(np.abs(stft)**2)

    def split_spectogram(self, amplitude, sample_length = gv.sample_length):
        """[summary]

        Args:
            amplitude ([type]): [description]
            sample_length ([type], optional): [description]. Defaults to gv.sample_lenth.

        Returns:
            [type]: [description]
        """
        slices =[]
        for i in range(0, amplitude.shape[1]//sample_length):
            _slice = amplitude[:, i*sample_length: (i+1)*sample_length]
            slices.append(_slice)
        return tuple(slices)

    def labeling(self, amplitude, sample_length=gv.sample_length):
        """[summary]

        Args:
            amplitude ([type]): [description]
            sample_length ([type], optional): [description]. Defaults to gv.sample_length.

        Returns:
            [type]: [description]
        """
        slices = []
        for i in range(0, amplitude.shape[1] // sample_length):
            _slice = []
            for j in range(0, amplitude.shape[0]):
                if amplitude[j,i*length+(math.ceil(length/2) if length > 1 else 0)] > 0.5:
                    _slice.append(1)
                else:
                    _slice.append(0)
            slices.append(_slice)
        return tuple(slices)
    
    def sliding_window(self, amplitude, length= gv.sample_length):
        """[summary]

        Args:
            amplitude ([type]): [description]
            length ([type], optional): [description]. Defaults to gv.sample_length.

        Returns:
            [type]: [description]
        """
        height = amplitude.shape[0]
        amplitude = np.column_stack((np.zeros((height, math.floor(length/2))), amplitude))
        amplitude = np.column_stack((amplitude, np.zeros((height, math.floor(length/2)))))
        slices = []
        for x in range(math.floor(length/2), amplitude.shape[1] - math.floor(length/2)):
            length_before = x - math.floor(length/2)
            length_after = x + math.floor(length/2)
            slices.append(np.array(amplitude[:, length_before : (length_after + 1)]))
        return slices

    def invers_mask(self, mask):
        """[summary]

        Args:
            mask ([type]): [description]

        Returns:
            [type]: [description]
        """
        return np.where((mask==0)|(mask==1), mask^1, mask)
    
    def apply_binary_mask(self, stft, mask):
        """[summary]

        Args:
            stft ([type]): [description]
            mask ([type]): [description]

        Returns:
            [type]: [description]
        """
        return np.multiply(self, stft, mask)

    def reverse_stft(self, stft):
        """[summary]

        Args:
            stft ([type]): [description]

        Returns:
            [type]: [description]
        """
        return librosa.istft(stft, hopp_lenght, window_size)

    def play_music(self, song):
        """[summary]

        Args:
            song ([type]): [description]
        """
        IPython.display.Audio(song, rate=sample_rate)
    




        


if __name__ == "__main__":
    pass

