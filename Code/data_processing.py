import sys
import os 
import librosa
import IPython.display
import numpy as np
import math 
import constant as cs


class DataProcessing():
    """
    This Class contain all functions necessary to process sound files 
    for the neural network in addition to a collection of functions for p
    """
    def __init__(self):
        pass
   

    def mixture_paths(self, path=cs.data_mixtures_path):
        """
        Each song is inside a file and , this function returns all the paths to 

        Args:
            path ([String]): [description]. Defaults to cs.data_mixtures_path.

        Returns:
            [tuple]: [A tuple of all mixtures wav files path in the mixtures file]
        """
        paths = []
        for subdir, dirs, files in os.walk(path):
            for file in files:
                filepath = subdir + os.sep + file
                if filepath.endswith(".wav"):
                    paths.append(filepath)
        return tuple(paths)
    
    def vocals_path(self, path=cs.data_vocals_path):
        """
        Loop through the whole data sets and find the path for each vocal

        Args:
            path ([String]): [path for the vocals files]. Defaults to cs.data_vocals_path.

        Returns:
            [tuple]: [A tuple of all vocals wav files path in the song file ]
        """

        paths = []
        for subdir, dirs, files in os.walk("/Users/sinwenm/Documents/GM5/PFE/DATA/Sources"):
            for file in files:
                filepath = subdir + os.sep + file
                if filepath.endswith("vocals.wav"):
                       paths.append(filepath)
        return tuple(paths)

    def clean_paths(self, mixtures_paths, vocals_paths):
        vocals = []
        mixtures = []
        for mix, voc in zip(mixtures_paths, vocals_paths):
            mix_song = mix.split('/')[-2]
            voc_song = voc.split('/')[-2]
        
            if voc_song == mix_song:
                mixtures.append(mix)
                vocals.append(voc)
            
        return tuple(mixtures), tuple(vocals)

    def get_raw_wave(self, song_path):
        """
        Given a song we read it with python

        Args:
            song_path ([String]): [Path to a wav file]

        Returns:
            [Numpy array, Int]: [this function will return an audio timeseries and a Sampling rate of it]
        """

        data, _ = librosa.load(song_path, sr=cs.sample_rate, mono=True)
        return data

    def compute_stft(self, raw_wave):
        """
        Compute tje Short-time Fourier transform

        Args:
            raw_wave ([type]): [description]

        Returns:
            np.array: [A complex-valued matrix D such as np.abs(D[f, t]) is the magnitude of frequency bin f at frame t]
        """
        return librosa.stft(raw_wave, cs.window_size, hop_length=cs.hop_length)
    
    def compute_amplitude(self, stft):
        """
        Given the STFT of a sound we compute the amplitude need to get the spectogram
        Args:
            stft ([type]): [description]
        """
        return librosa.power_to_db(np.abs(stft)**2)

    def split_spectogram(self, amplitude, sample_length = cs.sample_length):
        """
        Given the amplitude we compute and generate a spectogram for the sound (an image representation)

        Args:
            amplitude ([type]): [description]
            sample_length ([type], optional): [description]. Defaults to cs.sample_lenth.

        Returns:
            [type]: [description]
        """
        slices =[]
        for i in range(0, amplitude.shape[1]//sample_length):
            _slice = amplitude[:, i*sample_length: (i+1)*sample_length]
            slices.append(_slice)
        return tuple(slices)

    def labeling(self, amplitude, sample_length=cs.sample_length):
        """
        We create a mask for vocal where 1 is a pixel containing vocals and 0 no

        Args:
            amplitude ([type]): [description]
            sample_length ([type], optional): [description]. Defaults to cs.sample_length.

        Returns:
            [type]: [description]
        """
        slices = []
        for i in range(0, amplitude.shape[1] // sample_length):
            _slice = []
            for j in range(0, amplitude.shape[0]):
                if amplitude[j,i*sample_length+(math.ceil(sample_length/2) if sample_length > 1 else 0)] > 0.5:
                    _slice.append(1)
                else:
                    _slice.append(0)
            slices.append(_slice)
        return tuple(slices)
    
    def sliding_window(self, amplitude, length= cs.sample_length):
        """
        

        Args:
            amplitude ([type]): [description]
            length ([type], optional): [description]. Defaults to cs.sample_length.

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
        """
        

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
        return np.multiply(stft, mask)

    def reverse_stft(self, stft):
        """[summary]

        Args:
            stft ([type]): [description]

        Returns:
            [type]: [description]
        """
        return librosa.istft(stft, hopp_lenght, window_size)



    def make_mixture_data_cnn(self, paths,name="spectograms"):
        """[summary]

        Args:
            paths ([type]): [description]
            name (str, optional): [description]. Defaults to "spectograms".
        """
        data_arrays = []
        for path in paths:
            wave = self.get_raw_wave(path)
            stft = self.compute_stft(wave)
            amplitude = self.compute_amplitude(stft)
            spectogram_slices = self.split_spectogram(sample_length, amplitude)
            # reshape sclices to feed to cnn
            spectogram_slices = np.array(spectogram_slices).reshape(len(spectogram_slices),
                                                                    len(spectogram_slices[0]),
                                                                    len(spectogram_slices[0][0]),
                                                                    1)
            data_arrays.append(spectogram_slices)
        
        np.save("./mixtures_%s.npy" % name, np.vstack(data_arrays))

            
    def make_vocal_data_cnn(self, paths,name="spectograms"):
        """[summary]

        Args:
            paths ([type]): [description]
            name (str, optional): [description]. Defaults to "spectograms".
        """
        data_arrays = []
        for path in paths:
            wave = self.get_raw_wave(path)
            stft = self.compute_stft(wave)
            amplitude = self.compute_amplitude(stft)
            labels_slices = self.labeling(amplitude)
        #for i in range(math.ceil(sample_length/2),len(labels_slices),math.ceil(sample_length/2)):
            data_arrays.append(labels_slices)
        
        np.save("./vocals_%s.npy" % name, np.vstack(data_arrays))

    




if __name__ == "__main__":
    pass

