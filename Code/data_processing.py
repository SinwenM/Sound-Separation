import sys
import os 
import librosa
import IPython.display
import numpy as np
import math 
import constant as cs


class DataProcessing():
    """
    This Class contains all functions necessary to process sound 
    files for the neural network.
    """
    def __init__(self):
        pass
   

    def mixture_paths(self, path=cs.data_mixtures_path):
        """
        Each song is inside a file and this function returns all the paths for each song. 

        Args:
            path ([String]): Path to Mixter's files. Defaults to cs.data_mixtures_path.

        Returns:
            [tuple]: A tuple of all mixture's wav files path in the mixture's file
        """
        paths = []
        for subdir, dirs, files in os.walk(path):
            for file in files:
                filepath = subdir + os.sep + file
                if filepath.endswith(".wav"):
                    paths.append(filepath)
        return tuple(paths)
    
    def vocals_paths(self, path=cs.data_vocals_path):
        """
        Each song's vocal is inside a file and this function returns all the paths to it. 

        Args:
            path ([String]): Path for the Vocal's files. Defaults to cs.data_vocals_path.

        Returns:
            [tuple]: A tuple of all vocal's wav files path.
        """

        paths = []
        for subdir, dirs, files in os.walk(path):
            for file in files:
                filepath = subdir + os.sep + file
                if filepath.endswith("vocals.wav"):
                       paths.append(filepath)
        return tuple(paths)

    def clean_paths(self, mixtures_paths, vocals_paths):
        """
        In this Data set, we are missing some vocals or mixtures and this function 
        cleans paths returned by vocals_paths and mixture_paths to have the same
        mixtures and vocals.

        Args:
            mixtures_paths (Tuple)
            vocals_paths (Tupe)

        Returns:
            Tuple: cleaned mixture's paths
            Tuple: cleaned vocals's paths
        """
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
        This function is a warapper to the librosa function 

        Args:
            song_path ([String]): Path to a wav file

        Returns:
            [Numpy array, Int]: this function will return an audio timeseries and a Sampling rate of it
        """

        data, _ = librosa.load(song_path, sr=cs.sample_rate, mono=True)
        return data

    def compute_stft(self, raw_wave):
        """
        Compute the Short-time Fourier transform of a raw sound file.

        Args:
            raw_wave ([type]): [description]

        Returns:
            np.array: A complex-valued matrix D such as np.abs(D[f, t]) is the magnitude of frequency bin f at frame t
        """
        return librosa.stft(raw_wave, cs.window_size, hop_length=cs.hop_length)
    
    def compute_amplitude(self, stft):
        """
        Given the STFT of a sound we compute the amplitude needed to get the spectrogram.
        Args:
            stft ([np.ndarray])
        Returns:
            ([np.ndarray])
        
        """
        return librosa.power_to_db(np.abs(stft)**2)

    def split_spectogram(self, amplitude, sample_length = cs.sample_length):
        """
        Given the amplitude we compute and generate a spectogram for the sound (an image representation)

        Args:
            amplitude ([np.ndarray]): 
            sample_length ([type]): size of a sample. Defaults to cs.sample_lenth.

        Returns:
            [type]: [description]
        """
        slices =[]
        for i in range(0, amplitude.shape[1]//sample_length):
            _slice = amplitude[:, i*sample_length: (i+1)*sample_length]
            slices.append(_slice)
        return slices

    def labeling(self, amplitude, sample_length=cs.sample_length):
        """
        We create a mask for vocal where 1 is a pixel containing vocals and 0 no.

        Args:
            amplitude ([type])
            sample_length ([type]): size of a sample. Defaults to cs.sample_lenth.

        Returns:
            [tuple]: a mask of 1 and 0.
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
            amplitude ([type])
            length (int, optional): sample lenght. Defaults to cs.sample_length.

        Returns:
            [List]: a list of 
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
        This function invers a mask the 1's become 0 and 0 become 1.
        

        Args:
            mask ([np.array])

        Returns:
            [np.array]
        """
        return np.where((mask==0)|(mask==1), mask^1, mask)
    
    def apply_binary_mask(self, stft, mask):
        """
        [summary]

        Args:
            stft ([np.array])
            mask ([np.array])

        Returns:
            [np.array]
        """
        return np.multiply(stft, mask)

    def reverse_stft(self, stft):
        """
        This Function Converts a complex-valued spectrogram stft_matrix to time-series.

        Args:
            stft ([np.array])

        Returns:
            [np.ndarray [shape=(n,)]]
        """
        return librosa.istft(stft, cs.hop_length, cs.window_size)



    def make_mixture_data_cnn(self, paths,name="spectograms"):
        """
        Process mixtures data for our model.

        Args:
            paths ([tuple]): list of all mixtures's paths.
            name (str): A name for the processed data file. Defaults to "spectograms".
        """
        data_arrays = []
        for path in paths:
            wave = self.get_raw_wave(path)
            stft = self.compute_stft(wave)
            amplitude = self.compute_amplitude(stft)
            spectogram_slices = self.split_spectogram(amplitude=amplitude, sample_length=cs.sample_length)
            # reshape sclices to feed to cnn
            spectogram_slices = np.array(spectogram_slices).reshape(len(spectogram_slices),
                                                                    len(spectogram_slices[0]),
                                                                    len(spectogram_slices[0][0]),
                                                                    1)
            data_arrays.append(spectogram_slices)
        
        np.save("./mixtures_%s.npy" % name, np.vstack(data_arrays))

            
    def make_vocal_data_cnn(self, paths,name="spectograms"):
        """
         Process vocals data for our model.
        Args:
            paths ([tuple]): list of all vocals's paths.
            name (str): A name for the processed data file. Defaults to "spectograms".
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

