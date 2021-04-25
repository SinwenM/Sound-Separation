"""
This file contains all constant that are needed for the project  

"""
# Sound Processing constants
sample_rate = 22050
window_size = 1024
hop_length = 256
sample_length = 25
propotion = 0.7

#  Neural network constant
split_ratio = 0.7

validation_split = 0.1
epochs = 15
batch_size = 32

# Paths for the Raw data/ processed Data / model weights and song to apply the model on
data_mixtures_path = "/Users/sinwenm/Documents/GM5/PFE/DATA/Mixtures"
data_vocals_path = "/Users/sinwenm/Documents/GM5/PFE/DATA/Sources"

mixtures_path_processed = " "
vocals_path_processed = " "

model_weights = "/Users/sinwenm/Documents/Python_Project/Github/Sound_Separation/Final_1_cnn2d"

song = "/Users/sinwenm/Documents/Python_Project/Github/Sound_Separation/test_song.wav"
test_song = "/Users/sinwenm/Documents/Python_Project/Github/Sound_Separation/test_song.wav"
test_song2 = "/Users/sinwenm/Documents/GM5/PFE/DATA/Mixtures/Test/002 - ANiMAL - Rockshow/mixture.wav"

