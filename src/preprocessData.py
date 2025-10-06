import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os

import pyarrow as pa
import pandas as pd
import csv

def resize_audio(audio, time=10):
    numSamples = 48000*time
    audio = audio[:numSamples]
    zero_pad = np.zeros((numSamples - len(audio)))
    audio = np.concatenate((zero_pad, audio))
    return audio

if __name__ == "__main__":
    folder = "./data/mixed_wavs/"
    featherPath = "./data/more_spectrograms/"
    # print(os.getcwd())
    files = []
    for file in os.listdir(folder):
        if file.endswith(".wav"):
            files.append(file)

    # print(files)
    with open('./data/labels.csv', 'w') as labelFile:
        label = -1
        idx = 1
        prevLabel = ""
        for file in files:
            y, sr = librosa.load(folder + '/' + file, sr=48000)
            y = resize_audio(y, 5)
            
            #discard complex part for now
            D = np.abs(librosa.stft(y))
        
            np.save(featherPath + file.split('.')[0] + '.npy', D)
            
            if file.split("_")[0] != prevLabel:
                label += 1
                idx = 1
                prevLabel = file.split("_")[0]
            
            # labelFile.write(f'{idx},{label},{prevLabel}\n')
            idx += 1