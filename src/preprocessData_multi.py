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
    folder = "./data/overlapping/"
    featherPath = "./data/spectrograms_ol/"
    # print(os.getcwd())
    files = []
    for file in os.listdir(folder):
        if file.endswith(".wav"):
            files.append(file)

    # print(files)
    header = ["fileName","Bat","Cockatoo","Crocodile","Dingo","Duck","Frog","FrogmouthTawny","Koala","Kookaburra","Magpie","Platypus","Possum","Snake","Wombat"]
    df = pd.DataFrame(columns = header)

    for i, file in enumerate(files):
        y, sr = librosa.load(folder + '/' + file, sr=48000)
        y = resize_audio(y, 5)
        
        #discard complex part for now
        D = np.abs(librosa.stft(y))
    
        filename = file.split('.')[0]
        np.save(featherPath + filename + '.npy', D)
        
        c1, c2 = filename.split('_')[2:]
        c1 = int(c1)
        c2 = int(c2)
        df.loc[i] = [filename] + [1 if c1 == i or c2 == i else 0 for i in range(len(header) - 1)]

    df.to_csv("./data/multi_labels_ol.csv", index=False)

