from nn_spectrogram import setup_model, nClasses
from preprocessData_alt import resize_audio

import torch
import os
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms.functional as TF

import numpy as np

import librosa
import matplotlib.pyplot as plt

class Predictor():
    sampleLen = 5

    def __init__(self, savePath, classFile):
        self.model = self.setup(savePath)
        self.labels = pd.read_csv(classFile, index_col=0, header=None)
        print(self.labels)

    def setup(self, savePath):
        model, crit, opt = setup_model(nClasses, torch.device('cpu'))
        model.load_state_dict(torch.load(savePath))

        return model
    
    def audioFilter(self, spectrogram):
        return spectrogram
        
    def process_sample(self, sampleData):
        y = resize_audio(sampleData, self.sampleLen)
        D = np.abs(librosa.stft(y))
        D = self.audioFilter(D)
        # print(len(D), len(D[0]))
        return torch.from_numpy(D.astype('float32')).reshape((1, 1, len(D), len(D[0])))

    def predict(self, sample):
        data = self.process_sample(sample)
        # print(data.size())
        with torch.no_grad():
            outputs = self.model(data)

            # _, predicted = torch.max(outputs, 1)
            # return self.labels.iloc[predicted.item()][1]
            return outputs

if __name__ == "__main__":
    y1, sr = librosa.load('BatB.mp3', sr=48000)
    # y2, sr = librosa.load('./data/augmented_mp3s/BatB_7.wav', sr=48000)
    # y = y1 + y2
    y = y1[240000:480001]
    pred = Predictor('./data/chkpts/test92.pt', './data/classes.csv')
    print(pred.predict(y))