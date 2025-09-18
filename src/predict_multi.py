from nn_multiout import load_data, config_model, nClasses, ConvTest, predTransforms

# from nn_spectrogram import setup_model, nClasses
from preprocessData_multi import resize_audio

import torch
# import os
import pandas as pd
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader, random_split
# import torchvision.transforms.functional as TF
from scipy.signal import butter

import numpy as np

import librosa
# import matplotlib.pyplot as plt

class Predictor():
    sampleLen = 5

    def __init__(self, savePath, classFile):
        self.model = self.setup(savePath)
        self.labels = pd.read_csv(classFile, index_col=0, header=None)
        self.threshold = 0.2

        self.filt = butter(10, [0.05, 0.4], btype="bandpass", analog=True, output="sos")
        # print(self.labels)

    def setup(self, savePath):
        model = ConvTest.load_from_checkpoint(savePath, map_location="cpu")
        model.eval()
        # model, _, _ = config_model(ConvTest, None, None)
        # model.load_state_dict(torch.load(savePath))

        return model
    
    def audioFilter(self, audio):
        return scipy.signal.sosfilt(self.filt, audio)
        
    def process_sample(self, sampleData):
        y = resize_audio(sampleData, self.sampleLen)
        yfilt = self.audioFilter(y)
        D = np.abs(librosa.stft(yfilt))
        # D = predTransforms(D)
        # print(len(D), len(D[0]))
        return D

    def predict(self, sample):
        data = self.process_sample(sample).unsqueeze(0)
        # print(data.shape)

        self.model.eval()
        with torch.no_grad():
            out, x = self.model.predict_step(data)
        out = torch.softmax(out, 1)
        out = out.numpy().squeeze()

        labels = []
        for i in range(len(out)):
            if out[i] > self.threshold:
                labels.append(self.labels.iloc[i][1])
            
        return labels

if __name__ == "__main__":
    version = 1
    y1, sr = librosa.load('./data/augmented_mp3s/BatB_4.wav', sr=48000)
    # y2, sr = librosa.load('./data/augmented_mp3s/WombatA_2.wav', sr=48000)
    # y = y1 + y2
    y = y1
    # pred = Predictor('./data/chkpts/test92.pt', './data/classes.csv')
    pred = Predictor(f'./data/chkpts/lightning/chks/version_{version}.ckpt', './data/classes.csv')
    spec = pred.processSample(y)
    plt.plot(spec)