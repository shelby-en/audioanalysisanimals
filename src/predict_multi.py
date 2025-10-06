from nn_multiout import load_data, config_model, nClasses, ConvTest, predTransforms

# from nn_spectrogram import setup_model, nClasses
from preprocessData_multi import resize_audio

import torch
# import os
import pandas as pd
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader, random_split
# import torchvision.transforms.functional as TF
from scipy.signal import butter, sosfilt

import sklearn.metrics as skm

import numpy as np

import librosa
import matplotlib.pyplot as plt

PI = 3.14159265358

class Predictor():
    sampleLen = 5

    def __init__(self, savePath, classFile):
        self.model = self.setup(savePath)
        self.labels = pd.read_csv(classFile, index_col=0, header=None)
        self.threshold = 0.5

        self.filt = butter(15, [0.04, 0.45], btype="bandpass", analog=False, output="sos")
        # print(self.labels)

    def setup(self, savePath):
        model = ConvTest.load_from_checkpoint(savePath, map_location="cpu")
        model.eval()
        # model, _, _ = config_model(ConvTest, None, None)
        # model.load_state_dict(torch.load(savePath))

        return model
    
    def audioFilter(self, audio):
        return sosfilt(self.filt, audio)
        
    def process_sample(self, sampleData):
        y = resize_audio(sampleData, self.sampleLen)
        yfilt = self.audioFilter(y)
        # D = np.abs(librosa.stft(yfilt))
        D = np.abs(librosa.stft(y))
        # librosa.display.specshow(librosa.amplitude_to_db(D))
        # plt.show()
        D = predTransforms(D)
        # print(len(D), len(D[0]))
        return D

    def predict(self, sample, audio=True):
        if audio:
            data = self.process_sample(sample).unsqueeze(0)
        else:
            data = predTransforms(sample).unsqueeze(0)
        # print(data.shape)

        data = (data, 0)

        self.model.eval()
        with torch.no_grad():
            out, x = self.model.predict_step(data)
        # out = torch.softmax(out, 1)
        out = out.numpy().squeeze()
        # print(out)

        labels = []
        for i in range(len(out)):
            if out[i] > self.threshold:
                labels.append(self.labels.iloc[i][1])
            
        return labels

if __name__ == "__main__":
    version = 23
    pred = Predictor(f'./data/chkpts/lightning/chks/version_{version}.ckpt', './data/classes.csv')
    # print("start predicting..")
    # nums = [338,340,341]
    # for i in nums:
    #     y1, sr = librosa.load('new.mp3', sr=48000)
    #     # D = np.load(f"./data/mixed_set/{i}.npy")

    #     # print(f"{i}: {pred.predict(D, audio=False)}")
    #     print(pred.predict(y1))

    y1, sr = librosa.load('new.mp3', sr=48000)
    offset = 0
    while offset + 48000 * 5 < len(y1):
        print(pred.predict(y1[offset:]))
        offset += 48000 * 1

