from NatureLM.models import NatureLM
from NatureLM.infer import Pipeline
from transformers import AutoConfig

# import torch
# import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import butter, sosfilt
import librosa

PI = 3.14159265358

def matched_filter(sample, known):
    pass

def resize_audio(audio, time=10, sr=16000):
    numSamples = 48000*time
    audio = audio[:numSamples]
    zero_pad = np.zeros((numSamples - len(audio)))
    audio = np.concatenate((zero_pad, audio))
    return audio

def downSample(audio, srFrom, srTo):
    ratio = srFrom//srTo

    return audio[0::ratio]

class Predictor():
    sampleLen = 8

    def __init__(self, savePath, labelPath):
        self.modelSr = 16000

        self.labels = pd.read_csv(labelPath, index_col=0)

        self.model = self.setup(savePath)
        print("model loaded")
        self.pipeline = Pipeline(model=self.model)
        print("pipeline initialised")

        self.filt = butter(15, [0.05, 0.4], btype="bandpass", analog=False, output="sos")

        print("initialisation complete")
        

    def setup(self, savePath):
        # config = AutoConfig.from_pretrained(savePath + "config.json")
        model = NatureLM.from_pretrained(savePath, local_files_only=True)
        model.eval().to("cuda")

        return model
    
    def audioFilter(self, audio):
        return sosfilt(self.filt, audio)
        
    def process_sample(self, sampleData, sr):
        print("processing sample")
        y = resize_audio(sampleData, time=self.sampleLen)
        y = downSample(y, srFrom=sr, srTo=self.modelSr)
        yfilt = self.audioFilter(y)
        # D = np.abs(librosa.stft(yfilt))
        return yfilt
    
    def prediction_to_class(self, predicitions):
        pass

    def predict(self, sample, sr):
        data = self.process_sample(sample, sr)
        queries = ["What is the common name/s for the specie/s in the audio? Answer:"]

        print("predicting...")
        return self.pipeline([data], queries, window_length_seconds=self.sampleLen, hop_length_seconds=self.sampleLen)

if __name__ == "__main__":
    version = 1
    y1, sr = librosa.load('../data/new_wavs_48000/BatA.wav', sr=48000)
    # y2, sr = librosa.load('./data/augmented_mp3s/WombatA_2.wav', sr=48000)
    # y = y1 + y2
    y = y1
    pred = Predictor(f'../data/model/version2_1/version2_1/', '../data/classes.csv')
    exit()
    # print(pred.predict(y, sr))