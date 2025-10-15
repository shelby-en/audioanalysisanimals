from nn_multiout import load_data, config_model, nClasses, ConvTest, predTransforms

from noise_filtration_fast import build_noise_dict, denoise_with_nmf_component_mask
from new_filter import filter

# from nn_spectrogram import setup_model, nClasses

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

from scipy.io import wavfile

PI = 3.14159265358
INDENT = {
    "Bat": 11,
    "Cockatoo": 6,
    "Crocodile": 5,
    "Dingo": 9,
    "Duck": 10,
    "Frog": 10,
    "FrogmouthTawny": 0,
    "Koala": 9,
    "Kookaburra": 4,
    "Magpie": 8,
    "Platypus": 6,
    "Possum": 8,
    "Snake": 9,
    "Wombat": 8,
}

def resize_audio(audio, time=10):
    numSamples = 48000*time
    audio = audio[:numSamples]
    zero_pad = np.zeros((numSamples - len(audio)), dtype="int32")
    audio = np.concatenate((zero_pad, audio))
    return audio


class Predictor():
    sampleLen = 5

    def __init__(self, savePath, classFile, filt=False):
        self.model = self.setup(savePath)
        self.labels = pd.read_csv(classFile, index_col=0, header=None)
        self.threshold = 0.95
        self.filt = filt

        # self.filt = butter(15, [0.04, 0.45], btype="bandpass", analog=False, output="sos")

        if self.filt:
            # noise_files = [
            #     "./data/noise_files/cricket_only_speaker.wav",
            #     "./data/noise_files/Motor.wav",
            #     "./data/noise_files/running_water_speaker.wav",
            # ]
            # n_bases_list = [20, 6, 20]  # bases for cricket & water

            # self.sr = 48000
            # # Build noise dictionary
            # W_noise, sr, n_fft, hop = build_noise_dict(noise_files, n_bases_list, self.sr)
            # self.W_noise = W_noise
            pass
        # print(self.labels)

    def setup(self, savePath):
        model = ConvTest.load_from_checkpoint(savePath, map_location="cpu")
        model.eval()
        # model, _, _ = config_model(ConvTest, None, None)
        # model.load_state_dict(torch.load(savePath))

        return model
    
    def audioFilter(self, audio):
        denoised, _ = denoise_with_nmf_component_mask(
            audio,
            self.W_noise,
            self.sr,
            wav=False,
            n_animal=15,
            n_iter=250,
            n_fft=2048,
            hop=512,
            alpha_noise=0.9,       # attenuation strength for cricket
            n_cricket_bases=20,    # first N bases of W_noise reserved for cricket
            low_freq_cutoff=20,    # remove low-frequency noise below 20 Hz
            animal_gain=2          # amplify animal sounds
        )

        return denoised

    def otherFilter(self, audio, rate=48000):
        filt = np.array(filter(audio, "./data/noise_files/", rate=rate, audio=True))
        
        if filt.dtype != 'float32':
            dmax = np.iinfo(filt.dtype).max
        else:
            dmax = 1
        
        # print(dmax)
        
        return filt.astype('float32') / dmax
        
    def process_sample(self, sampleData):
        y = resize_audio(sampleData, self.sampleLen)
        # print(y)
        if self.filt:
            # yfilt = self.audioFilter(y)
            yfilt = self.otherFilter(y)
        else:
            yfilt = y
        # print(yfilt)
        # wavfile.write("test.wav", 48000, yfilt)
        D = np.abs(librosa.stft(yfilt))
        # D = np.abs(librosa.stft(y))
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
        print(out)

        labels = []
        for i in range(len(out)):
            # if out[i] > self.threshold:
            label = self.labels.iloc[i][1]
            confidence = int(round(float(out[i]), 2)*100)
            prediction = [f"{INDENT[label] * " "}{label.upper() if (out[i] > self.threshold) else label.lower()}: {" " if confidence < 10 else ""}", confidence, f"% [{(confidence//4) * "="}{(25 - (confidence//4)) * " "}] {"<---" if (out[i] > self.threshold) else ""}"]
            labels.append(prediction)
        labels = sorted(labels, key=lambda sublist: sublist[1])
        labels.reverse()
        return labels

if __name__ == "__main__":
    version = 35
    pred = Predictor(f'./data/chkpts/lightning/chks/version_{version}.ckpt', './data/classes.csv', filt=True)
    # print("start predicting..")
    # nums = [338,340,341]
    # for i in nums:
    #     y1, sr = librosa.load('new.mp3', sr=48000)
    #     # D = np.load(f"./data/mixed_set/{i}.npy")

    #     # print(f"{i}: {pred.predict(D, audio=False)}")
    #     print(pred.predict(y1))

    rate, y1 = wavfile.read("./data/sample_5s/CockatooA_5.wav")
    print(y1)
    print(pred.predict(y1, audio=True))
    # offset = 0
    # step = 1
    # while offset + 48000 * 5 < len(y1):
    #     offset += 48000 * step

