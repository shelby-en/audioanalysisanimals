from nn_multiout import load_data, config_model, nClasses, ConvTest, predTransforms

from newDataset import classes

# from nn_spectrogram import setup_model, nClasses
from preprocessData_multi import resize_audio

import torch
# import os
import pandas as pd
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader, random_split
# import torchvision.transforms.functional as TF

import sklearn.metrics as skm

import numpy as np

import librosa
import matplotlib.pyplot as plt

savePath = "./data/chkpts/lightning/chks/version_4.ckpt"
labelPath = "./data/mixed_labels.csv"
dataPath = "./data/mixed_set/"
device = "cuda"

model = ConvTest.load_from_checkpoint(savePath, map_location=device)
model.eval()

_, trainer, _ = config_model(ConvTest, './data/chkpts/lightning/logs/', './data/chkpts/lightning/chks/',max_epochs=100)

ds, _, loader, _ = load_data(labelPath, dataPath, [0, 1, 0])

print(len(ds))

ytrue = [val[1] for val in ds]
results = trainer.predict(model, loader)
ypreds = np.concatenate([res[0].numpy() for res in results], axis=0)

confusion = skm.multilabel_confusion_matrix(ytrue, ypreds)

labels = pd.read_csv("./data/classes.csv", index_col=0, header=None)

for i in range(len(confusion)):
    disp = skm.ConfusionMatrixDisplay(confusion_matrix=confusion[i], display_labels=["True", "False"])
    disp.plot()
    disp.ax_.set_title(f'{labels.iloc[i][1]}')
    plt.savefig(f'{labels.iloc[i][1]}.jpg')

