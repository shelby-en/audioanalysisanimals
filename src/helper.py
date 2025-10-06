import os
import pandas as pd
import numpy as np
import librosa

from scipy.signal import butter, sosfilt
import scipy.io.wavfile as wavf
# folder = "./data/mp3s/"
# # print(os.getcwd())
# files = []
# for file in os.listdir(folder):
#     if file.endswith(".mp3"):
#             files.append(file[:-4])

# df = pd.DataFrame(files)
# df.to_csv('./data/labels.csv', index = False, header = False)

# with open('./data/labels.csv', 'r') as file:
#     with open('./data/classes.csv', 'w') as wfile:
#         prevName = ""
#         for line in file:
#             name, index = line.split(',')
#             name = name.strip().split('_')[0][:-1]
#             index = int(index)

#             if name == prevName:
#                 pass
#             else:
#                 prevName = name
#                 wfile.write(f'{index},{name}\n')

# with open('./data/labels.csv') as labelFile:
#     header = ["fileName","Bat","Cockatoo","Crocodile","Dingo","Duck","Frog","FrogmouthTawny","Koala","Kookaburra","Magpie","Platypus","Possum","Snake","Wombat"]
#     df = pd.DataFrame(columns = header)

#     # print(len(labelFile.readlines()))

#     for i, line in enumerate(labelFile):
#         name, label = line.strip().split(',')
#         df.loc[i] = [name] + [1 if int(label) == i else 0 for i in range(len(header) - 1)]

#         # print(df.iloc(i))

#     df.to_csv("./data/multi_labels_ol.csv", index=False)

# def downSample(audio, srFrom, srTo):
#     ratio = srFrom//srTo

#     return audio[0::ratio]

# folder = "./data/new_wavs_48000/"
# outPath = "./data/filt_wavs/"

# filt = butter(15, [0.05, 0.4], btype="bandpass", analog=False, output="sos")

# # print(os.getcwd())
# files = []
# for file in os.listdir(folder):
#     if file.endswith(".wav"):
#         files.append(file)

# for file in files:
#     y, sr = librosa.load(folder + file)
#     y = downSample(y, sr, 16000)
#     yfilt = sosfilt(filt, y)
#     wavf.write(outPath+file, 16000, yfilt)


# np.load("./data/mixed_set/1.npy")

folder = "./data/augmented_mp3s/"
outFolder = "./data/mixed_wavs/"

increment = 100

files = []
for file in os.listdir(folder):
    if file.endswith(".wav"):
        files.append(file)

for file in files:
    y, sr = librosa.load(folder + file)
    name, number = file.split('_')
    print(number[:-4])
    wavf.write(outFolder+f"{name}_{int(number[:-4]) + increment}.wav", 48000, y)

