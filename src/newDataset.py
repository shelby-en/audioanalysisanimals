import librosa
import os
import numpy as np
import pandas as pd

import soundfile as sf

classes = {"Bat": 0, "Cockatoo": 1, "Crocodile": 2, "Dingo": 3, "Duck": 4, "Frog": 5, "FrogmouthTawny": 6, "Koala": 7, "Kookaburra": 8, "Magpie": 9, "Platypus": 10, "Possum": 11, "Snake": 12, "Wombat": 13}

def resize_audio(audio, time=10):
    numSamples = 48000*time
    audio = audio[:numSamples]
    zero_pad = np.zeros((numSamples - len(audio)))
    audio = np.concatenate((zero_pad, audio))
    return audio

if __name__ == "__main__":
    # folder = "./data/more_spectrograms/"
    # outPath = "./data/more_multi/"
    # # labelPath = "./data/more_labels.csv"
    # folder = "./data/filtered_spectrograms/"
    # outPath = "./data/filt_ol/"
    # labelPath = "./data/filt_ol_labels.csv"
    folder = "./data/other_mic_spectrograms/"
    outPath = "./data/other_mic_ol/"
    labelPath = "./data/other_mic_labels.csv"

    files = []
    for file in os.listdir(folder):
        if file.endswith(".npy"):
            files.append(file)

    print(len(files))

    props = [1, 0.5, 0.00005]

    # noise = ["water","bugs","other_birds","mid_noise","low_noise"]
    noise = []
    header = ["fileName","Bat","Cockatoo","Crocodile","Dingo","Duck","Frog","FrogmouthTawny","Koala","Kookaburra","Magpie","Platypus","Possum","Snake","Wombat"] #+ noise

    df = pd.DataFrame(columns = header)

    # old_noise = pd.read_csv("./data/noise_labels_og.csv", header=0)
    # new_noise = pd.read_csv("./data/noise_labels_new.csv",header=0)

    # print(new_noise)

    # df.loc[i] = [filename] + [1 if c1 == i or c2 == i else 0 for i in range(len(header) - 1)]

    count = 1
    n = len(files)
    for i in range(n):
        # print(files[i])
        name = files[i][:-4].split('_')[0]
        name1 = files[i][:-4].split('_')[0][:-1]
        number = int(files[i][:-4].split('_')[1])

        c1 = classes[name1]

        # if number > 99:
        #     noises1 = old_noise.loc[old_noise["Filename"] == f"{name}_{number - 100}"]
        # else:
        #     noises1 = new_noise.loc[new_noise["Filename"] == f"{name}_{number}"]
        # noises1 = noises1[noise].to_numpy().flatten()
        # if len(noises1) == 0:
        #     noises1 = np.zeros((5,))

        # print(noises1)
        
        s1 = np.load(folder + files[i]).astype('float32')

        if np.random.uniform(0, 1) <= props[0]:
            np.save(outPath + f'{count}.npy', s1)
            # print()
            df.loc[count] = [count] + [1 if c1 == i else 0 for i in range(len(header) - 1 - len(noise))] #+ list(noises1)
            count += 1

        for j in range(i+1, len(files)):
            name = files[j][:-4].split('_')[0]
            name2 = files[j][:-4].split('_')[0][:-1]
            number = int(files[j][:-4].split('_')[1])

            c2 = classes[name2]

            # if number > 99:
            #     noises2 = old_noise.loc[old_noise["Filename"] == f"{name}_{number - 100}"]
            # else:
            #     noises2 = new_noise.loc[new_noise["Filename"] == f"{name}_{number}"]
            # noises2 = noises2[noise].to_numpy().flatten()
            # if len(noises2) == 0:
            #     noises2 = np.zeros((5,))

            # print(noises2)

            # print(noises1 + noises2)

            s12 = np.load(folder + files[j]).astype('float32') + s1

            if np.random.uniform(0, 1) <= props[1]:
                np.save(outPath + f'{count}.npy', s12)
                df.loc[count] = [count] + [1 if c1 == i or c2 == i else 0 for i in range(len(header) - 1 - len(noise))]# + list(np.logical_or(noises1,noises2))
                count += 1

            # print(count)

            for k in range(j+1, len(files)):
                if np.random.uniform(0, 1) <= props[2]:
                    name = files[k][:-4].split('_')[0]
                    name3 = files[k][:-4].split('_')[0][:-1]
                    number = int(files[k][:-4].split('_')[1])

                    c3 = classes[name3]

                    # if number > 99:
                    #     noises3 = old_noise.loc[old_noise["Filename"] == f"{name}_{number - 100}"]
                    # else:
                    #     noises3 = new_noise.loc[new_noise["Filename"] == f"{name}_{number}"]
                    # noises3 = noises3[noise].to_numpy().flatten()
                    # if len(noises3) == 0:
                    #     noises3 = np.zeros((5,))

                    s123 = np.load(folder + files[k]).astype('float32') + s12
                    
                    np.save(outPath + f'{count}.npy', s123)
                    df.loc[count] = [count] + [1 if c1 == i or c2 == i or c3 == i else 0 for i in range(len(header) - 1 - len(noise))] #+ list(np.logical_or(np.logical_or(noises1,noises2),noises3))
                    count += 1

                if count > 100000:
                    print("oh no")

    df.to_csv(labelPath, index=False)
