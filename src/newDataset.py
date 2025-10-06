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
    folder = "./data/more_spectrograms/"
    outPath = "./data/more_multi/"
    labelPath = "./data/more_labels.csv"

    files = []
    for file in os.listdir(folder):
        if file.endswith(".npy"):
            files.append(file)

    print(len(files))

    props = [1, 0.3, 0.001]

    header = ["fileName","Bat","Cockatoo","Crocodile","Dingo","Duck","Frog","FrogmouthTawny","Koala","Kookaburra","Magpie","Platypus","Possum","Snake","Wombat"]
    df = pd.DataFrame(columns = header)

    # df.loc[i] = [filename] + [1 if c1 == i or c2 == i else 0 for i in range(len(header) - 1)]

    count = 1
    for i in range(len(files)):
        name1 = files[i][:-4].split('_')[0][:-1]
        c1 = classes[name1]
        
        s1 = np.load(folder + files[i]).astype('float32')

        if np.random.uniform(0, 1) <= props[0]:
            np.save(outPath + f'{count}.npy', s1)
            df.loc[count] = [count] + [1 if c1 == i else 0 for i in range(len(header) - 1)]
            count += 1

        for j in range(i+1, len(files)):
            name2 = files[j][:-4].split('_')[0][:-1]
            c2 = classes[name2]

            s12 = np.load(folder + files[j]).astype('float32') + s1

            if np.random.uniform(0, 1) <= props[1]:
                np.save(outPath + f'{count}.npy', s12)
                df.loc[count] = [count] + [1 if c1 == i or c2 == i else 0 for i in range(len(header) - 1)]
                count += 1

            # print(count)

            for k in range(j+1, len(files)):
                if np.random.uniform(0, 1) <= props[2]:
                    name3 = files[k][:-4].split('_')[0][:-1]
                    c3 = classes[name3]

                    s123 = np.load(folder + files[k]).astype('float32') + s12
                    
                    np.save(outPath + f'{count}.npy', s123)
                    df.loc[count] = [count] + [1 if c1 == i or c2 == i or c3 == i else 0 for i in range(len(header) - 1)]
                    count += 1

                if count > 100000:
                    print("oh no")

    df.to_csv(labelPath, index=False)
