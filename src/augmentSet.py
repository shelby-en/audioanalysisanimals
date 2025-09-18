import librosa
import os
import numpy as np

import soundfile as sf

classes = {"Bat": 0, "Cockatoo": 1, "Crocodile": 2, "Dingo": 3, "Duck": 4, "Frog": 5, "FrogmouthTawny": 6, "Koala": 7, "Kookaburra": 8, "Magpie": 9, "Platypus": 10, "Possum": 11, "Snake": 12, "Wombat": 13}

def resize_audio(audio, time=10):
    numSamples = 48000*time
    audio = audio[:numSamples]
    zero_pad = np.zeros((numSamples - len(audio)))
    audio = np.concatenate((zero_pad, audio))
    return audio

def makeAudios(audio, name, time=5, sr=48000, overlap = 5):
    audios = []
    # print(len(audio))
    if len(audio) <= time*sr:
        # sf.write(name + f'_{1}_ol.wav', resize_audio(audio, 5), sr)
        sf.write(name + f'.wav', resize_audio(audio, 5), sr)
    else:
        i = 0
        count = 1
        while i+time*sr < len(audio):
            audios.append(audio[i:i+time*sr])
            # sf.write(name + f'_{count}_ol.wav', audio[i:i+time*sr], sr)
            sf.write(name + f'.wav', audio[i:i+time*sr], sr)
            i += overlap*sr
            count += 1

        print(len(audios))


        

if __name__ == "__main__":
    folder = "./data/augmented_mp3s/"
    outPath = "./data/overlapping/"

    files = []
    for file in os.listdir(folder):
        if file.endswith(".wav"):
            files.append(file)
    
    for i in range(len(files)):
        y1, sr = librosa.load(folder + '/' + files[i], sr=48000)
        for j in range(i+1, len(files)):
            y2, sr = librosa.load(folder + '/' + files[j], sr=48000)
            c1 = files[i].split('_')[0][:-1]
            c2 = files[j].split('_')[0][:-1]
            if c1 == c2:
                continue
            # print(c1, c2)
            makeAudios(y1 + y2, outPath + f'{i}_{j}_{classes[c1]}_{classes[c2]}')