import librosa
import os
import numpy as np

import soundfile as sf

def resize_audio(audio, time=10):
    numSamples = 48000*time
    audio = audio[:numSamples]
    zero_pad = np.zeros((numSamples - len(audio)))
    audio = np.concatenate((zero_pad, audio))
    return audio

def makeAudios(audio, name, time=5, sr=48000, overlap = 5):
    audios = []
    if len(audio) < time*sr:
        sf.write(name + f'_{1}.wav', resize_audio(audio, 5), sr)
    else:
        i = 0
        count = 1
        while i+time*sr < len(audio):
            audios.append(audio[i:i+time*sr])
            sf.write(name + f'_{count}.wav', audio[i:i+time*sr], sr)
            i += overlap*sr
            count += 1

        print(len(audios))


        

if __name__ == "__main__":
    folder = "./data/original_mp3s/"
    outPath = "./data/augmented_mp3s/"

    files = []
    for file in os.listdir(folder):
        if file.endswith(".mp3"):
            files.append(file)
    
    for file in files:
        y, sr = librosa.load(folder + '/' + file, sr=48000)
        makeAudios(y, outPath + file[:-4])