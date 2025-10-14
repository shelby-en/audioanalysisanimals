import noisereduce

from scipy.io import wavfile
import librosa
import noisereduce as nr

from pydub import AudioSegment, effects

import numpy as np

import os

props = [1, 1]

def filter(y, noisePath, outpath = "", rate = None, audio = False):
    if not audio:   
        rate, data = wavfile.read(y)
    else:
        data = y

    noiseFiles = []
    for file in os.listdir(noisePath):
        if file.endswith(".wav"):
            noiseFiles.append(file)

    for i, sound in enumerate(noiseFiles):
        # print(sound)
        
        _, noise = wavfile.read(noisePath + sound)
        
        # perform noise reduction
        reduced_noise = nr.reduce_noise(y=data, sr=rate, y_noise = noise, prop_decrease=props[i], time_constant_s=1)
        data = reduced_noise

    wavfile.write("temp.wav", rate, reduced_noise)
    segment = AudioSegment.from_wav("temp.wav")
    norm = effects.normalize(segment)
    if not audio:
        # wavfile.write(outpath, rate, reduced_noise)
        norm.export(outpath, format="wav", bitrate=rate)
    else:
        return norm.get_array_of_samples()

if __name__ == "__main__":
    rate, y1 = wavfile.read("./data/sample_5s/WombatB_4.wav")
    # y1, rate = librosa.load("./data/new_wavs_48000/Platypu/sA.wav")

    # wavfile.write("temp1.wav", rate1, data1)
    # wavfile.write("temp2.wav", rate1, data1)


    # print(data.dtype)
    # print(max(data), min(data))
    test = np.array(filter(y1, "./data/noise_files/", rate=rate, audio=True))
    # print(np.array(filter(data2, "./data/noise_files/", rate=rate2, audio=True)))
    wavfile.write("test.wav", rate, test)


    # # og_folder = "./data/original_mp3s/"
    # dataFolder = "./data/mixed_wavs/"
    # # for file in os.listdir(og_folder):
    # #     data, rate = librosa.load(og_folder + file)
    # #     wavfile.write(dataFolder + file[:-4] + ".wav", rate, data)
    
    # dataFiles = []
    # for file in os.listdir(dataFolder):
    #     if file.endswith(".wav"):
    #         dataFiles.append(file)

    

    # noiseFolder = "./data/noise_files/"
    # noiseFiles = []
    # for file in os.listdir(noiseFolder):
    #     if file.endswith(".wav"):
    #         noiseFiles.append(file)

    # outPath = "./data/filtered_wavs/"

    # for sound in dataFiles:
    #     filter(dataFolder + sound, noiseFolder, outPath + sound)