import csv
import scipy.io.wavfile as wav

def getSlice(audio, start, end, sr):
    return audio[start*sr:end*sr]

def downSample(audio, srFrom, srTo):
    ratio = srFrom//srTo

    return audio[0::ratio]

def main():
    labelFile = "./data/help.csv"
    soundFile = "./data/all_animals.wav"
    outFolder = "./data/new_wavs_48000/"
    rate = 48000

    sr, data = wav.read(soundFile)
    data = data[:, 0] # get only left channel
    print(len(data))
    print(sr)

    data = downSample(data, sr, rate)
    print(len(data))

    with open(labelFile, 'r') as names:
        reader = csv.reader(names)
        for row in reader:
            newAudio = getSlice(data, int(row[1]), int(row[2]), rate)
            wav.write(outFolder + row[0] + ".wav", rate, newAudio)


if __name__ == "__main__":
    main()