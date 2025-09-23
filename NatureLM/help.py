import av
import numpy as np
import scipy.io.wavfile as wavf
import time
import threading
import datetime

from new_predict import Predictor


# TODO: update model path and class file path here
model = Predictor(f'../data/model/version2_1/version2_1/', '../data/classes.csv')

# TODO: fill in appropriate port
port = 4446

sr = 48000
t = 5
n = sr*t
interval = 2

globalBuffer = np.array([0]*n)

def predict_sample(sample):
    global timer
    timer = threading.Timer(interval, predict_sample, args=(globalBuffer,))
    timer.start()
    # predicts = model.predict(sample, sr)
    time.sleep(5)
    print(f"time: {datetime.datetime.now().strftime('%H:%M:%S')},\n animals: {sample[0]}")

timer = threading.Timer(interval, predict_sample, args=(globalBuffer,))

class FakeIter:
    def __init__(self, frameSize):
            self.frameSize = frameSize
            self.curr = -1*self.frameSize

    def __iter__(self):
        return self
    
    def __next__(self):
        self.curr += self.frameSize
        return np.array([i for i in range(self.curr, self.curr + self.frameSize)])

def listen(port):
    container = av.open(f"rtp://0.0.0.0:{port}", options={
        "probesize": "50000000",
        "analyzeduration": "10000000",
        "protocol_whitelist": "file,udp,rtp"
    })

    audio_stream = next(s for s in container.streams if s.type == 'audio')

    return container, audio_stream


def receive_audio_samples():
    container = av.open("rtp://0.0.0.0:4445", options={
        "probesize": "50000000",
        "analyzeduration": "10000000",
        "protocol_whitelist": "file,udp,rtp"
    })

    audio_stream = next(s for s in container.streams if s.type == 'audio')
    print(audio_stream)
    print(f"Audio stream opened: {audio_stream.rate} Hz, {audio_stream.channels} channels")

    count = 0
    frames = []

    try:
        for packet in container.demux(audio_stream):
            for frame in packet.decode():
                samples = frame.to_ndarray()

                if frame.format.name.endswith('planar'):
                    # Use only the first channel (mono) for now
                    raw_samples = samples[0]
                else:
                    raw_samples = samples

                # Normalize to float32 [-1.0, 1.0] if needed
                if frame.format.name.startswith('s16'):
                    float_samples = 100*raw_samples.astype(np.float32) / 32768.0
                else:
                    float_samples = raw_samples.astype(np.float32)

                frames.append(frame.to_ndarray()[0])

    except KeyboardInterrupt:
        wavf.write("out.wav", 48000, np.concatenate(frames))

def sliding_window(buffer, data, startIdx):
    idx = startIdx
    for i in range(len(data)):
        buffer[idx] = data[i]
        idx = (idx + 1) % len(buffer)

    return buffer, idx

def main():
    global globalBuffer, n, t, interval, timer, port
    model = Predictor(f'../data/model/version2_1/version2_1/', '../data/classes.csv')
    container, stream = listen(port)
    print("streaming")

    # make moving window stream hop = 1, len = 5
    buffer = np.array([0]*n)

    thing = FakeIter(3)

    idx = 0
    start = time.time()
    lastTime = start
    timerBool = False
    try:
        while True:
            if time.time() - start > t and not timerBool:
                timer.start()
                timerBool = True
            if time.time() - lastTime > interval:
                globalBuffer = buffer
                lastTime = time.time()

            ch1 = next(container.demux(stream)).decode()[0].to_ndarray()[0]
            # ch1 = next(thing)

            buffer, idx = sliding_window(buffer, ch1, idx)
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("finished gathersing samples")
        timer.cancel()

if __name__ == "__main__":
    main()