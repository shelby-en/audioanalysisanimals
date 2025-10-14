#!/usr/bin/env python3
import soundfile as sf
import numpy as np
import time
import threading
import datetime
import av

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import String
from rclpy.executors import ExternalShutdownException

from predict_multi import Predictor

# ffmpeg -ar 48000 -codec:a pcm_s16le -f dshow -i audio="Microphone Array (Realtek(R) Audio)" -codec:a mp2 -f rtp rtp://127.0.0.1:4444

PORT = 4444
RTP_URL = f"rtp://0.0.0.0:{PORT}"
# RTP_URL = f"rtp://127.0.0.1:{PORT}"
DURATION_TIME = 5
MODEL_TIME = 1 # in seconds
CHANNELS = 2
SAMPLE_RATE = 48000
FORMAT = 's16'

version = 35
filt = True
model = Predictor(f'./data/version_{version}.ckpt', './data/classes.csv', filt=filt)

class AudioClassificationNode(Node):
    _predicts_publisher: rclpy.publisher.Publisher
    _threads: list
    _file_number: int

    def __init__(self):
        super().__init__('audio_model_node')

        self._threads = []
        self._file_number = 0

        self._predicts_publisher = self.create_publisher(
            String,
            "audio_model",
            qos_profile_sensor_data,
        )

        self.get_logger().info("Audio classification node started")

        self.use_pyav()

    def publish_predicts(self) -> None:
        string = String()
        string.data = ', '.join(self.predicts)
        self._predicts_publisher.publish(string)

    def write_wav_file(self, filename, audio_data, sample_rate):
        sf.write(filename, audio_data, sample_rate)

    def predict_sample(self, sample):
        # print(sample[:,0].shape)
        self.predicts = model.predict(sample[:,0])
        print(f"time: {datetime.datetime.now().strftime('%H:%M:%S')}\n, animals: {self.predicts}")
        self.publish_predicts()
        # self.write_wav_file(f"test{self._file_number}.wav", sample, SAMPLE_RATE)


    def use_pyav(self):
        container = av.open(RTP_URL)
        stream = container.streams.audio[0]
        stream.codec_context.sample_rate = SAMPLE_RATE
        stream.codec_context.format = FORMAT
        
        frames = []
        duration_time = time.time()
        model_time = time.time()
        timestamps = []
        i = 0
        # self._file_number = 0

        try:

            for packet in container.demux(stream):

                for frame in packet.decode():
                    frames.append(frame.to_ndarray())
                    timestamps.append(time.time())

                if (time.time() - duration_time) > DURATION_TIME:
                    if (time.time() - model_time) > (MODEL_TIME):
                        # print((len(frames), len(timestamps)))
                        # print(frames[0:10])
                        
                        model_time = time.time()
                        audio_data = np.concatenate(frames, axis=1).T

                        for i in range(len(timestamps)):
                            if (time.time() - timestamps[i]) <= DURATION_TIME:
                                break
                        timestamps = timestamps[i:]
                        frames = frames[i:]

                        # self.write_wav_file(f'recordings/file{self._file_number}.wav', audio_data, SAMPLE_RATE)
                        t = threading.Thread(target=self.predict_sample, args=(audio_data,))
                        
                        t.start()
                        self._threads.append(t)

                        self._file_number += 1

        except KeyboardInterrupt:
            for thread in self._threads:
                thread.join()
            print("Interrupted by user.")

if __name__ == "__main__":
    print("Starting audio reception...")

    rclpy.init()
    audio_classifier = AudioClassificationNode()
    try:
        rclpy.spin(audio_classifier)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    audio_classifier.destroy_node()

    print("Done.")
