import os

import joblib
import matplotlib.pyplot as plt

from util.sound import Sound


class DatasetTester(object):
    @staticmethod
    def test_spectrogram(path='./waves/'):
        k = os.listdir(path)
        for i, filename in enumerate(k):
            if filename.endswith("wav"):
                plt.subplot(len(k), 1, 1 + i)
                Sound.graph_spectrogram(path + filename)
                plt.title(filename)
                # wave = AudioSegment.from_wav(path + filename)
                # waves.append(wave)
        plt.show()

    @staticmethod
    def test_pkl(path):
        with open(path, 'rb') as f:
            data = joblib.load(f)

        plt.plot(data[0])




if __name__ == "__main__":
    DatasetTester.test_pkl(path="../dataset/partitions/partition-0.pkl")
