import os

import matplotlib.pyplot as plt

from dataset.generator import Generator


class DatasetTester(object):
    @staticmethod
    def test(path='./waves/'):
        k = os.listdir(path)
        for i, filename in enumerate(k):
            if filename.endswith("wav"):
                plt.subplot(len(k), 1, 1 + i)
                Generator.graph_spectrogram(path + filename)
                plt.title(filename)
                # wave = AudioSegment.from_wav(path + filename)
                # waves.append(wave)
        plt.show()


if __name__ == "__main__":
    DatasetTester.test()