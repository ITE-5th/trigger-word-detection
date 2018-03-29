import os

import matplotlib.pyplot as plt
from pydub import AudioSegment
from scipy.io import wavfile


class Sound(object):
    @staticmethod
    def load_audio_from_dir(dir: str):
        data = []
        for filename in os.listdir(dir):
            if filename.endswith("wav"):
                datum = AudioSegment.from_wav(dir + filename)
                data.append(datum)

        return data

    # Load a wav file
    @staticmethod
    def get_wav_info(wav_file):
        rate, data = wavfile.read(wav_file)
        return rate, data

    # Calculate and plot spectrogram for a wav audio file
    @staticmethod
    def graph_spectrogram(wav_file):
        rate, data = Sound.get_wav_info(wav_file)
        # data = Generator.replaceZeroes(data)
        nfft = 200  # Length of each window segment
        fs = 8000  # Sampling frequencies
        noverlap = 120  # Overlap between windows
        nchannels = data.ndim
        if nchannels == 1:
            # freqs, _, pxx = signal.spectrogram(data, fs=fs,nperseg=nfft, nfft=nfft, noverlap=noverlap)
            pxx, freqs, bins, im = plt.specgram(data, nfft, fs, noverlap=noverlap)
        elif nchannels == 2:
            # freqs, _, pxx = signal.spectrogram(data[:, 0], fs=fs, nperseg=nfft, nfft=nfft, noverlap=noverlap)
            pxx, freqs, bins, im = plt.specgram(data[:, 0], nfft, fs, noverlap=noverlap)
        return pxx

    # Used to standardize volume of audio clip
    @staticmethod
    def match_target_amplitude(sound, target_dBFS):
        change_in_dBFS = target_dBFS - sound.dBFS
        return sound.apply_gain(change_in_dBFS)
