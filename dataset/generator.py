import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from pydub import AudioSegment
from scipy.io import wavfile


class Generator(object):
    Ty = 1375
    Tx = 5511
    n_freq = 101

    def __init__(self) -> None:
        super().__init__()
        self.activates, self.negatives, self.backgrounds = self.load_raw_audio()

    @staticmethod
    def replaceZeroes(data):
        min_nonzero = np.min(data[np.nonzero(data)])
        data[data == 0] = min_nonzero
        return data

    def create_dataset(self, count):
        dataset = []
        for i in range(count):
            background_index = np.random.randint(low=0, high=len(self.backgrounds))
            x, y = Generator.create_training_example(self.backgrounds[background_index], self.activates,
                                                     self.negatives)
            # Export new training example
            # filepath = "./waves/train" + str(i) + ".wav"
            filepath = 'train.wav'
            # x = Generator.replaceZeroes(x)
            x.export(filepath, format="wav")
            # a = x.get_array_of_samples().astype(np.float64)
            # Get and plot spectrogram of the new recording (background with superposition of positive and negatives)
            x = Generator.graph_spectrogram(filepath)

            dataset.append((x, y))

            # if i % 100 == 0:
            print(i)

        with open('dataset.pkl', 'wb') as f:
            pickle.dump(dataset, f)
        print("Dataset was saved in your directory!")

    @staticmethod
    def get_random_time_segment(segment_ms):
        """
        Gets a random time segment of duration segment_ms in a 10,000 ms audio clip.

        Arguments:
        segment_ms -- the duration of the audio clip in ms ("ms" stands for "milliseconds")

        Returns:
        segment_time -- a tuple of (segment_start, segment_end) in ms
        """

        segment_start = np.random.randint(low=0,
                                          high=10000 - segment_ms)  # Make sure segment doesn't run past the 10sec background
        segment_end = segment_start + segment_ms - 1

        return (segment_start, segment_end)

    @staticmethod
    def is_overlapping(segment_time, previous_segments):
        """
        Checks if the time of a segment overlaps with the times of existing segments.

        Arguments:
        segment_time -- a tuple of (segment_start, segment_end) for the new segment
        previous_segments -- a list of tuples of (segment_start, segment_end) for the existing segments

        Returns:
        True if the time segment overlaps with any of the existing segments, False otherwise
        """

        segment_start, segment_end = segment_time

        # Step 1: Initialize overlap as a "False" flag. (≈ 1 line)
        overlap = False

        # Step 2: loop over the previous_segments start and end times.
        # Compare start/end times and set the flag to True if there is an overlap (≈ 3 lines)
        for previous_start, previous_end in previous_segments:
            # if segment_start in range(previous_start, previous_end+1) or segment_end in range(previous_start, previous_end+1):
            if segment_end >= previous_start and segment_start <= previous_end:
                overlap = True
                break

        return overlap

    @staticmethod
    def insert_audio_clip(background, audio_clip, previous_segments):
        """
        Insert a new audio segment over the background noise at a random time step, ensuring that the
        audio segment does not overlap with existing segments.

        Arguments:
        background -- a 10 second background audio recording.
        audio_clip -- the audio clip to be inserted/overlaid.
        previous_segments -- times where audio segments have already been placed

        Returns:
        new_background -- the updated background audio
        """

        # Get the duration of the audio clip in ms
        segment_ms = len(audio_clip)

        # Step 1: Use one of the helper functions to pick a random time segment onto which to insert
        # the new audio clip. (≈ 1 line)
        segment_time = Generator.get_random_time_segment(segment_ms)

        # Step 2: Check if the new segment_time overlaps with one of the previous_segments. If so, keep
        # picking new segment_time at random until it doesn't overlap. (≈ 2 lines)
        while Generator.is_overlapping(segment_time, previous_segments):
            segment_time = Generator.get_random_time_segment(segment_ms)

        # Step 3: Add the new segment_time to the list of previous_segments (≈ 1 line)
        previous_segments.append(segment_time)

        # Step 4: Superpose audio segment and background
        new_background = background.overlay(audio_clip, position=segment_time[0])

        return new_background, segment_time

    @staticmethod
    def insert_ones(y, segment_end_ms):
        """
        Update the label vector y. The labels of the 50 output steps strictly after the end of the segment
        should be set to 1. By strictly we mean that the label of segment_end_y should be 0 while, the
        50 followinf labels should be ones.


        Arguments:
        y -- numpy array of shape (1, Ty), the labels of the training example
        segment_end_ms -- the end time of the segment in ms

        Returns:
        y -- updated labels
        """

        # duration of the background (in terms of spectrogram time-steps)
        segment_end_y = int(segment_end_ms * Generator.Ty / 10000.0)

        # Add 1 to the correct index in the background label (y)
        y[0, segment_end_y + 1:segment_end_y + 51] = 1

        return y

    @staticmethod
    def create_training_example(background, activates, negatives):
        """
        Creates a training example with a given background, activates, and negatives.

        Arguments:
        background -- a 10 second background audio recording
        activates -- a list of audio segments of the word "activate"
        negatives -- a list of audio segments of random words that are not "activate"

        Returns:
        x -- the spectrogram of the training example
        y -- the label at each time step of the spectrogram
        """

        # Make background quieter
        background = background - 20

        # Step 1: Initialize y (label vector) of zeros (≈ 1 line)
        y = np.zeros((1, Generator.Ty))

        # Step 2: Initialize segment times as empty list (≈ 1 line)
        previous_segments = []

        # Select 0-4 random "activate" audio clips from the entire list of "activates" recordings
        number_of_activates = np.random.randint(0, 5)
        random_indices = np.random.randint(len(activates), size=number_of_activates)
        random_activates = [activates[i] for i in random_indices]

        # Step 3: Loop over randomly selected "activate" clips and insert in background
        for random_activate in random_activates:
            # Insert the audio clip on the background
            background, segment_time = Generator.insert_audio_clip(background, random_activate, previous_segments)
            # Retrieve segment_start and segment_end from segment_time
            segment_start, segment_end = segment_time
            # Insert labels in "y"
            y = Generator.insert_ones(y, segment_end)

        # Select 0-2 random negatives audio recordings from the entire list of "negatives" recordings
        number_of_negatives = np.random.randint(0, 3)
        random_indices = np.random.randint(len(negatives), size=number_of_negatives)
        random_negatives = [negatives[i] for i in random_indices]

        # Step 4: Loop over randomly selected negative clips and insert in background
        for random_negative in random_negatives:
            # Insert the audio clip on the background
            background, _ = Generator.insert_audio_clip(background, random_negative, previous_segments)

        # Standardize the volume of the audio clip
        x = Generator.match_target_amplitude(background, -20.0)

        return x, y

    # Calculate and plot spectrogram for a wav audio file
    @staticmethod
    def graph_spectrogram(wav_file):
        rate, data = Generator.get_wav_info(wav_file)
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

    # Load a wav file
    @staticmethod
    def get_wav_info(wav_file):
        rate, data = wavfile.read(wav_file)
        # song = AudioSegment.from_wav(wav_file)
        # play(song)
        return rate, data

    # Used to standardize volume of audio clip
    @staticmethod
    def match_target_amplitude(sound, target_dBFS):
        change_in_dBFS = target_dBFS - sound.dBFS
        return sound.apply_gain(change_in_dBFS)

    # Load raw audio files for speech synthesis
    @staticmethod
    def load_raw_audio():
        activates = []
        backgrounds = []
        negatives = []
        for filename in os.listdir("./raw_data/activates"):
            if filename.endswith("wav"):
                activate = AudioSegment.from_wav("./raw_data/activates/" + filename)
                activates.append(activate)
        for filename in os.listdir("./raw_data/backgrounds"):
            if filename.endswith("wav"):
                background = AudioSegment.from_wav("./raw_data/backgrounds/" + filename)
                backgrounds.append(background)
        for filename in os.listdir("./raw_data/negatives"):
            if filename.endswith("wav"):
                negative = AudioSegment.from_wav("./raw_data/negatives/" + filename)
                negatives.append(negative)
        return activates, negatives, backgrounds


if __name__ == "__main__":
    generator = Generator()
    generator.create_dataset(5000)
