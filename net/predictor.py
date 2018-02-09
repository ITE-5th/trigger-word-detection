import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from pydub import AudioSegment
from pydub.playback import play

from dataset.generator import graph_spectrogram
from net.network import Network


def detect_triggerword(filename, model):
    plt.subplot(2, 1, 1)

    x = graph_spectrogram(filename).astype(np.float32)
    # the spectogram outputs (freqs, Tx) and we want (Tx, freqs) to input into the model
    # x = x.swapaxes(0, 1)
    x = np.expand_dims(x, axis=0)
    predictions = model.predict(x).cpu().data.numpy()

    plt.subplot(2, 1, 2)
    plt.plot(predictions[0, :, 0])
    plt.ylabel('probability')
    plt.show()
    return predictions

def chime_on_activate(filename, predictions, threshold):
    chime_file = "../audio/chime.wav"
    audio_clip = AudioSegment.from_wav(filename)
    chime = AudioSegment.from_wav(chime_file)
    Ty = predictions.shape[1]
    # Step 1: Initialize the number of consecutive output steps to 0
    consecutive_timesteps = 0
    # Step 2: Loop over the output steps in the y
    for i in range(Ty):
        # Step 3: Increment consecutive output steps
        consecutive_timesteps += 1
        # Step 4: If prediction is higher than the threshold and more than 75 consecutive output steps have passed
        if predictions[0, i, 0] > threshold and consecutive_timesteps > 75:
            # Step 5: Superpose audio and background using pydub
            audio_clip = audio_clip.overlay(chime, position=((i / Ty) * audio_clip.duration_seconds) * 1000)
            # Step 6: Reset consecutive output steps to 0
            consecutive_timesteps = 0

    audio_clip.export("../outputs/chime_output.wav", format='wav')

def main():
    net = Network().cuda()
    print(net.eval())
    net.load_state_dict(torch.load(os.path.join('../models/', 'net.pkl')))

    filename = '../examples/2.wav'
    prediction = detect_triggerword(filename, net)
    chime_on_activate(filename, prediction, 0.5)
    # playsound("chime_output.wav")
    song = AudioSegment.from_wav("../outputs/chime_output.wav")
    play(song)


if __name__ == "__main__":
    main()
