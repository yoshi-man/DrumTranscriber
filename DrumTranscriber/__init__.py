"""
Created on Fri Feb 24 19:17:30 2023
@author: yoshi-man
"""


import tensorflow as tf
import numpy as np
import pandas as pd

from utils.config import SETTINGS
from utils.audio_utils import get_mel_spectrogram, get_onset_times, get_onset_samples


class DrumTranscriber:
    def __init__(self):
        self.model = tf.keras.models.load_model(SETTINGS["SAVED_MODEL_PATH"])

    def predict(self, samples: np.array, sr: int) -> pd.DataFrame:
        """
        :param file_path (str): Path to audio file to predict
        :param seconds (int): amount of seconds to predict
        :return predictions (np.array): Hits probability predicted by the model
        """
        # get onset
        onset_samples = get_onset_samples(samples, sr=sr)

        # convert to mel spectrogram
        mel_specs = np.array([get_mel_spectrogram(s, sr=sr)
                             for s in onset_samples])
        mel_specs = np.expand_dims(mel_specs, axis=-1).repeat(3, axis=-1)

        # onset times for each hit
        hit_times = get_onset_times(samples, sr)

        # get the predicted label
        predictions = self.model.predict(mel_specs)

        df = pd.DataFrame(predictions,
                          columns=list(SETTINGS['LABELS_INDEX'].values()))

        df['time'] = hit_times

        return df
