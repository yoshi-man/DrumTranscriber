import librosa
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from utils.config import SETTINGS


def fix_audio_length(samples: np.array, sr: int, length: int) -> np.array:
    """
    :param samples (np.array): samples array of the audio
    :param sr (int): sample rate used for the samples
    :param length (int): target length in seconds
    :return samples (np.array): samples with padding or trimmings performed
    """
    desired_length = int(sr*length)
    if len(samples) > desired_length:
        # trim from both ends, symmetrically
        trim_amount = (len(samples) - desired_length)//2
        return samples[trim_amount:len(samples)-trim_amount]

    else:
        # add silence from both ends
        add_amount = (desired_length - len(samples))//2
        return np.concatenate((np.zeros(add_amount), samples, np.zeros(add_amount)))


def get_onset_frames(samples: np.array, sr: int = 44100) -> list:
    """
    :param samples (np.array): samples array of the audio
    :param sr (int): sample rate used for the samples
    :return onset_frames (list): list of frames where onsets have been detected in the format of [(s, e), ...] where s and e are the start and end frames, respectively
    """
    onset_backtracks = librosa.onset.onset_detect(y=samples, sr=sr,
                                                  units='samples', backtrack=True)

    # this to include the last frame end as onset_detect backtracking goes to the previous min point
    onset_backtracks = np.append(onset_backtracks, min(
        onset_backtracks[-1]+sr, len(samples)))

    onset_frames = list(zip(onset_backtracks[:-1], onset_backtracks[1:]))
    return onset_frames


def get_onset_samples(samples: np.array, sr: int = 44100, onset_frames: list = None) -> list:
    """
    :param samples (np.array): samples array of the audio
    :param sr (int): sample rate used for the samples
    :para onset_frames (list): if provided, will use precomputed onset_frames (optional)
    :return on_set_samples (list): list of np.arrays containing samples for each onset
    """
    if onset_frames is None:
        onset_frames = get_onset_frames(samples, sr)

    onset_samples = [fix_audio_length(samples[s:e], sr, 1)
                     for s, e in onset_frames]

    return onset_samples


def get_onset_times(samples: np.array, sr: int = 44100) -> np.array:
    """
    :param samples (np.array): samples array of the audio
    :param sr (int): sample rate used for the samples
    :return onset_times (np.array): np.array containing onset times in seconds along the samples
    """
    onset_times = librosa.onset.onset_detect(y=samples, sr=sr,
                                             units='time', backtrack=False)

    return onset_times


def get_mel_spectrogram(samples: np.array, sr: int = 44100, target_shape=SETTINGS['TARGET_SHAPE']) -> np.array:
    """
    :param samples (np.array): samples array of the audio
    :param sr (int): sample rate used for the samples
    :return mel_spectrogram (np.array): np.array containing melspectrogram features in decibels
    """
    hop_length = len(samples)//target_shape[0]

    mel_features = librosa.feature.melspectrogram(
        y=samples, sr=sr, hop_length=hop_length, n_mels=target_shape[0])

    mel_features = mel_features[:, :target_shape[1]]

    mel_in_db = librosa.power_to_db(mel_features, ref=np.max)
    scaler = MinMaxScaler(feature_range=(0, 1))

    return scaler.fit_transform(mel_in_db)
