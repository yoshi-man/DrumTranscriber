from utils.audio_utils import *
from utils.config import SETTINGS

import json

import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras import utils
from PIL import Image

from collections import Counter

import os


class Labels():
    def __init__(self, json_path):
        f = open(json_path, "r")
        annotations = json.load(f).get('annotations')

        self.annotations = [
            x for x in annotations if x[1] in SETTINGS['LABELS_INDEX'].values()]

    def get_audio_path(self):
        if self.annotations is None:
            return None

        audio_path = '/'.join(self.annotations[0][0].split("/")[:-1])
        return audio_path

    def generate_data(self):
        if self.annotations is None:
            return None

        audio_path = self.get_audio_path()

        samples, _ = librosa.load(audio_path, sr=44100)
        onset_samples = get_onset_samples(samples)

        labeled_samples = [
            onset_samples[int(i[0].split('/')[-1])] for i in self.annotations]

        labels = np.array([y[1] for y in self.annotations])

        return labeled_samples, labels


class Dataset():
    def __init__(self, folder_path):
        self.folder_path = folder_path

    def generate_data(self, verbose=False):
        X = None
        Y = None
        labels_jsons = [x for x in os.listdir(
            self.folder_path) if x.split('.')[-1] == 'json']

        for labels_json in labels_jsons:
            json_path = f"{self.folder_path}/{labels_json}"
            if verbose:
                print(f"Reading {json_path=}...")
            labels = Labels(json_path)
            x, y = labels.generate_data()

            if X is None:
                X = x.copy()

            else:
                X = np.concatenate([X, x], axis=0)

            if Y is None:
                Y = y.copy()
            else:
                Y = np.concatenate([Y, y], axis=0)

        return X, Y


class Preprocessor():
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def train_val_test_split(self):
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=SETTINGS['VAL_TEST_RATIO'], stratify=self.y)

        X_val, X_test, y_val, y_test = train_test_split(
            X_test, y_test, test_size=SETTINGS['TEST_RATIO'], stratify=y_test)

        return X_train, y_train, X_val, y_val, X_test, y_test

    def balance_dataset(self, X_train, y_train, N=None, verbose=False):
        label_counts = dict(Counter(y_train))

        X_train_sampled = None
        y_train_sampled = None

        # upsampling all other labels to max_count
        if N is None:
            N = max(label_counts.values())

        for label in SETTINGS['LABELS_INDEX'].values():
            if verbose:
                print(f"Sampling {label} from {label_counts[label]} to {N}...")

            label_indices = np.where(y_train == label)[0]

            selected_samples_indices = np.random.choice(
                label_indices, N, replace=True)

            selected_samples_X = X_train[selected_samples_indices, :]
            selected_samples_y = y_train[selected_samples_indices]

            if X_train_sampled is None:
                X_train_sampled = selected_samples_X
            else:
                X_train_sampled = np.concatenate(
                    [X_train_sampled, selected_samples_X])

            if y_train_sampled is None:
                y_train_sampled = selected_samples_y
            else:
                y_train_sampled = np.concatenate(
                    [y_train_sampled, selected_samples_y])

        return X_train_sampled, y_train_sampled

    def augment_train_data(self, X_train):
        return np.array([apply_augmentation(x) for x in X_train])

    def convert_to_mel_spectrograms(self, X_train, X_val, X_test):
        X_train = np.array([get_mel_spectrogram(x) for x in X_train])
        X_val = np.array([get_mel_spectrogram(x) for x in X_val])
        X_test = np.array([get_mel_spectrogram(x) for x in X_test])

        return X_train, X_val, X_test

    def convert_y_to_categorical(self, y_train, y_val, y_test):
        labels_dict_reverse = {v: k for k,
                               v in SETTINGS['LABELS_INDEX'].items()}
        num_classes = len(labels_dict_reverse.keys())

        y_train_int = [labels_dict_reverse[y] for y in y_train]
        y_val_int = [labels_dict_reverse[y] for y in y_val]
        y_test_int = [labels_dict_reverse[y] for y in y_test]

        y_train_int = utils.to_categorical(y_train_int, num_classes)
        y_val_int = utils.to_categorical(y_val_int, num_classes)
        y_test_int = utils.to_categorical(y_test_int, num_classes)

        return y_train_int, y_val_int, y_test_int

    def preprocess(self, balance_dataset=True, verbose=False):
        if verbose:
            print('Splitting data to train_test_split...')
        X_train, y_train, X_val, y_val, X_test, y_test = self.train_val_test_split()

        if balance_dataset:
            if verbose:
                print('Balancing train set...')
            X_train, y_train = self.balance_dataset(
                X_train, y_train, N=SETTINGS['TRAINING_SAMPLES_PER_LABEL'], verbose=verbose)

        if verbose:
            print('Augmenting train set...')
        X_train = self.augment_train_data(X_train)

        if verbose:
            print('Converting X to mel spectrograms...')
        X_train, X_val, X_test = self.convert_to_mel_spectrograms(
            X_train, X_val, X_test)

        if verbose:
            print('Converting y to categoricals...')
        y_train, y_val, y_test = self.convert_y_to_categorical(
            y_train, y_val, y_test)

        return X_train, y_train, X_val, y_val, X_test, y_test


if __name__ == '__main__':
    # create the directories
    parent_folder = os.listdir('./')
    if 'dataset' not in parent_folder:
        os.mkdir('./dataset')

    dataset_folder = os.listdir('./dataset')
    for dataset in ['train', 'val', 'test']:
        if dataset not in dataset_folder:
            os.mkdir(f"./dataset/{dataset}")
            for label in SETTINGS['LABELS_INDEX'].values():
                if label not in dataset_folder:
                    os.mkdir(f"./dataset/{dataset}/{label}")

    # preprocessing dataset
    dataset = Dataset('./labels')

    X, y = dataset.generate_data(verbose=True)

    preprocessor = Preprocessor(X, y)

    X_train, y_train, X_val, y_val, X_test, y_test = preprocessor.preprocess(
        balance_dataset=True, verbose=True)

    print(f"{X_train.shape=} || {y_train.shape=}")
    print(f"{X_val.shape=} || {y_val.shape=}")
    print(f"{X_test.shape=} || {y_test.shape=}")

    # add the images to the dataset in png format
    y_labels = [SETTINGS['LABELS_INDEX'][y]
                for y in np.argmax(y_train, axis=1)]

    i = 0
    for x, label in zip(X_train, y_labels):
        print(f"Generating train image #{i+1}...", end='\r')
        im = Image.fromarray(x*255.0).convert('RGB')
        im.save(f"./dataset/train/{label}/{i}.png")
        i += 1
    print(f"Generated All Train Images........")

    # add the images to the dataset in png format
    y_labels = [SETTINGS['LABELS_INDEX'][y]
                for y in np.argmax(y_val, axis=1)]

    i = 0
    for x, label in zip(X_val, y_labels):
        print(f"Generating val image #{i+1}...", end='\r')
        im = Image.fromarray(x*255.0).convert('RGB')
        im.save(f"./dataset/val/{label}/{i}.png")
        i += 1
    print(f"Generated All Val Images........")

    # add the images to the dataset in png format
    y_labels = [SETTINGS['LABELS_INDEX'][y]
                for y in np.argmax(y_test, axis=1)]
    i = 0
    for x, label in zip(X_test, y_labels):
        print(f"Generating test image #{i+1}...", end='\r')

        im = Image.fromarray(x*255.0).convert('RGB')
        im.save(f"./dataset/test/{label}/{i}.png")
        i += 1

    print(f"Generated All Test Images........")
