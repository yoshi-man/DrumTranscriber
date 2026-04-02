import time
from pytube import YouTube
from streamlit_player import st_player

from DrumTranscriber import DrumTranscriber
from utils.config import SETTINGS

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt

import librosa
import librosa.display


@st.experimental_memo
def get_predictions_from_youtube(input, start_from=0):
    yt = YouTube(input)

    video = yt.streams.filter(only_audio=True).first()

    out_file = video.download(output_path=".")

    base, ext = os.path.splitext(out_file)
    new_file = base + '.wav'
    os.rename(out_file, new_file)

    samples, sr = librosa.load(
        new_file, sr=44100, duration=30, offset=start_from)
    st.audio(samples, sample_rate=sr)

    os.remove(new_file)

    preds = transcriber.predict(samples, sr)

    return preds, samples, sr


def get_predictions_from_file(uploaded_file, start_from=0):
    # Save uploaded file temporarily
    temp_path = f"temp_{uploaded_file.name}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    samples, sr = librosa.load(
        temp_path, sr=44100, duration=30, offset=start_from)
    st.audio(samples, sample_rate=sr)

    os.remove(temp_path)

    preds = transcriber.predict(samples, sr)

    return preds, samples, sr


@st.experimental_singleton
def initialise_transcriber():
    transcriber = DrumTranscriber()

    return transcriber


def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')


transcriber = initialise_transcriber()

st.title('Drum Transcriber Demo')

tab1, tab2 = st.tabs(["YouTube URL", "Upload WAV File"])

preds = None
samples = None
sr = None
start_from = 0

with tab1:
    input = st.text_input('Input youtube link here',
                          value='https://www.youtube.com/watch?v=4SDBJp_B5qQ')
    start_from = st.number_input(label='Start from (in seconds)', min_value=0)

    if input:
        st_player(input)

    if input and st.button('Transcribe', key='transcribe_yt'):
        st.title('Predictions')
        preds, samples, sr = get_predictions_from_youtube(input, start_from)

with tab2:
    uploaded_file = st.file_uploader("Choose a WAV file", type="wav")
    start_from_file = st.number_input(label='Start from (in seconds)', min_value=0, key='start_from_file')

    if uploaded_file is not None:
        st.audio(uploaded_file)

    if uploaded_file is not None and st.button('Transcribe', key='transcribe_file'):
        st.title('Predictions')
        preds, samples, sr = get_predictions_from_file(uploaded_file, start_from_file)

if preds is not None:
    labelled_preds = [SETTINGS['LABELS_INDEX'][i] for i in
                      np.argmax(
                          preds[SETTINGS['LABELS_INDEX'].values()].to_numpy(), axis=1)
                      ]

    preds['prediction'] = labelled_preds
    preds['confidence'] = preds.apply(
        lambda x: f"{x[x['prediction']]*100:.1f}%", axis=1)
    preds['time'] = preds['time'].round(2)

    st.write(preds[['time', 'prediction', 'confidence']].T)

    fig, ax = plt.subplots(sharex=True, nrows=7, figsize=(20, 20))

    librosa.display.waveshow(
        samples, sr=sr, offset=0, ax=ax[0])

    ax[0].set_yticklabels([])
    ax[0].set_xlabel(None)

    for i in range(1, 7):
        label_name = SETTINGS['LABELS_INDEX'][i-1]

        hit_times = np.array(
            preds[preds['prediction'] == label_name]['time'].to_list())

        ax[i].vlines(hit_times, -1, 1)
        ax[i].set_ylabel(label_name, rotation=0, fontsize=20)
        ax[i].set_yticklabels([])

        if i == 6:
            ax[i].set_xlabel('time')
            ax[i].set_xlabel('time')

    st.pyplot(fig)

    st.download_button(
        "Press to Download",
        convert_df(preds),
        "predictions.csv",
        "text/csv",
        key='download-csv'
    )

    st.write(preds, use_container_width=True)
