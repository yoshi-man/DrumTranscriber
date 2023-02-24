# Drum Transcriber - Transcribe Drum Audio Clips

This package helps users transcribes drum audio hits into 6 classes - Hihat, Crash, Kick Drum, Snare, Ride, and Toms.

![demo]('https://github.com/yoshi-man/DrumTranscriber/blob/main/assets/demo.gif?raw=true')



## Dependencies

Run the following to install the python dependencies:

```
pip install librosa tensorflow numpy pandas scikit-learn streamlit streamlit-player
```

## Usage

### Basic Usage
```Python
from DrumTranscriber import DrumTranscriber
import librosa

samples, sr = librosa.load(PATH/TO/AUDIO/CLIP)

transcriber = DrumTranscriber()

# pandas dataframe containing probabilities of classes
predictions = transcriber.predict(samples, sr)
```

### For Streamlit

cd to the parent directory and run the following command:
```
streamlit run frontend.py
```
A localhost website will appear with the demo app.


## Getting Started

1. Clone/Zip the directory
2. Redownload the model .h5 file from `/model/drum_transcriber.h5`

**Note**: There is an issue with Github zipping the .h5 model file. To properly get the model to work, I suggest downloading the model file directly to replace the model from the clone/zipped folder.
