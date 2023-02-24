# Drum Transcriber - Transcribe Drum Audio Clips

This package helps users transcribes drum audio hits into 6 classes - Hihat, Crash, Kick Drum, Snare, Ride, and Toms.


## Installation

Run the following to install the python dependencies:

```
pip install librosa tensorflow numpy pandas scikit-learn
```

To install this package via pip, run:
```
pip install drum-transcriber
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

## Demo
