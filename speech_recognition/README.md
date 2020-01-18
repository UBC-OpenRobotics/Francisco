# Speech Recognition

## General Notes on Speech Recognition

Goal of ASR is to take raw audio and return a predicted transcript of the audio.
This is done using three main tools - Feature Extraction, Acoustic Model, Decoder.

  * __Feature Extraction__ : Convert raw audio to commonly used feature representations

  * __Acoustic Model__ : Takes audio features and returns probability distrubtion over all possible transcriptions

  * __Decoder__ : Take probability distribution and output predicted transcription

## Training Data

I'll be using a subset of [*LibriSpeech*](http://www.openslr.org/12/) to train the ASR model. Specifically, I'm using *dev-clean.tar.gz*. All the audio files are flac, so I've converted them to wav using ffmpeg.

Next, I generated a training and testing JSON which contained the duration of the .wav file, the text transcript and a pointer to the wav file itself.

## Feature Extraction

I'll be using the popular MFCC, which stands for *Mel Frequency Cepstral Coefficients*, and collectively, they make up a signal's Mel-Frequency Cepstral - which can be thought of as a short term power spectrum.

*python_speech_features* has an implementation of mfcc which I'll be using.


# Speech Recognition in ROS

For implementing speech recognition inside ROS, I will be using a pocketsphinx wrapper for ROS, allowing use of the CMUSphinx Speech recognizer.

I will be using this implementation - https://github.com/Pankaj-Baranwal/pocketsphinx

In addition to having the wrapper, we need a language model. I will be using the hub4wsj_sc_8k model found [*here*](https://www.google.com/url?q=https://sourceforge.net/projects/cmusphinx/files/Acoustic%2520and%2520Language%2520Models/Archive/US%2520English%2520HUB4WSJ%2520Acoustic%2520Model/&sa=D&ust=1579375130160000&usg=AFQjCNFBVru_4LZ-ORGAT5XunUzL3Ym1UA)

