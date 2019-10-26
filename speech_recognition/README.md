# Speech Recognition

## General Notes on Speech Recognition

Goal of ASR is to take raw audio and return a predicted transcript of the audio.
This is done using three main tools - Feature Extraction, Acoustic Model, Decoder.

  * __Feature Extraction__ : Convert raw audio to commonly used feature representations

  * __Acoustic Model__ : Takes audio features and returns probability distrubtion over all possible transcriptions

  * __Decoder__ : Take probability distribution and output predicted transcription
