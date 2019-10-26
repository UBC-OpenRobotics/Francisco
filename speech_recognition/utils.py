import pyaudio
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from scipy.io import wavfile
from python_speech_features import mfcc
from IPython.display import Audio
from IPython.display import display

def recordWAV(filename,length):
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100
    CHUNK = 1024
    RECORD_SECONDS = length
    WAVE_OUTPUT_FILENAME = filename
    
    #Instatiate PyAudio
    audio = pyaudio.PyAudio()

    # Start Stream
    stream = audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)
    
    print("Recording")
    frames = []
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    
    print "Done Recording"
    stream.stop_stream()
    stream.close()
    audio.terminate()

    waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()
    print("Wrote {}".format(filename))

##This function takes the generated json files for the training and validation set and
#loads them into dictionaries.

#train_json_path = '/home/francisco/DNN-Speech-Recognizer/train_corpus.json' #defaults
#test_json_path = '/home/francisco/DNN-Speech-Recognizer/valid_corpus.json' #defaults

def load_json(train_json_path, test_json_path):
    audio_paths = []
    durations = []
    texts = []
    max_duration = 10.0

    #Load train_corpus.json
    with open(train_json_path) as json_line_file:
        for line_num, json_line in enumerate(json_line_file):
            try:
                spec = json.loads(json_line)
                if float(spec['duration']) > max_duration:
                    continue
                audio_paths.append(spec['key'])
                durations.append(float(spec['duration']))
                texts.append(spec['text'])
            except Exception as e:
                # Change to (KeyError, ValueError) or
                # (KeyError,json.decoder.JSONDecodeError), depending on
                # json module version
                print('Error reading line #{}: {}'
                            .format(line_num, json_line))
    train = {'durations':durations,
             'texts':texts,
             'paths':audio_paths}

    #Load validate_corpus.json
    audio_paths = []
    durations = []
    texts = []
    max_duration = 10.0
    with open(test_json_path) as json_line_file:
        for line_num, json_line in enumerate(json_line_file):
            try:
                spec = json.loads(json_line)
                if float(spec['duration']) > max_duration:
                    continue
                audio_paths.append(spec['key'])
                durations.append(float(spec['duration']))
                texts.append(spec['text'])
            except Exception as e:
                # Change to (KeyError, ValueError) or
                # (KeyError,json.decoder.JSONDecodeError), depending on
                # json module version
                print('Error reading line #{}: {}'
                            .format(line_num, json_line))
    test = {'durations':durations,
             'texts':texts,
             'paths':audio_paths}
    
    return train, test

#Explores an entry in the dataset. Returns duration, transcript and the raw signal processed
def probe_dataset(data, index, dataset_path):
    duration = data['durations'][index]
    path = data['paths'][index]
    text = data['texts'][index]
    
    audio_path=os.path.join(dataset_path, path)
    
    print('Index {} of the data has duration {}\n'.format(index, duration))
    print('Transcript: {}'.format(text))
    
    rate, signal = wavfile.read(audio_path)
    #Compute MFCC
    mfcc_feat = mfcc(signal, rate)
    
    #Plot
    fig,(ax1, ax2) = plt.subplots(2)
    fig.set_size_inches(12,8)
    fig.set_facecolor('w')
    ax1.plot(signal)
    ax2.imshow(mfcc_feat, aspect='auto')
    plt.show()
    
    Audio(audio_path)

#Input a dict with texts and paths, returns list of mfcc features
def compute_mfcc(data, dataset_path):
    mfcc_features = []
    for index in range(len(data['texts'])):
        path = data['paths'][index]
        signal_path = os.path.join(dataset_path, path)
        
        rate, signal = wavfile.read(signal_path)
        mfcc_feat = mfcc(signal, rate)
        mfcc_features.append(mfcc_feat)
    return mfcc_features