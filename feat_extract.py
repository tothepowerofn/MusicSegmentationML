import librosa
import librosa.display
import numpy as np
import os

#Extracts MFCC features into a numpy array according to the parameters given.
def extractMFCCForWav(wavPath, S=None, n_mfcc=20, dct_type=2, norm='ortho', lifter=0, **kwargs):
    audio_path = wavPath
    sound, sampleRate = librosa.load(audio_path)
    mfccs = librosa.feature.mfcc(sound, sampleRate, S, n_mfcc, dct_type, norm, lifter, **kwargs)
    return mfccs

#Saves MFCC features to a .csv file according to the parameters given.
def saveMFCCForWav(wavPath, outpath, S=None, n_mfcc=20, dct_type=2, norm='ortho', lifter=0, **kwargs):
    mfccs = extractMFCCForWav(wavPath, S, n_mfcc, dct_type, norm, lifter, **kwargs)
    np.savetxt(outpath, mfccs.T, delimiter=",")

#Saves MFCC features for a folder of WAV files to .csv files.
def saveMFCCForWavs(wavFolderPath, outFolderPath):
    if not os.path.exists(outFolderPath):
        os.makedirs(outFolderPath)
    for filename in os.listdir(wavFolderPath):
        if filename.endswith(".wav"):
            print(filename)
            print(outFolderPath + filename + "-mfcc.csv")
            saveMFCCForWav(os.path.join(wavFolderPath, filename),outFolderPath + "/" + os.path.splitext(filename)[0] + "-mfcc.csv")

