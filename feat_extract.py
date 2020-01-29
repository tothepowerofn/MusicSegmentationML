import librosa
import librosa.display
import numpy as np
from numpy import genfromtxt
import os
import math

#Extracts MFCC features into a numpy array according to the parameters given.
def extractMFCCForWav(wavPath, S=None, n_mfcc=20, dct_type=2, norm='ortho', lifter=0, **kwargs):
    audio_path = wavPath
    sound, sampleRate = librosa.load(audio_path)
    mfccs = librosa.feature.mfcc(sound, sampleRate, S, n_mfcc, dct_type, norm, lifter, **kwargs)
    return mfccs.T

#Saves MFCC features to a .csv file according to the parameters given.
def saveMFCCForWav(wavPath, outpath, S=None, n_mfcc=20, dct_type=2, norm='ortho', lifter=0, **kwargs):
    mfccs = extractMFCCForWav(wavPath, S, n_mfcc, dct_type, norm, lifter, **kwargs)
    np.savetxt(outpath, mfccs, delimiter=",")

#Saves MFCC features for a folder of WAV files to .csv files.
def saveMFCCForWavs(wavFolderPath, outFolderPath):
    if not os.path.exists(outFolderPath):
        os.makedirs(outFolderPath)
    for filename in os.listdir(wavFolderPath):
        if filename.endswith(".wav"):
            print(filename)
            print(outFolderPath + filename + "-mfcc.csv")
            saveMFCCForWav(os.path.join(wavFolderPath, filename),outFolderPath + "/" + os.path.splitext(filename)[0] + "-mfcc.csv")

#Returns the sample corresponding to the desired time for the specified samplerate
def timestampToSampleNumber(time, sampleRate):
    return time*sampleRate

def timestampToHop(time, sampleRate, hopLength):
    return int(math.ceil(timestampToSampleNumber(time, sampleRate)/hopLength))

def generateTrainingDataForAudio(wavPath, annotationPath, featureOutputPath, S=None, n_mfcc=20, dct_type=2, norm='ortho', lifter=0, **kwargs):
    hopLength = 512 #TODO: WARNING! Change this to be based on kwargs
    sampleRate = 22050
    mfccs = extractMFCCForWav(wavPath, S, n_mfcc, dct_type, norm, lifter, **kwargs)
    annotationData = genfromtxt(annotationPath, delimiter=',')
    numWavSamples = mfccs.shape[0]
    print(numWavSamples)
    classifications = np.zeros(shape=(numWavSamples, 1))
    lastHop = 0
    for segment in annotationData:
        currentHop = timestampToHop(segment[0], sampleRate, hopLength)
        print(currentHop)
        classifications[lastHop:currentHop, 0] = segment[1]
        lastHop = currentHop
    classifications[lastHop:numWavSamples, 0] = classifications[lastHop-1, 0]
    feats = np.hstack((mfccs, classifications))
    np.savetxt(featureOutputPath, feats, delimiter=",")

def getFeatsAndClassificationFromFile(filepath):
    feats = genfromtxt(filepath, delimiter=',')
    return (feats[:,0:feats.shape[1]-1], feats[:,feats.shape[1]-1:feats.shape[1]])