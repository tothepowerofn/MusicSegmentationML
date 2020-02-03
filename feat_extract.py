import librosa
import librosa.display
import numpy as np
from numpy import genfromtxt
from numpy import zeros, newaxis
from keras.utils import to_categorical
import os
import math


#Extracts MFCC features into a numpy array according to the parameters given.
def extractMFCCForWav(wavPath, S=None, n_mfcc=20, dct_type=2, norm='ortho', lifter=0, normalize = True, **kwargs):
    audio_path = wavPath
    sound, sampleRate = librosa.load(audio_path)
    mfccs = librosa.feature.mfcc(sound, sampleRate, S, n_mfcc, dct_type, norm, lifter, **kwargs)
    retVal = mfccs.T
    if normalize:
        return retVal / retVal.max(axis = 0)
    else:
        return retVal

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
    hopLength = 512  # TODO: WARNING! Change this to be based on kwargs
    sampleRate = 22050  # TODO: WARNING! Change this to be based on kwargs
    for key, value in kwargs.items():
        if key is "hop_length": #this is also usually 512
            print("setting hop length")
            hopLength = value
        elif key is "n_fft": #this is usually 2048
            print("setting window size")
    mfccs = extractMFCCForWav(wavPath, S, n_mfcc, dct_type, norm, lifter, **kwargs)
    annotationData = genfromtxt(annotationPath, delimiter=',')
    numWavSamples = mfccs.shape[0]
    classifications = np.zeros(shape=(numWavSamples, 1))
    lastHop = 0
    for segment in annotationData:
        currentHop = timestampToHop(segment[0], sampleRate, hopLength)
        classifications[lastHop:currentHop, 0] = segment[1]
        lastHop = currentHop
    classifications[lastHop:numWavSamples, 0] = classifications[lastHop-1, 0]
    feats = np.hstack((mfccs, classifications))
    np.savetxt(featureOutputPath, feats, delimiter=",")
def generateTrainingDataForAudios(wavFolderPath, annotationFolderPath, featureOutputFolderPath, S=None, n_mfcc=20, dct_type=2, norm='ortho', lifter=0, **kwargs):
    if not os.path.exists(featureOutputFolderPath):
        os.makedirs(featureOutputFolderPath)
    for filename in os.listdir(wavFolderPath):
        constructedAnnotationPath = annotationFolderPath + "/" + os.path.splitext(filename)[0] + "-annotation.csv"
        constructedFeatureOutputPath = featureOutputFolderPath + "/" + os.path.splitext(filename)[0] + "-feats.csv"
        if filename.endswith(".wav") and os.path.exists(constructedAnnotationPath):
            generateTrainingDataForAudio(wavFolderPath + "/" + filename, constructedAnnotationPath, constructedFeatureOutputPath, S, n_mfcc, dct_type, norm, lifter, **kwargs)


def getFeatsAndClassificationsFromFile(filepath):
    feats = genfromtxt(filepath, delimiter=',')
    features = feats[:,0:feats.shape[1]-1]
    classifications = feats[:,feats.shape[1]-1:feats.shape[1]]
    #np.savetxt(filepath + "-class.csv", to_categorical(classifications), delimiter=",")
    return (features[newaxis, :, :], to_categorical(classifications)[newaxis, :, :])

def trainingGeneratorFromFolder(folderpath):
    # stackedFeatures = np.stack((getFeatsAndClassificationsFromFile(folderpath + "/" + filename)[0] if filename.endswith("-feats.csv") else None for filename in os.listdir(folderpath)),axis=0)
    # print(stackedFeatures.shape)
    while True:
        for filename in os.listdir(folderpath):
            if filename.endswith("-feats.csv"):
                yield getFeatsAndClassificationsFromFile(folderpath + "/" + filename)

def getNumberOfFeatFiles(folderpath):
    # https://stackoverflow.com/questions/1320731/count-number-of-files-with-certain-extension-in-python
    return len([f for f in os.listdir(folderpath) if f.endswith('-feats.csv') and os.path.isfile(os.path.join(folderpath, f))])
