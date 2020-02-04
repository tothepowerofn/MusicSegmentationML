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

class MFCCFeature:
    def __init__(self, dataPath, S=None, n_mfcc=20, dct_type=2, norm='ortho', lifter=0, normalize = True, **kwargs):
        self.S = S
        self.n_mfcc = n_mfcc
        self.dct_type = dct_type
        self.norm = norm
        self.lifter = lifter
        self.normalize = True
        self.kwargs = kwargs
        self.dataPath = dataPath
        self.songs = {}

    def extractMFCCForWav(self,wavPath, S=None, n_mfcc=20, dct_type=2, norm='ortho', lifter=0, normalize=True, **kwargs):
        audio_path = wavPath
        sound, sampleRate = librosa.load(audio_path)
        mfccs = librosa.feature.mfcc(sound, sampleRate, S, n_mfcc, dct_type, norm, lifter, **kwargs)
        retVal = mfccs.T
        print(("!", retVal.shape))
        if normalize:
            return retVal / retVal.max(axis=0)
        else:
            return retVal

    def extractSingle(self, wavPath):
        hopLength = 512  # TODO: WARNING! Change this to be based on kwargs
        sampleRate = 22050  # TODO: WARNING! Change this to be based on kwargs
        n_fft = 2048
        for key, value in self.kwargs.items():
            if key is "hop_length":  # this is also usually 512
                hopLength = value
            elif key is "n_fft":  # this is usually 2048
                n_fft = value
        mfccs = self.extractMFCCForWav(wavPath, self.S, self.n_mfcc, self.dct_type, self.norm, self.lifter, hop_length=hopLength, n_fft=n_fft)
        return mfccs

    def extract(self):
        if(self.songs):
            return self.songs
        else:
            for filename in os.listdir(self.dataPath):
                if filename.endswith(".wav"):
                    self.songs[os.path.splitext(filename)[0]] = self.extractSingle(self.dataPath + "/" + filename)
            return self.songs

class Pooling1DFeature: #this is NOT relate to max-pooling.
    def __init__(self, featureToPool, numSamples):
        self.featureToPool = featureToPool
        self.numSamples = numSamples
        if numSamples < 2:
            raise Exception("You need to pool more than 1 sample! It makes no sense to pool 1 sample, just use the feature itself in that case!")
        self.extractedFeatures = {}
    def extractSingle(self, name):
        featureDict = self.featureToPool.extract()
        features = featureDict[name]
        beginningPoolList = []
        #pool beginning with 0 pad
        for i in range(1, self.numSamples+1):
            beginning = np.zeros((1, (self.numSamples - i)*features.shape[1]))
            beginning = np.hstack((beginning, features[0:i,:].flatten()[None, :]))
            beginningPoolList.append(beginning)
            # print(beginning.shape)
        beginningPool = np.stack(beginningPoolList, axis=1)
        # for i in range(self.numSamples, features.shape[0]):
        #     print(features[i-self.numSamples:i,:].flatten()[None, :].shape)

        regularPool = np.stack((features[i-self.numSamples:i,:].flatten()[None, :] for i in range(self.numSamples, features.shape[0])), axis=1)
        # print(beginningPool.shape)
        # print(regularPool.shape)
        pool = np.vstack([beginningPool[0], regularPool[0]])
        # print(pool.shape)
        return pool
    def extract(self):
        if self.extractedFeatures:
            return self.extractedFeatures
        else:
            for name in self.featureToPool.extract().keys():
                self.extractedFeatures[name] = self.extractSingle(name)
            return self.extractedFeatures





class AnnotatedSongLabeler:
    def __init__(self, dataPath, sample_rate, hop_length):
        self.sampleRate = sample_rate
        self.hopLength = hop_length
        self.dataPath = dataPath

    def getClassifiableDataNames(self):
        songList = []
        for filename in os.listdir(self.dataPath):
            if filename.endswith("-annotation.csv"):
                songList.append(os.path.basename(filename).split("-annotation.csv")[0])
        return songList

    def labelByName(self, songName, features):
        annotationPath = self.dataPath + "/" + songName + "-annotation.csv"
        annotationData = genfromtxt(annotationPath, delimiter=',')
        numWavSamples = features.shape[0]
        classifications = np.zeros(shape=(numWavSamples, 1))
        lastHop = 0
        for segment in annotationData:
            currentHop = timestampToHop(segment[0], self.sampleRate, self.hopLength)
            classifications[lastHop:currentHop, 0] = segment[1]
            lastHop = currentHop
        classifications[lastHop:numWavSamples, 0] = classifications[lastHop - 1, 0]
        return classifications



def generateLabeledFeatures(labeler, *args):
    dictList = []
    generatedLabeledFeaturesDict = {}
    for feature in args:
        dictList.append(feature.extract())
    for name in labeler.getClassifiableDataNames():
        combinedFeatures = np.hstack((dict[name] for dict in dictList))
        combinedFeaturesAndClassifications = np.hstack((combinedFeatures, labeler.labelByName(name, combinedFeatures)))
        generatedLabeledFeaturesDict[name] = combinedFeaturesAndClassifications
    return generatedLabeledFeaturesDict

def saveFeatures(generatedFeaturesDict, featureOutputFolderPath):
    for name, features in generatedFeaturesDict.items():
        np.savetxt(featureOutputFolderPath + "/" + name + "-feats.csv", features, delimiter=",")



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
