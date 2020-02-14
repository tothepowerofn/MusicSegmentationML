import librosa
import librosa.display
import numpy as np
from numpy import genfromtxt
from numpy import zeros, newaxis
from keras.utils import to_categorical
import os
import math
import random


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


class Feature:
    def extract(self):
        pass
    def copy(self, newName):
        pass
    def getName(self):
        pass
    def extract(self):
        pass
    def save(self, featureBasePath):
        pass

class MFCCFeature(Feature):
    def __init__(self, featureName, dataPath, S=None, n_mfcc=20, dct_type=2, norm='ortho', lifter=0, normalize = True, **kwargs):
        self.S = S
        self.n_mfcc = n_mfcc
        self.dct_type = dct_type
        self.norm = norm
        self.lifter = lifter
        self.normalize = True
        self.kwargs = kwargs
        self.dataPath = dataPath
        self.songs = {}
        self.featureName = featureName

    def copy(self, newName): #Initialize as newly named feature with old data
        newInstance = MFCCFeature(self.featureName, self.dataPath, self.S, self.n_mfcc, self.dct_type, self.norm, self.lifter, self.normalize, self.kwargs)
        newInstance.songs = self.songs
        return newInstance
    def getName(self):
        return self.featureName

    def extractMFCCForWav(self, wavPath, S=None, n_mfcc=20, dct_type=2, norm='ortho', lifter=0, normalize=True, **kwargs):
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
    def save(self, featureBasePath):
        if not os.path.exists(featureBasePath):
            os.makedirs(featureBasePath)
        if not os.path.exists(featureBasePath + "/" + self.featureName):
            os.makedirs(featureBasePath + "/" + self.featureName)
        if not self.songs:
            self.extract()
        for name, features in self.songs.items():
            np.savetxt(featureBasePath + "/" + self.featureName + "/" + name + "-" + self.featureName + ".csv", features, delimiter=",")


class Pooling1DFeature(Feature): #this is NOT relate to max-pooling.
    def __init__(self, featureName, featureToPool, numSamples):
        self.featureToPool = featureToPool
        self.numSamples = numSamples
        self.featureName = featureName
        if numSamples < 2:
            raise Exception("You need to pool more than 1 sample! It makes no sense to pool 1 sample, just use the feature itself in that case!")
        self.extractedFeatures = {}
    def copy(self, newName): #Initialize as newly named feature with old data
        newInstance = Pooling1DFeature(newName, self.featureToPool, self.numSamples)
        newInstance.extractedFeatures = self.extractedFeatures
        return newInstance

    def getName(self):
        return self.featureName

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
    def save(self, featureBasePath):
        if not os.path.exists(featureBasePath):
            os.makedirs(featureBasePath)
        if not os.path.exists(featureBasePath + "/" + self.featureName):
            os.makedirs(featureBasePath + "/" + self.featureName)
        if not self.extractedFeatures:
            self.extract()
        for name, features in self.extractedFeatures.items():
            np.savetxt(featureBasePath + "/" + self.featureName + "/" + name + "-" + self.featureName + ".csv", features, delimiter=",")

class DelayFeature(Feature):
    def __init__(self, featureName, featureToDelay, numSamplesDelay):
        self.featureName = featureName
        self.featureToDelay = featureToDelay
        self.numSamplesDelay = numSamplesDelay
        if numSamplesDelay < 1:
            raise Exception(
                "You need to delay for more than 1 sample! It makes no sense to delay by 0 samples, just use the feature itself in that case!")
        self.extractedFeatures = {}
    def copy(self, newName):
        newInstance = DelayFeature(self.featureName, self.featureToDelay, self.numSamplesDelay)
        newInstance.extractedFeatures = self.extractedFeatures

    def getName(self):
        return self.featureName

    def extractSingle(self, name):
        featureDict = self.featureToDelay.extract()
        features = featureDict[name]
        base = np.zeros((self.numSamplesDelay, features.shape[1]))
        return np.vstack([base, features[0:features.shape[0]-self.numSamplesDelay,:]])
    def extract(self):
        if self.extractedFeatures:
            return self.extractedFeatures
        else:
            for name in self.featureToDelay.extract().keys():
                self.extractedFeatures[name] = self.extractSingle(name)
            return self.extractedFeatures
    def save(self, featureBasePath):
        if not os.path.exists(featureBasePath):
            os.makedirs(featureBasePath)
        if not os.path.exists(featureBasePath + "/" + self.featureName):
            os.makedirs(featureBasePath + "/" + self.featureName)
        if not self.extractedFeatures:
            self.extract()
        for name, features in self.extractedFeatures.items():
            np.savetxt(featureBasePath + "/" + self.featureName + "/" + name + "-" + self.featureName + ".csv", features, delimiter=",")

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
    def saveLabels(self, labelPath, featuresDictList):
        if not os.path.exists(labelPath):
            os.makedirs(labelPath)
        for name in self.getClassifiableDataNames():
            labels = self.labelByName(name, featuresDictList[0][1][name])
            np.savetxt(labelPath + "/" + name + "-labels.csv", labels, delimiter=",")



def generateLabeledFeatures(labeler, featureList):
    dictList = []
    generatedLabeledFeaturesDict = {}
    for feature in featureList:
        dictList.append((feature.getName(), feature.extract()))
    for name in labeler.getClassifiableDataNames():
        combinedFeatures = np.hstack((dict[name][1] for dict in dictList))
        combinedFeaturesAndClassifications = np.hstack((combinedFeatures, labeler.labelByName(name, combinedFeatures)))
        generatedLabeledFeaturesDict[name] = combinedFeaturesAndClassifications
    return generatedLabeledFeaturesDict

def saveTrainingData(featureBasePath, featureList, labeler):
    dictList = []
    generatedLabeledFeaturesDict = {}
    for feature in featureList:
        dictList.append((feature.getName(), feature.extract()))
        feature.save(featureBasePath)
    labeler.saveLabels(featureBasePath + "/" + "labels", dictList)

def saveFeatures(generatedFeaturesDict, featureOutputFolderPath):
    for name, features in generatedFeaturesDict.items():
        np.savetxt(featureOutputFolderPath + "/" + name + "-feats.csv", features, delimiter=",")


class DataGeneratorModule():
    def __init__(self, featureFolderBasePath):
        self.featureFolderBasePath = featureFolderBasePath
    def loadForFile(self, filepath):
        pass
    def load(self, name):
        pass

class PooledDataGeneratorModule(DataGeneratorModule):
    def __init__(self, stepsToPool, featureFolderBasePath, featureFolderName):
        self.stepsToPool = stepsToPool
        super().__init__(featureFolderBasePath)
        self.featureFolderName = featureFolderName
        if stepsToPool < 1:
            raise("You need to pool at least 1 time step!")
    def loadForFile(self, filepath):
        features = genfromtxt(filepath, delimiter=',')
        feats_list = []
        for i in range(0,self.stepsToPool):
            base = np.zeros((i, features.shape[1]))
            stacked = np.vstack([base, features[0:features.shape[0]-i,:]])
            feats_list.append( stacked[None, :, :] )
        return feats_list
    def load(self, name):
        featureFilePath = self.featureFolderBasePath + "/" + self.featureFolderName + "/" + name + "-" + self.featureFolderName + ".csv"
        return self.loadForFile(featureFilePath)


class ModularDataGenerator():
    def __init__(self, featuresBasePath, labelsFolderName):
        self.featuresBasePath = featuresBasePath
        self.labelsFolderName = labelsFolderName
    def getNumberOfFeatFiles(self):
        return len([f for f in os.listdir(self.featuresBasePath + "/" + self.labelsFolderName) if
                    f.endswith('.csv') and os.path.isfile(os.path.join(self.featuresBasePath + "/" + self.labelsFolderName, f))])

    def chunkIt(self, seq, num): #https://stackoverflow.com/questions/2130016/splitting-a-list-into-n-parts-of-approximately-equal-length
        avg = len(seq) / float(num)
        out = []
        last = 0.0

        while last < len(seq):
            out.append(seq[int(last):int(last + avg)])
            last += avg

        return out
    def generateKFoldLists(self, k):
        filenames = os.listdir(self.featuresBasePath + "/" + self.labelsFolderName)
        csvFilenames = []
        for filename in filenames:
            if filename.endswith(".csv"):
                csvFilenames.append(filename)
        random.shuffle(csvFilenames)
        return self.chunkIt(csvFilenames, k)
    def generateFromList(self, filenameList, modulesList):
        while True:
            for filename in filenameList:
                if filename.endswith(".csv"):
                    name = (os.path.splitext(filename)[0]).split("-" + self.labelsFolderName)[0]
                    labelFilepath = self.featuresBasePath + "/" + self.labelsFolderName + "/" + name + "-" + self.labelsFolderName + ".csv"
                    labs = genfromtxt(labelFilepath, delimiter=',')
                    featuresList = []
                    for module in modulesList:
                        featuresList.extend(module.load(name))
                    labels = to_categorical(labs)[None, :, :]
                    yield (featuresList, labels)