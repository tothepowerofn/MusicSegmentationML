import pickle

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
    def updateProperties(self, examplesProperties):
        self.examplesProperties = examplesProperties
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
    def __init__(self, featureName, featureToPool, numSamples, **kwargs):
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
    def __init__(self, featureName, featureToDelay, numSamplesDelay, **kwargs):
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

class SegmentOrderFeature(Feature):
    def __init__(self, featureName, dataPath, numClasses, maxLength, **kwargs):
        self.extractedFeatures = None
        self.featureName = featureName
        self.dataPath = dataPath
        self.numClasses = numClasses
        self.extractedFeatures = {}
        self.maxLength = maxLength
    def copy(self, newName):
        newInstance = SegmentOrderFeature(self.featureName)
        newInstance.extractedFeatures = self.extractedFeatures
    def getName(self):
        return self.featureName
    def extractSingle(self, filepath):
        annotationData = genfromtxt(filepath, delimiter=',')
        segmentList = annotationData[:, 1:2]
        segmentList2D = to_categorical(segmentList, self.numClasses)
        if segmentList2D.shape[0] < self.maxLength:
            return np.vstack([segmentList2D, np.zeros((self.maxLength - segmentList2D.shape[0],self.numClasses))])
        else:
            return segmentList2D[0:self.maxLength, :]
    def extract(self):
        if self.extractedFeatures:
            return self.extractedFeatures
        else:
            for filename in os.listdir(self.dataPath):
                if filename.endswith("-annotation.csv"):
                    self.extractedFeatures[filename.split("-annotation.csv")[0]] = self.extractSingle(self.dataPath + "/" + filename)
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

class IndexerModule:
    def getPropertyForExample(self, filepath):
        pass

class SongMetadataIndexerModule(IndexerModule):
    def getPropertyForExample(self, filepath):
        properties = {}
        properties["wav_filepath"] = filepath
        return properties

class MFCCDimensionsIndexerModule(IndexerModule):
    def __init__(self, hop_length, n_mfcc):
        self.hop_length = hop_length
        self.n_mfcc = n_mfcc
    def getPropertyForExample(self, filepath):
        properties = {}
        sound, sampleRate = librosa.load(filepath)
        properties["samples"] = sound.shape[0]
        properties["mfcc_length"] = math.ceil(sound.shape[0]/self.hop_length)
        properties["mfcc_width"] = self.n_mfcc
        return properties

class Indexer:
    def __init__(self, dataPath, modulesList):
        self.dataPath = dataPath
        self.examplesPropertiesDict = {}
        self.modulesList = modulesList
    def getPropertiesForExample(self, filepath, modulesList):
        propertiesDict = {}
        for module in modulesList:
            propertiesDict.update(module.getPropertyForExample(filepath))
        return propertiesDict
    def getPropertiesForExamples(self):
        if self.examplesPropertiesDict:
            return self.examplesPropertiesDict
        for filename in os.listdir(self.dataPath):
            if filename.endswith(".wav"):
                currentExampleName = os.path.basename(filename).split(".wav")[0]
                currentExampleFilepath = self.dataPath + "/" + filename
                currentExampleProperties = self.getPropertiesForExample(currentExampleFilepath, self.modulesList)
                self.examplesPropertiesDict[currentExampleName] = currentExampleProperties
        return self.examplesPropertiesDict
    def saveProperties(self, featureBasePath):
        if not self.examplesPropertiesDict:
            self.getPropertiesForExamples()
        savedExamplePropertiesDictFile = open(featureBasePath + "/" + "properties.p", 'wb')
        pickle.dump(self.examplesPropertiesDict, savedExamplePropertiesDictFile)

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

def saveTrainingData(featureBasePath, featureList, indexer, labeler):
    dictList = []
    examplesProperties = indexer.getPropertiesForExamples()
    for name, exampleProperties in examplesProperties.items():
        print((name, exampleProperties))
    indexer.saveProperties(featureBasePath)
    for feature in featureList:
        feature.updateProperties(examplesProperties)
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

class BasicDataGeneratorModule(DataGeneratorModule):
    def __init__(self, featureFolderBasePath, featureFolderName, outputExtraDim=True):
        self.featureFolderBasePath = featureFolderBasePath
        super().__init__(featureFolderBasePath)
        self.featureFolderName = featureFolderName
        self.outputExtraDim = outputExtraDim
    def loadForFile(self, filepath):
        features = genfromtxt(filepath, delimiter=',')
        if self.outputExtraDim:
            return [features[None, :, :]]
        else:
            return [features]
    def load(self, name):
        featureFilePath = self.featureFolderBasePath + "/" + self.featureFolderName + "/" + name + "-" + self.featureFolderName + ".csv"
        return self.loadForFile(featureFilePath)

class ChunkedMFCCDataGeneratorModule(DataGeneratorModule):
    def __init__(self, featureFolderBasePath, featureFolderName, chunkSize, jumpLength):
        self.featureFolderBasePath = featureFolderBasePath
        self.chunkSize = chunkSize
        self.jumpLength = jumpLength
        super().__init__(featureFolderBasePath)
        self.featureFolderName = featureFolderName
    def loadForFile(self, filepath):
        jumpLength = self.jumpLength
        chunkSize = self.chunkSize
        features1D = genfromtxt(filepath, delimiter=',')
        features1DLength = features1D.shape[0]
        featudes1DWidth = features1D.shape[1]
        beginningArrs = []
        iInPadded = chunkSize-jumpLength
        while iInPadded >= 0:
            zeros = np.zeros((iInPadded, featudes1DWidth))
            feats = features1D[0:chunkSize-iInPadded, :]
            beginArr = np.vstack([zeros, feats])
            beginningArrs.append(beginArr)
            iInPadded -= jumpLength
        padded = np.stack(beginningArrs, axis=0)
        stacked = np.stack((features1D[i:i+chunkSize, :] for i in range((-1)*iInPadded, features1DLength-chunkSize, jumpLength)) , axis=0)
        stackedWithPadded = np.vstack([padded, stacked])
        return [stackedWithPadded]
    def load(self, name):
        featureFilePath = self.featureFolderBasePath + "/" + self.featureFolderName + "/" + name + "-" + self.featureFolderName + ".csv"
        loadedArr = self.loadForFile(featureFilePath)
        return loadedArr

class Delayed2DDataGeneratorModule(DataGeneratorModule):
    def __init__(self, moduleToDelay, stepsToDelay):
        self.moduleToDelay = moduleToDelay
        self.stepsToDelay = stepsToDelay
    def load(self, name):
        featsList = self.moduleToDelay.load(name)
        newFeatsList = []
        for feat in featsList:
            featLength = feat.shape[0]
            padShape = (1, feat.shape[1], feat.shape[2])
            stackedZeros = np.vstack((np.zeros(padShape) for i in range(0, self.stepsToDelay)))
            adjustedFeats = feat[0:featLength-self.stepsToDelay,:,:]
            newFeat = np.vstack([stackedZeros, adjustedFeats])
            newFeatsList.append(newFeat)
            # for arr in newFeat:
            #     print("new")
            #     for ar in arr:
            #         print(ar)
        return newFeatsList


class PooledDataGeneratorModule(DataGeneratorModule):
    def __init__(self, stepsToPool, featureFolderBasePath, featureFolderName, outputExtraDim=True):
        self.stepsToPool = stepsToPool
        super().__init__(featureFolderBasePath)
        self.featureFolderName = featureFolderName
        self.outputExtraDim = outputExtraDim
        if stepsToPool < 1:
            raise("You need to pool at least 1 time step!")
    def loadForFile(self, filepath):
        features = genfromtxt(filepath, delimiter=',')
        feats_list = []
        for i in range(0,self.stepsToPool):
            base = np.zeros((i, features.shape[1]))
            stacked = np.vstack([base, features[0:features.shape[0]-i,:]])
            if self.outputExtraDim:
                feats_list.append( stacked[None, :, :] )
            else:
                feats_list.append(stacked)
        return feats_list
    def load(self, name):
        featureFilePath = self.featureFolderBasePath + "/" + self.featureFolderName + "/" + name + "-" + self.featureFolderName + ".csv"
        return self.loadForFile(featureFilePath)


class SegmentOrderDataGeneratorModule(DataGeneratorModule):
    def __init__(self, featureFolderBasePath, featureFolderName, outputExtraDim=True):
        super().__init__(featureFolderBasePath)
        self.featureFolderName = featureFolderName
        self.outputExtraDim = outputExtraDim
    def loadForFile(self, filePath, examplesProperties):
        features = genfromtxt(filePath, delimiter=',')
        name = os.path.basename(filePath).split("-" + self.featureFolderName + ".csv")[0]
        numRows = examplesProperties[name]["mfcc_length"]
        featuresRow = features.flatten()[None, :]
        finalFeatures = np.repeat(featuresRow, numRows, axis=0)
        if(self.outputExtraDim):
            return [finalFeatures[None, :, :]]
        else:
            return [finalFeatures]
    def load(self, name):
        featureFilePath = self.featureFolderBasePath + "/" + self.featureFolderName + "/" + name + "-" + self.featureFolderName + ".csv"
        examplesProperties = pickle.load(open(self.featureFolderBasePath + "/" + "properties.p", "rb"))
        return self.loadForFile(featureFilePath, examplesProperties)

class GeneratorLabeler1D:
    def __init__(self, dataPath, sample_rate, hop_length):
        self.dataPath = dataPath
        self.sampleRate = sample_rate
        self.hopLength = hop_length
    def getLabels(self, songName, featuresLength, outputExtraDim=True):
        annotationPath = self.dataPath + "/" + songName + "-annotation.csv"
        annotationData = genfromtxt(annotationPath, delimiter=',')
        numWavSamples = featuresLength
        classifications = np.zeros(shape=(numWavSamples, 1))
        lastHop = 0
        for segment in annotationData:
            currentHop = timestampToHop(segment[0], self.sampleRate, self.hopLength)
            classifications[lastHop:currentHop, 0] = segment[1]
            lastHop = currentHop
        classifications[lastHop:numWavSamples, 0] = classifications[lastHop - 1, 0]
        if outputExtraDim:
            return to_categorical(classifications)[None,:,:]
        else:
            return to_categorical(classifications)

class GeneratorLabeler2D:
    def __init__(self, dataPath, sample_rate, hop_length, jumpLength, numClasses=6):
        self.dataPath = dataPath
        self.sampleRate = sample_rate
        self.hopLength = hop_length
        self.jumpLength = jumpLength
        self.numClasses = numClasses
    def timestampToSampleNumber(time, sampleRate):
        return time * sampleRate
    def timestampToJump(self, time, sampleRate, hopLength, jumpLength):
        return int(math.ceil(timestampToSampleNumber(time, sampleRate) / (hopLength*jumpLength)))

    def getLabels(self, songName, featuresLength):
        annotationPath = self.dataPath + "/" + songName + "-annotation.csv"
        annotationData = genfromtxt(annotationPath, delimiter=',')

        classifications = np.zeros(shape=(featuresLength, 1))
        lastJump = 0
        for segment in annotationData:
            currentJump = self.timestampToJump(segment[0], self.sampleRate, self.hopLength, self.jumpLength)
            classifications[lastJump:currentJump, 0] = segment[1]
            lastJump = currentJump
        if lastJump < featuresLength:
            classifications[lastJump:featuresLength, 0] = classifications[lastJump - 1, 0]
        # for classification in classifications:
        #     print(classification)
        return to_categorical(classifications, self.numClasses)[:, :]

class ModularDataGenerator():
    def __init__(self, featuresBasePath, labelsFolderName, modulesList, generatorLabeler, samplesShapeIndex=1, outputExtraDim=True):
        self.featuresBasePath = featuresBasePath
        self.labelsFolderName = labelsFolderName
        self.outputExtraDim = outputExtraDim
        self.modulesList = modulesList
        self.generatorLabeler = generatorLabeler
        self.samplesShapeIndex = samplesShapeIndex
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
    def generateFromList(self, filenameList):
        while True:
            for filename in filenameList:
                if filename.endswith(".csv"):
                    name = (os.path.splitext(filename)[0]).split("-" + self.labelsFolderName)[0]
                    labelFilepath = self.featuresBasePath + "/" + self.labelsFolderName + "/" + name + "-" + self.labelsFolderName + ".csv"
                    featuresList = []
                    for module in self.modulesList:
                        featuresList.extend(module.load(name))

                    featureLen = featuresList[0].shape[self.samplesShapeIndex]
                    labels = self.generatorLabeler.getLabels(name, featureLen)
                    yield (featuresList, labels)
