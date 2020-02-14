import keras as k
from keras.callbacks import ModelCheckpoint
from keras.models import Model
from keras.layers import (BatchNormalization, Conv1D, Dense, Input,
                          TimeDistributed, Activation, Bidirectional, SimpleRNN, GRU, LSTM, MaxPooling1D)
from keras.utils import to_categorical
from keras.layers import Concatenate
from numpy import zeros, newaxis
from keras.models import clone_model
import pickle
from keras.layers import Lambda
from keras.layers import Dropout, Conv2D, Flatten
from keras.models import load_model
from keras.backend import slice, stack
import math

#https://github.com/JackBurdick/ASR_DL/blob/master/sample_models.py roughly used as a starting point
from feat_extract import ModularDataGenerator


def stupidSimpleRNNModel(inputDimension, numPerRecurrentLayer, numRecurrentLayers, outputDimension, numConvFilters=250, kernelSize=11):
    #Input Layer
    inputLayer = Input(shape=(None, inputDimension))
    #Convolutional Layer
    convLayer = Conv1D(filters=numConvFilters, kernel_size=kernelSize,
                       strides=1,
                       padding='same', # I think that 'same' is important to detecting "Intro" cues. See how Keras defines 'same'.
                       activation='elu')(inputLayer)
    currentRecurrentLayerInput = convLayer
    for i in range(0, numRecurrentLayers):
        rnnLayer = GRU(numPerRecurrentLayer, activation='relu', return_sequences=True, implementation=2)(currentRecurrentLayerInput)
        currentRecurrentLayerInput = rnnLayer
    timeDistLayer = TimeDistributed(Dense(outputDimension))(currentRecurrentLayerInput)
    outputLayer = Activation('softmax', name='softmax')(timeDistLayer)

    #Defining the actual model
    model = Model(inputs=inputLayer, outputs=outputLayer)
    return model

def dumbSimpleRNNModel(inputDimension, numPerRecurrentLayer, numRecurrentLayers, outputDimension, numConvFilters=250, kernelSize=11):
    #Input Layer
    inputLayer = Input(shape=(None, inputDimension))
    #Convolutional Layer
    convLayer = Conv1D(filters=numConvFilters, kernel_size=kernelSize,
                       strides=1,
                       padding='same',
                       activation='elu')(inputLayer)
    batchNormConvLayer = BatchNormalization()(convLayer)
    currentRecurrentLayerInput = batchNormConvLayer
    for i in range(0, numRecurrentLayers):
        rnnLayer = GRU(numPerRecurrentLayer, activation='relu', return_sequences=True, implementation=2)(currentRecurrentLayerInput)
        batchNormRNNLayer = BatchNormalization()(rnnLayer)
        currentRecurrentLayerInput = batchNormRNNLayer
    timeDistLayer = TimeDistributed(Dense(outputDimension))(currentRecurrentLayerInput)
    outputLayer = Activation('softmax', name='softmax')(timeDistLayer)

    #Defining the actual model
    model = Model(inputs=inputLayer, outputs=outputLayer)
    return model

def stupidSimpleRNNModelTimeDisLess(inputDimension, numPerRecurrentLayer, numRecurrentLayers, outputDimension, numConvFilters=250, kernelSize=11):
    #Input Layer
    inputLayer = Input(shape=(None, inputDimension))
    #Convolutional Layer
    convLayer = Conv1D(filters=numConvFilters, kernel_size=kernelSize,
                       strides=1,
                       padding='same', # I think that 'same' is important to detecting "Intro" cues. See how Keras defines 'same'.
                       activation='elu')(inputLayer)
    currentRecurrentLayerInput = convLayer
    for i in range(0, numRecurrentLayers):
        rnnLayer = GRU(numPerRecurrentLayer, activation='relu', return_sequences=True, implementation=2)(currentRecurrentLayerInput)
        currentRecurrentLayerInput = rnnLayer
    outputLayer = Dense(outputDimension, activation='softmax', name='softmax')(currentRecurrentLayerInput)

    #Defining the actual model
    model = Model(inputs=inputLayer, outputs=outputLayer)
    return model

def modelV1(inputDimension, numPerRecurrentLayer, numRecurrentLayers, outputDimension, numConvFilters=250, kernelSize=11, stride=1):
    #Input Layer
    inputLayer = Input(shape=(None, inputDimension))
    #Convolutional Layer
    convLayer = Conv1D(filters=numConvFilters, kernel_size=kernelSize,
                       strides=stride,
                       padding='same', # I think that 'same' is important to detecting "Intro" cues. See how Keras defines 'same'.
                       activation='elu')(inputLayer)
    currentRecurrentLayerInput = convLayer
    for i in range(0, numRecurrentLayers):
        rnnLayer = GRU(numPerRecurrentLayer, activation='relu', return_sequences=True, implementation=2)(currentRecurrentLayerInput)
        currentRecurrentLayerInput = rnnLayer
    outputLayer = Dense(outputDimension, activation='softmax', name='softmax')(currentRecurrentLayerInput)

    #Defining the actual model
    model = Model(inputs=inputLayer, outputs=outputLayer)
    return model

def modelV4(inputDimension, numPerRecurrentLayer, numRecurrentLayers, outputDimension, numConvFilters=250, kernelSize=11, stride=1):
    #Input Layer
    inputLayer = Input(shape=(None, inputDimension))
    #Convolutional Layer
    convLayer = Conv1D(filters=numConvFilters, kernel_size=kernelSize,
                       strides=stride,
                       padding='same', # I think that 'same' is important to detecting "Intro" cues. See how Keras defines 'same'.
                       activation='elu')(inputLayer)
    denseLayer = Dense(50, activation='relu')(convLayer)
    currentRecurrentLayerInput = denseLayer
    for i in range(0, numRecurrentLayers):
        rnnLayer = GRU(numPerRecurrentLayer, activation='relu', return_sequences=True, implementation=2)(currentRecurrentLayerInput)
        currentRecurrentLayerInput = rnnLayer
    outputLayer = Dense(outputDimension, activation='softmax', name='softmax')(currentRecurrentLayerInput)

    #Defining the actual model
    model = Model(inputs=inputLayer, outputs=outputLayer)
    return model

def poolingConvModel(numInputs, perInputDimension, numPerRecurrentLayer, numRecurrentLayers, numDenseLayerUnits, outputDimension, numConvFiltersPerConv=250, kernelSizePerConv=11, stride=1):
    #Input Layers
    inputLayerList = []
    for i in range(0, numInputs):
        inputLayerList.append(Input(shape=(None, perInputDimension)))
    #Convolutional Layers
    convLayerList = []
    for inputLayer in inputLayerList:
        convLayer = Conv1D(filters=numConvFiltersPerConv, kernel_size=kernelSizePerConv,
                           strides=stride,
                           padding='same',
                           # I think that 'same' is important to detecting "Intro" cues. See how Keras defines 'same'.
                           activation='elu')(inputLayer)
        convLayerList.append(convLayer)
    #Concat Layer
    merged = Concatenate()(convLayerList)
    #Dense Layer
    denseLayer = Dense(numDenseLayerUnits, activation='relu')(merged)
    currentRecurrentLayerInput = denseLayer
    for i in range(0, numRecurrentLayers):
        rnnLayer = GRU(numPerRecurrentLayer, activation='relu', return_sequences=True, implementation=2)(
            currentRecurrentLayerInput)
        currentRecurrentLayerInput = rnnLayer
    outputLayer = Dense(outputDimension, activation='softmax', name='softmax')(currentRecurrentLayerInput)

    # Defining the actual model
    model = Model(inputs=inputLayerList, outputs=outputLayer)
    return model

def poolingRecConvModel(numInputs, perInputDimension, numPerConvRecurrentLayer, numPerRecurrentLayer, numRecurrentLayers, numDenseLayerUnits, outputDimension, numConvFiltersPerConv=250, kernelSizePerConv=11, stride=1):
    #Input Layers
    inputLayerList = []
    for i in range(0, numInputs):
        inputLayerList.append(Input(shape=(None, perInputDimension)))
    #Convolutional Layers
    convRecLayerList = []
    for inputLayer in inputLayerList:
        convLayer = Conv1D(filters=numConvFiltersPerConv, kernel_size=kernelSizePerConv,
                           strides=stride,
                           padding='same',
                           # I think that 'same' is important to detecting "Intro" cues. See how Keras defines 'same'.
                           activation='elu')(inputLayer)
        convRecLayer = GRU(numPerRecurrentLayer, activation='relu', return_sequences=True, implementation=2)(convLayer)
        convRecLayerList.append(convRecLayer)
    #Concat Layer
    merged = Concatenate()(convRecLayerList)
    #Dense Layer
    denseLayer = Dense(numDenseLayerUnits, activation='relu')(merged)
    currentRecurrentLayerInput = denseLayer
    for i in range(0, numRecurrentLayers):
        rnnLayer = GRU(numPerRecurrentLayer, activation='relu', return_sequences=True, implementation=2)(
            currentRecurrentLayerInput)
        currentRecurrentLayerInput = rnnLayer
    outputLayer = Dense(outputDimension, activation='softmax', name='softmax')(currentRecurrentLayerInput)

    # Defining the actual model
    model = Model(inputs=inputLayerList, outputs=outputLayer)
    return model

def poolingConvModelWithDropout(numInputs, perInputDimension, numPerRecurrentLayer, numRecurrentLayers, numDenseLayerUnits, outputDimension, dropoutRate=0.2, numConvFiltersPerConv=250, kernelSizePerConv=11, stride=1):
    #Input Layers
    inputLayerList = []
    for i in range(0, numInputs):
        inputLayerList.append(Input(shape=(None, perInputDimension)))
    #Convolutional Layers
    convLayerList = []
    for inputLayer in inputLayerList:
        convLayer = Conv1D(filters=numConvFiltersPerConv, kernel_size=kernelSizePerConv,
                           strides=stride,
                           padding='same',
                           # I think that 'same' is important to detecting "Intro" cues. See how Keras defines 'same'.
                           activation='elu')(inputLayer)
        convLayerList.append(convLayer)
    #Concat Layer
    merged = Concatenate()(convLayerList)
    dropoutLayerConv = Dropout(dropoutRate)(merged)
    #Dense Layer
    denseLayer = Dense(numDenseLayerUnits, activation='relu')(dropoutLayerConv)
    dropoutLayer1 = Dropout(dropoutRate)(denseLayer)
    currentRecurrentLayerInput = dropoutLayer1
    for i in range(0, numRecurrentLayers):
        rnnLayer = GRU(numPerRecurrentLayer, activation='relu', return_sequences=True, implementation=2)(
            currentRecurrentLayerInput)

        currentRecurrentLayerInput = rnnLayer
    dropoutLayer2 = Dropout(dropoutRate)(currentRecurrentLayerInput)
    outputLayer = Dense(outputDimension, activation='softmax', name='softmax')(dropoutLayer2)

    # Defining the actual model
    model = Model(inputs=inputLayerList, outputs=outputLayer)
    return model

def trainWithModelSingleSong(model, features, classifications, epochs):
    x_train = features[newaxis,:,:]
    y_train = to_categorical(classifications)[newaxis, :,:,]
    print(x_train.shape)
    print(y_train.shape)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
    history = model.fit(x_train, y_train, epochs=epochs)

def trainModel(model, features, classifications, epochs):
    x_train = features
    y_train = classifications
    print(x_train.shape)
    print(y_train.shape)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
    history = model.fit(x_train, y_train, epochs=epochs)

def trainModelWithGenerator(model, generator, modelName, epochs):
    modelSaveName = modelName + ".h5"
    numberOfFeatFiles = generator.getNumberOfFeatFiles()
    print(numberOfFeatFiles)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
    checkpoint = ModelCheckpoint(modelSaveName, monitor='categorical_accuracy', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    history = model.fit(generator.generate(), epochs=epochs, steps_per_epoch = numberOfFeatFiles, callbacks=callbacks_list)
    #model.save(modelSaveName)

######################################################################
#                          UPDATED OOP CODE                          #
######################################################################

class MModel:
    def __init__(self, modelName):
        self.modelName = modelName
        self.model = None

    def load(self, savedModelPath):
        self.savedModelPath = savedModelPath
        self.model = load_model(savedModelPath)
    def build(self):
        pass
    def clone(self, newName):
        clonedKModel = clone_model(self.model)
        clonedKModel.set_weights(self.model.get_weights())
        newModel = MModel(self.modelName)
        newModel.model = clonedKModel
        newModel.modelName = newName
        return newModel
    def compile(self, **kwargs):
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
    def summary(self):
        self.model.summary()
    def fitWithGenerator(self, generator, generatorList, modulesList, epochs, steps_per_epoch, saveBestOnly=False, **kwargs):
        checkpoint = ModelCheckpoint(self.modelName + ".h5", monitor='categorical_accuracy', verbose=1,
                                     save_best_only=saveBestOnly, mode="max")
        callbacks_list = [checkpoint]
        history = self.model.fit(generator.generateFromList(generatorList, modulesList), epochs=epochs,
                                   steps_per_epoch=steps_per_epoch,
                                   callbacks=callbacks_list)
        return history

    def evaluateWithGenerator(self, generator, generatorList, modulesList, steps):
        results = self.model.evaluate(generator.generateFromList(generatorList, modulesList),
                              steps=steps)
        return results

class PoolingConvModelWithDropout(MModel):
    def build(self, dropoutRate=0.5, numInputs=400, perInputDimension=10, numPerRecurrentLayer=60, numRecurrentLayers=2,
              numDenseLayerUnits=40, outputDimension=6, numConvFiltersPerConv=250, kernelSizePerConv=5, stride=1):
        # Input Layers
        inputLayerList = []
        for i in range(0, numInputs):
            inputLayerList.append(Input(shape=(None, perInputDimension)))
        # Convolutional Layers
        convLayerList = []
        for inputLayer in inputLayerList:
            convLayer = Conv1D(filters=numConvFiltersPerConv, kernel_size=kernelSizePerConv,
                               strides=stride,
                               padding='same',
                               # I think that 'same' is important to detecting "Intro" cues. See how Keras defines 'same'.
                               activation='elu')(inputLayer)
            convLayerList.append(convLayer)
        # Concat Layer
        merged = Concatenate()(convLayerList)
        dropoutLayerMerged = Dropout(dropoutRate)(merged)
        # Dense Layer
        denseLayer = Dense(numDenseLayerUnits, activation='relu')(dropoutLayerMerged)
        dropoutLayer1 = Dropout(dropoutRate)(denseLayer)
        currentRecurrentLayerInput = dropoutLayer1
        for i in range(0, numRecurrentLayers):
            rnnLayer = GRU(numPerRecurrentLayer, activation='relu', return_sequences=True, implementation=2)(
                currentRecurrentLayerInput)

            currentRecurrentLayerInput = rnnLayer
        dropoutLayer2 = Dropout(dropoutRate)(currentRecurrentLayerInput)
        outputLayer = Dense(outputDimension, activation='softmax', name='softmax')(dropoutLayer2)

        # Defining the actual model
        model = Model(inputs=inputLayerList, outputs=outputLayer)
        self.model = model

class PoolingModelWithDropout(MModel):
    def build(self, dropoutRate=0.5, numInputs=400, perInputDimension=10, numPerRecurrentLayer=60, numRecurrentLayers=2,
              numDenseLayerUnits=40, outputDimension=6):
        # Input Layers
        inputLayerList = []
        for i in range(0, numInputs):
            inputLayerList.append(Input(shape=(None, perInputDimension)))
        # Concat Layer
        merged = Concatenate()(inputLayerList)
        dropoutLayerConv = Dropout(dropoutRate)(merged)
        # Dense Layer
        denseLayer = Dense(numDenseLayerUnits, activation='relu')(dropoutLayerConv)
        dropoutLayer1 = Dropout(dropoutRate)(denseLayer)
        currentRecurrentLayerInput = dropoutLayer1
        for i in range(0, numRecurrentLayers):
            rnnLayer = GRU(numPerRecurrentLayer, activation='relu', return_sequences=True, implementation=2)(
                currentRecurrentLayerInput)

            currentRecurrentLayerInput = rnnLayer
        dropoutLayer2 = Dropout(dropoutRate)(currentRecurrentLayerInput)
        outputLayer = Dense(outputDimension, activation='softmax', name='softmax')(dropoutLayer2)

        # Defining the actual model
        model = Model(inputs=inputLayerList, outputs=outputLayer)
        self.model = model

class FadingPoolingModelWithDropout(MModel):
    def build(self, dropoutRate=0.5, numInputs=400, perInputDimension=10, fadingMaxUnits=None, perInputDenseUnitFadingList=None, numPerRecurrentLayer=60, numRecurrentLayers=2,
              numDenseLayerUnits=40, outputDimension=6):
        gradingList = perInputDenseUnitFadingList
        if not gradingList:
            gradingList = []
            for i in range(numInputs,-1,-1):
                it = i
                if it is 0:
                    it = 1
                currentLog = math.log(it, numInputs)*fadingMaxUnits
                numUnits = math.ceil(currentLog)
                if numUnits is 0:
                    numUnits = 1
                gradingList.append(numUnits)
        # Input Layers
        inputLayerList = []
        inputDenseList = []
        for i in range(0, numInputs):
            input = Input(shape=(None, perInputDimension))
            inputLayerList.append(input)
            denseInputLayer = Dense(gradingList[i], activation='relu')(input)
            inputDenseList.append(denseInputLayer)

        # Concat Layer
        merged = Concatenate()(inputDenseList)
        dropoutLayerConv = Dropout(dropoutRate)(merged)
        # Dense Layer
        denseLayer = Dense(numDenseLayerUnits, activation='relu')(dropoutLayerConv)
        dropoutLayer1 = Dropout(dropoutRate)(denseLayer)
        currentRecurrentLayerInput = dropoutLayer1
        for i in range(0, numRecurrentLayers):
            rnnLayer = GRU(numPerRecurrentLayer, activation='relu', return_sequences=True, implementation=2)(
                currentRecurrentLayerInput)

            currentRecurrentLayerInput = rnnLayer
        dropoutLayer2 = Dropout(dropoutRate)(currentRecurrentLayerInput)
        outputLayer = Dense(outputDimension, activation='softmax', name='softmax')(dropoutLayer2)

        # Defining the actual model
        model = Model(inputs=inputLayerList, outputs=outputLayer)
        self.model = model


class ModelEvaluator:
    def __init__(self, featureFolderPath, labelFolderPath):
        self.featureFolderPath = featureFolderPath
        self.labelFolderPath = labelFolderPath

    def trainKFolds(self, modelName, modelList, filenameLists, modulesList, generator, epochs, saveBestOnly=False):
        currListNum = 0
        k = len(modelList)
        for filenameList in filenameLists:
            print(">> List " + str(currListNum))
            for filename in filenameList:
                print(filename)
            currListNum += 1
        for model in modelList:
            model.compile()
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        for i in range(1, epochs + 1):
            accuracies = []
            for j in range(0, k):
                print("----------------------------------------------------------------------------")
                print(">> Currently on epoch " + str(i) + " of " + str(epochs) + " with fold " + str(j))
                print(">> Now training folds that are not " + str(j))
                currentTrainingList = []
                for l in range(0, len(filenameLists)):
                    if l is not j:
                        for filename in filenameLists[l]:
                            currentTrainingList.append(filename)
                currentModel = modelList[j]
                numberOfFeatFiles = len(currentTrainingList)
                history = currentModel.fitWithGenerator(generator=generator, generatorList=currentTrainingList, modulesList=modulesList,
                                                        epochs=1, steps_per_epoch=numberOfFeatFiles, saveBestOnly=saveBestOnly)
                print(">> Now testing model on fold " + str(j))
                results = currentModel.evaluateWithGenerator(generator, filenameLists[j], modulesList, len(filenameLists[j]))
                print("Model had validation accuracy " + str(results[1]) + " and loss " + str(
                    results[0]) + " on fold " + str(j))
                accuracies.append(results[1])
                print("----------------------------------------------------------------------------")
                print(" ")
            print(
                "This epoch (" + str(i) + ") had average validation accuracy " + str(sum(accuracies) / len(accuracies)))
            print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

    def trainWithKFoldEval(self, model, k, modelName, modulesList, epochs, saveBestOnly=True):
        generator = ModularDataGenerator(self.featureFolderPath, self.labelFolderPath)
        filenameLists = generator.generateKFoldLists(k)
        savedListsFile = open(modelName + "-folds.p", 'wb')
        pickle.dump(filenameLists, savedListsFile)
        savedModulesListFile = open(modelName + "-moduleslist.p", 'wb')
        pickle.dump(modulesList, savedModulesListFile)
        modelList = []
        for j in range(0, k):
            clonedModel = model.clone(modelName + "-fold-" + str(j))
            clonedModel.compile()
            modelList.append(clonedModel)
        self.trainKFolds(modelName, modelList, filenameLists, modulesList, generator, epochs, saveBestOnly)
    def trainWithSavedKFoldEval(self, modelName, epochs, saveBestOnly):
        generator = ModularDataGenerator(self.featureFolderPath, self.labelFolderPath)
        filenameLists = pickle.load(open(modelName + "-folds.p", "rb"))
        modulesList = pickle.load(open(modelName + "-moduleslist.p", "rb"))
        modelList = []
        for j in range(0, len(filenameLists)):
            openedModel = MModel(modelName)
            openedModel.load(modelName + "-fold-" + str(j) + ".h5")
            openedModel.compile()
            modelList.append(openedModel)
        self.trainKFolds(modelName, modelList, filenameLists, modulesList, generator, epochs, saveBestOnly)
