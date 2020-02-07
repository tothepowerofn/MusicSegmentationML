import keras as k
from keras.callbacks import ModelCheckpoint
from keras.models import Model
from keras.layers import (BatchNormalization, Conv1D, Dense, Input,
    TimeDistributed, Activation, Bidirectional, SimpleRNN, GRU, LSTM)
from keras.utils import to_categorical
from keras.layers import Concatenate
from numpy import zeros, newaxis
from keras.models import clone_model
import pickle
from keras.layers import Lambda
from keras.models import load_model
from keras.backend import slice

#https://github.com/JackBurdick/ASR_DL/blob/master/sample_models.py roughly used as a starting point
from feat_extract import DataGenerator


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

class ModelEvaluator:
    def __init__(self, featureFolderPath, labelFolderPath, featureList, coalesceInput=False):
        self.featureFolderPath = featureFolderPath
        self.labelFolderPath = labelFolderPath
        self.featureList = featureList
        self.coalesceInput = coalesceInput
    def trainWithKFoldEval(self, model, k, modelName, epochs, saveBestOnly=True):
        self.model = model
        saveMode = "max"
        generator = DataGenerator(self.featureFolderPath, self.labelFolderPath, self.featureList, coalesceInput=self.coalesceInput)
        filenameLists = generator.generateKFoldLists(k)
        savedListsFile = open(modelName + "-folds.p", 'wb')
        pickle.dump(filenameLists, savedListsFile)
        modelList = []
        currListNum=0
        for filenameList in filenameLists:
            print(">> List " + str(currListNum))
            for filename in filenameList:
                print(filename)
            currListNum += 1
        for j in range(0, k):
            clonedModel = clone_model(self.model)
            clonedModel.set_weights(self.model.get_weights())
            modelList.append(clonedModel)
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        for i in range(1,epochs+1):
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
                currentModel.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
                checkpoint = ModelCheckpoint(modelName + "-fold-" + str(j) + ".h5", monitor='categorical_accuracy', verbose=1,
                                             save_best_only=saveBestOnly, mode=saveMode)
                callbacks_list = [checkpoint]
                history = currentModel.fit(generator.generateFromList(currentTrainingList), epochs=1, steps_per_epoch=numberOfFeatFiles,
                                    callbacks=callbacks_list)
                print(">> Now testing model on fold " + str(j))
                results = currentModel.evaluate(generator.generateFromList(filenameLists[j]),
                                                steps=len(filenameLists[j]))
                print("Model had validation accuracy " + str(results[1]) + " and loss " + str(
                    results[0]) + " on fold " + str(j))
                accuracies.append(results[1])
                print("----------------------------------------------------------------------------")
                print(" ")
            print("This epoch (" + str(i) + ") had average validation accuracy " + str(sum(accuracies)/len(accuracies)))
            print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    def trainWithSavedKFoldEval(self, modelName, epochs, saveBestOnly):
        saveMode = "max"
        generator = DataGenerator(self.featureFolderPath, self.labelFolderPath, self.featureList,
                                  coalesceInput=self.coalesceInput)
        filenameLists = pickle.load( open(modelName + "-folds.p", "rb" ) )
        modelList = []
        currListNum = 0
        for filenameList in filenameLists:
            print(">> List " + str(currListNum))
            for filename in filenameList:
                print(filename)
            currListNum += 1
        for j in range(0, len(filenameLists)):
            openedModel = load_model(modelName + "-fold-" + str(j) + ".h5")
            modelList.append(openedModel)
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        for i in range(1, epochs + 1):
            accuracies = []
            for j in range(0, len(filenameLists)):
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
                currentModel.compile(loss='categorical_crossentropy', optimizer='adam',
                                     metrics=['categorical_accuracy'])
                checkpoint = ModelCheckpoint(modelName + "-fold-" + str(j) + ".h5", monitor='categorical_accuracy',
                                             verbose=1,
                                             save_best_only=saveBestOnly, mode=saveMode)
                callbacks_list = [checkpoint]
                history = currentModel.fit(generator.generateFromList(currentTrainingList), epochs=1,
                                           steps_per_epoch=numberOfFeatFiles,
                                           callbacks=callbacks_list)
                print(">> Now testing model on fold " + str(j))
                results = currentModel.evaluate(generator.generateFromList(filenameLists[j]),
                                                steps=len(filenameLists[j]))
                print("Model had validation accuracy " + str(results[1]) + " and loss " + str(
                    results[0]) + " on fold " + str(j))
                accuracies.append(results[1])
                print("----------------------------------------------------------------------------")
                print(" ")
            print("This epoch (" + str(i) + ") had average validation accuracy " + str(sum(accuracies)/len(accuracies)))
            print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")