import keras as k
from keras.callbacks import ModelCheckpoint
from keras.models import Model
from keras.layers import (BatchNormalization, Conv1D, Dense, Input,
                          TimeDistributed, Activation, Bidirectional, SimpleRNN, GRU, LSTM, MaxPooling1D, Reshape)
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
    def fitWithGenerator(self, generator, generatorList, epochs, steps_per_epoch, saveBestOnly=False, **kwargs):
        checkpoint = ModelCheckpoint(self.modelName + ".h5", monitor='categorical_accuracy', verbose=1,
                                     save_best_only=saveBestOnly, mode="max")
        callbacks_list = [checkpoint]
        history = self.model.fit(generator.generateFromList(generatorList), epochs=epochs,
                                   steps_per_epoch=steps_per_epoch,
                                   callbacks=callbacks_list)
        return history

    def evaluateWithGenerator(self, generator, generatorList, steps):
        results = self.model.evaluate(generator.generateFromList(generatorList),
                              steps=steps)
        return results

class SimpleGRU(MModel):
    def build(self, inputDimension=10, numPerRecurrentLayer=60, numRecurrentLayers=2, outputDimension=6):
        # Input Layer
        inputLayer = Input(shape=(None, inputDimension))
        currentRecurrentLayerInput = inputLayer
        for i in range(0, numRecurrentLayers):
            rnnLayer = GRU(numPerRecurrentLayer, activation='relu', return_sequences=True, implementation=2)(
                currentRecurrentLayerInput)
            currentRecurrentLayerInput = rnnLayer
        outputLayer = Dense(outputDimension, activation='softmax', name='softmax')(currentRecurrentLayerInput)
        # Defining the actual model
        model = Model(inputs=inputLayer, outputs=outputLayer)
        self.model = model

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
              numDenseLayerUnits=40, outputDimension=6, outputExtraDim=True):

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
            if outputExtraDim:
                input = Input(shape=(None, perInputDimension))
                inputLayerList.append(input)
                denseInputLayer = Dense(gradingList[i], activation='relu')(input)
                inputDenseList.append(denseInputLayer)
            else:
                input = Input(shape=(perInputDimension,))
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
        for layer in model.layers:
            print(layer.output_shape)
        self.model = model

class FadingPoolingModelWithDropoutAndSegOrder(MModel):
    def build(self, dropoutRate=0.5, numInputs=400, perInputDimension=10, fadingMaxUnits=None, perInputDenseUnitFadingList=None, numPerRecurrentLayer=60, numRecurrentLayers=2,
              numDenseLayerUnits=40, outputDimension=6, segOrderDimension=6*15):
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
        segOrderLayer = Input(shape=(None, segOrderDimension))
        inputLayerList.append(segOrderLayer)

        # Concat Layer
        merged = Concatenate()(inputDenseList)
        dropoutLayerConv = Dropout(dropoutRate)(merged)
        # Dense Layer
        denseLayer = Dense(numDenseLayerUnits, activation='relu')(dropoutLayerConv)
        dropoutLayer1 = Dropout(dropoutRate)(denseLayer)
        currentRecurrentLayerInput = dropoutLayer1
        for i in range(0, numRecurrentLayers-1):
            rnnLayer = LSTM(numPerRecurrentLayer, activation='relu', return_sequences=True, implementation=2)(
                currentRecurrentLayerInput)

            currentRecurrentLayerInput = rnnLayer
        finalRecurrentLayerConcat = Concatenate()([currentRecurrentLayerInput, segOrderLayer])
        finalRecurrentLayer = rnnLayer = GRU(numPerRecurrentLayer, activation='relu', return_sequences=True, implementation=2)(
                finalRecurrentLayerConcat)
        dropoutLayer2 = Dropout(dropoutRate)(finalRecurrentLayer)
        outputLayer = Dense(outputDimension, activation='softmax', name='softmax')(dropoutLayer2)

        # Defining the actual model
        model = Model(inputs=inputLayerList, outputs=outputLayer)
        self.model = model
class Simple2DTest(MModel):
    def build(self, chunkLength, numFeats, numRecUnits):
        input = Input(shape=(chunkLength, numFeats))
        conv1 = Conv1D(filters=256,
                       kernel_size=8,
                       strides=1,
                       activation='relu', name="conv1D")(input)
        flatten = Flatten(name="flatten1")(conv1)
        dropoutLayer = Dropout(0.5)(flatten)
        reshaped = Reshape((1, flatten._keras_shape[1]))(dropoutLayer)
        rnnLayer = GRU(50, activation='relu', return_sequences=True, implementation=2, recurrent_dropout=0.5)(
            reshaped)
        rnnLayer2 = GRU(50, activation='relu', return_sequences=True, implementation=2, recurrent_dropout=0.5)(
            rnnLayer)
        dropoutLayer2 = Dropout(0.5)(rnnLayer2)
        flatten2 = Flatten()(dropoutLayer2)
        outputLayer = Dense(6, activation='softmax', name='softmax2')(flatten2)
        model = Model(inputs=input, outputs=outputLayer)
        self.model = model

class Faded2DConvModel(MModel):
    def build(self, numClasses, inputShapeList, inputConvFilterNumList, inputConvKernelSizeList, convDenseSizeList, convMaxPoolSizeList,
              convDenseActivation="relu", postConvDropout=None, preRNNDenseSize=None, preRNNDenseActivation="relu",
              preRNNDropout=None, rNNUnitsList=[100,100], rnnDropoutList = [0,0], postRNNDropout=None):
        inputLayersList = []
        convSectionOutputList = []
        for i in range(len(inputShapeList)):
            input = Input(shape=inputShapeList[i])
            inputLayersList.append(input)
            conv = Conv1D(filters=inputConvFilterNumList[i],kernel_size=inputConvKernelSizeList[i],strides=1,
                                  activation='relu')(input)
            n = conv
            if convMaxPoolSizeList[i]:
                n = MaxPooling1D(pool_size=convMaxPoolSizeList[i])(n)
            n = Flatten()(n)
            if convDenseSizeList[i]:
                n = Dense(convDenseSizeList[i], activation=convDenseActivation)(n)
            convSectionOutputList.append(n)
        next = 0
        if(len(convSectionOutputList) > 1):
            next = Concatenate()(convSectionOutputList)
        else:
            next = convSectionOutputList[0]

        if postConvDropout:
            next = Dropout(postConvDropout)(next)
        if preRNNDenseSize:
            next = Dense(preRNNDenseSize, activation=preRNNDenseActivation)(next)
        if preRNNDropout:
            next = Dropout(preRNNDropout)(next)
        reshaped = Reshape((1, next._keras_shape[1]))(next)
        next = reshaped
        for i in range(0, len(rNNUnitsList)):
            next = GRU(rNNUnitsList[i], activation='relu', return_sequences=True, implementation=2, recurrent_dropout=rnnDropoutList[i])(
                next)
        if postRNNDropout:
            next = Dropout(postRNNDropout)(next)
        next = Flatten()(next)
        outputLayer = Dense(numClasses, activation='softmax')(next)
        model = Model(inputs=inputLayersList, outputs=outputLayer)
        self.model = model

class ModelEvaluator:
    def __init__(self, generator):
        self.generator = generator
    def trainKFolds(self, modelName, modelList, filenameLists, epochs, savedAccuracies=None, saveBestOnly=False):
        generator = self.generator
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
        accuraciesAll = savedAccuracies
        if not accuraciesAll:
            accuraciesAll = []
            for i in range(0, k):
                accuraciesAll.append([])

        for i in range(1, epochs + 1):
            thisEpochAccs = []
            for j in range(0, k):
                accuracies = accuraciesAll[j]
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
                history = currentModel.fitWithGenerator(generator=generator, generatorList=currentTrainingList,
                                                        epochs=1, steps_per_epoch=numberOfFeatFiles, saveBestOnly=saveBestOnly)
                print(">> Now testing model on fold " + str(j))
                results = currentModel.evaluateWithGenerator(generator, filenameLists[j], len(filenameLists[j]))
                print("Model had validation accuracy " + str(results[1]) + " and loss " + str(
                    results[0]) + " on fold " + str(j))
                print("----------------------------------------------------------------------------")
                print(" ")
                accuracies.append(results[1])
                thisEpochAccs.append(results[1])
            print(
                "This epoch (" + str(i) + ") had average validation accuracy " + str(sum(thisEpochAccs) / len(thisEpochAccs)))
            print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            savedAccFile = open(modelName + "-accs.p", 'wb')
            pickle.dump(accuraciesAll, savedAccFile)
        bestAccs = []
        for i in range(0, k):
            bestAccs.append(max(accuraciesAll[i]))
        bestAccAvg = sum(bestAccs) / len(bestAccs)
        print("")
        print("")
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        print("Your superscored validation accuracy for this model is " +  str(bestAccAvg) + ".")
        print("Here are the best fold accuracies you had:")
        for i in range(0,k):
            print("Fold " + str(i) + ": " + str(bestAccs[i]))
        epochAccs = []
        for i in range(0,len(accuraciesAll[0])):
            currSum = 0
            for j in range(0,len(accuraciesAll)):
                currSum += accuraciesAll[j][i]
            epochAccs.append(currSum/len(accuraciesAll))
        print("Your best epoch validation accuracy is " + str(max(epochAccs)) + ".")
        print("These were your folds:")
        currListNum = 0
        for filenameList in filenameLists:
            print(">> List " + str(currListNum))
            for filename in filenameList:
                print(filename)
            currListNum += 1
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    def trainWithKFoldEval(self, model, k, modelName, epochs, outputExtraDim=True, saveBestOnly=True):
        generator = self.generator
        filenameLists = generator.generateKFoldLists(k)
        savedListsFile = open(modelName + "-folds.p", 'wb')
        pickle.dump(filenameLists, savedListsFile)
        savedModulesListFile = open(modelName + "-moduleslist.p", 'wb')
        pickle.dump(generator.modulesList, savedModulesListFile)
        modelList = []
        for j in range(0, k):
            clonedModel = model.clone(modelName + "-fold-" + str(j))
            clonedModel.compile()
            modelList.append(clonedModel)
        self.trainKFolds(modelName, modelList, filenameLists, epochs, saveBestOnly)
    def trainWithSavedKFoldEval(self, modelName, epochs, saveBestOnly, generatorLabeler, outputExtraDim=True):
        accs = pickle.load(open(modelName + "-accs.p", "rb"))
        filenameLists = pickle.load(open(modelName + "-folds.p", "rb"))
        modulesList = pickle.load(open(modelName + "-moduleslist.p", "rb"))
        modelList = []
        for j in range(0, len(filenameLists)):
            openedModel = MModel(modelName)
            openedModel.load(modelName + "-fold-" + str(j) + ".h5")
            openedModel.compile()
            modelList.append(openedModel)
        self.trainKFolds(modelName, modelList, filenameLists, epochs, savedAccuracies=accs, saveBestOnly=saveBestOnly)
