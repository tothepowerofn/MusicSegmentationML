import keras as k
from keras.callbacks import ModelCheckpoint
from keras.models import Model
from keras.layers import (BatchNormalization, Conv1D, Dense, Input,
    TimeDistributed, Activation, Bidirectional, SimpleRNN, GRU, LSTM)
from keras.utils import to_categorical
from numpy import zeros, newaxis
from keras.layers import Lambda
from keras.backend import slice

#https://github.com/JackBurdick/ASR_DL/blob/master/sample_models.py roughly used as a starting point
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

def adjustedConvModel(inputDimension, numPerRecurrentLayer, convWindowSize, numRecurrentLayers, outputDimension, numConvFilters=250, perkernelSize=11, stride=1):
    # Input Layer
    inputLayer = Input(shape=(None, inputDimension))
    convSlices = []
    convSliceInput = inputLayer
    for i in range(0, inputDimension/convWindowSize):
        slicedInput = Lambda(lambda x: x[:,:,0:i*convWindowSize], output_shape=(None,None,convWindowSize))(convSliceInput)
    #Convolutional Slices

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

def trainModelWithGenerator(model, generator, modelSaveName, epochs):
    numberOfFeatFiles = generator.getNumberOfFeatFiles()
    print(numberOfFeatFiles)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
    checkpoint = ModelCheckpoint(modelSaveName, monitor='categorical_accuracy', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    history = model.fit(generator.generate(), epochs=epochs, steps_per_epoch = numberOfFeatFiles, callbacks=callbacks_list)
    #model.save(modelSaveName)