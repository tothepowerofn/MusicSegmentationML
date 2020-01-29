import keras as k
from keras.models import Model
from keras.layers import (BatchNormalization, Conv1D, Dense, Input,
    TimeDistributed, Activation, Bidirectional, SimpleRNN, GRU, LSTM)
from keras.utils import to_categorical
from numpy import zeros, newaxis

#https://github.com/JackBurdick/ASR_DL/blob/master/sample_models.py roughly used as a starting point
def stupidSimpleRNNModel(inputDimension, numPerRecurrentLayer, numRecurrentLayers, outputDimension, numConvFilters=250, kernelSize=11):
    #Input Layer
    inputLayer = Input(shape=(None, inputDimension))
    #Convolutional Layer
    convLayer = Conv1D(filters=numConvFilters, kernel_size=kernelSize,
                       strides=1,
                       padding='same',
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

def trainWithModelSingleSong(model, features, classifications):
    x_train = features[newaxis,:,:]
    y_train = to_categorical(classifications)[newaxis, :,:,]
    print(x_train.shape)
    print(y_train.shape)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(x_train, y_train,
                        epochs=4)