import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display

from feat_extract import saveMFCCForWavs
from feat_extract import generateTrainingDataForAudio
from feat_extract import generateTrainingDataForAudios
from ml import stupidSimpleRNNModel
from ml import stupidSimpleRNNModelTimeDisLess
from ml import dumbSimpleRNNModel
from ml import trainWithModelSingleSong
from ml import trainModel
from ml import trainModelWithGenerator
from feat_extract import getFeatsAndClassificationsFromFile
from feat_extract import trainingGeneratorFromFolder
from keras.models import load_model
from numpy import zeros, newaxis

def junk():
    audio_path = "test.wav"
    x , sr = librosa.load(audio_path)
    print(type(x), type(sr))
    print(sr)

    #Wavefrom
    plt.figure(figsize=(14, 5))
    librosa.display.waveplot(x, sr=sr)
    #plt.show()

    #Spectrogram
    X = librosa.stft(x)
    Xdb = librosa.amplitude_to_db(abs(X))
    plt.figure(figsize=(14, 5))
    librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
    #If to pring log of frequencies
    #librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar()
    #plt.show()

    mfccs = librosa.feature.mfcc(x, sr=sr)
    print(mfccs.shape)
    #Displaying  the MFCCs:
    librosa.display.specshow(mfccs, sr=sr, x_axis='time')
    #plt.show()

    #save the extracted features as csv
    np.savetxt("test.csv", mfccs.T, delimiter=",")


#generateTrainingDataForAudios("wavs", "annotations", "features", hop_length=2048, n_fft=8192)

numSegmentTypes = 6
load = True
modelname = "stupidSimpleRNNModelTimeDistLess_kern11_200_3"
#model = stupidSimpleRNNModel(inputDimension=20, numPerRecurrentLayer=150, numRecurrentLayers=2, outputDimension=numSegmentTypes)

if load:
    model = load_model(modelname)
else:
    model = stupidSimpleRNNModelTimeDisLess(inputDimension=20, numPerRecurrentLayer=200, numRecurrentLayers=3,
                                            outputDimension=numSegmentTypes)
    model.summary()

trainModelWithGenerator(model, trainingGeneratorFromFolder, "features", modelname, 10)
