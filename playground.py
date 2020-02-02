import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display

from feat_extract import saveMFCCForWavs
from feat_extract import generateTrainingDataForAudio
from feat_extract import generateTrainingDataForAudios
from ml import stupidSimpleRNNModel
from ml import trainWithModelSingleSong
from ml import trainModel
from ml import trainModelWithGenerator
from feat_extract import getFeatsAndClassificationsFromFile
from feat_extract import trainingGeneratorFromFolder
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

#saveMFCCForWavs("wavs", "mfccs")
#generateTrainingDataForAudio("wavs/test.wav", "annotations/test-annotation.csv", "features/test-feats.csv")
generateTrainingDataForAudios("wavs", "annotations", "features")
#trainWithModelSingleSong(1,1,1)

#getFeatsAndClassificationFromFolder("features")

feats, classifications = getFeatsAndClassificationsFromFile("features/test-feats.csv")
model = stupidSimpleRNNModel(inputDimension=20, numPerRecurrentLayer=150, numRecurrentLayers=2, outputDimension=4)
model.summary()
trainModelWithGenerator(model, trainingGeneratorFromFolder, "features",  10)
# trainModel(model, feats, classifications, 30)
# predictionss = model.predict(feats)
# for predictions in predictionss:
#     for prediction in predictions:
#         print(prediction)