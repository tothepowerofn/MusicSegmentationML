from feat_extract import saveMFCCForWavs, AnnotatedSongLabeler, generateLabeledFeatures, Pooling1DFeature, saveFeatures
from feat_extract import generateTrainingDataForAudio
from feat_extract import generateTrainingDataForAudios
from ml import stupidSimpleRNNModel
from ml import stupidSimpleRNNModelTimeDisLess
from ml import dumbSimpleRNNModel
from ml import trainWithModelSingleSong
from ml import trainModel
from ml import trainModelWithGenerator
from ml import modelV1
from feat_extract import getFeatsAndClassificationsFromFile
from feat_extract import trainingGeneratorFromFolder
from keras.models import load_model
from numpy import zeros, newaxis
from feat_extract import MFCCFeature


numSegmentTypes = 6
load = False
modelname = "pooled_model_v1"

if load:
    model = load_model(modelname)
else:
    model = modelV1(inputDimension=200, numPerRecurrentLayer=100, numRecurrentLayers=2,
                                            outputDimension=numSegmentTypes, kernelSize=10, stride=1)
    model.summary()

trainModelWithGenerator(model, trainingGeneratorFromFolder, "features", modelname, 10)