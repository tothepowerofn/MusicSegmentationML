from keras.engine.saving import load_model

from feat_extract import saveTrainingData, MFCCFeature, AnnotatedSongLabeler, DelayFeature
from ml import ModelEvaluator, poolingConvModelWithDropout, Model, PoolingConvModelWithDropout,\
    PoolingModelWithDropout, FadingPoolingModelWithDropout

from feat_extract import PooledDataGeneratorModule

extract = False
train = True
load = True
poolSize = 10
k = 4
numSegmentTypes = 6
epochs = 1
modelName = "PoolingModelWithDropout10"

if extract:
    mfccFeature = MFCCFeature("mfcc_input_1", "wavs", hop_length=4096, n_fft=8192, n_mfcc=10)
    featureList = []
    featureList.append(mfccFeature)
    annot = AnnotatedSongLabeler("annotations", sample_rate=22050, hop_length=4096)
    saveTrainingData("features", featureList, annot)

if train:
    model = FadingPoolingModelWithDropout(modelName)
    if not load:
        model.build(dropoutRate=0.45, numInputs=poolSize, perInputDimension=10, numPerRecurrentLayer=100, numRecurrentLayers=2,
                    numDenseLayerUnits=100, outputDimension=numSegmentTypes, fadingMaxUnits=5)
        model.summary()

    modulesList =[]
    pooledDataGeneratorModule = PooledDataGeneratorModule(poolSize, "features", "mfcc_input_1")
    modulesList.append(pooledDataGeneratorModule)
    evaluator = ModelEvaluator("features", "labels")

    if load:
        evaluator.trainWithSavedKFoldEval(modelName, epochs, saveBestOnly=False)
    else:
        evaluator.trainWithKFoldEval(model=model, k=k, modelName=modelName, modulesList=modulesList, epochs=epochs, saveBestOnly=False)
