from keras.engine.saving import load_model

from feat_extract import saveTrainingData, MFCCFeature, AnnotatedSongLabeler, DelayFeature
from ml import ModelEvaluator, poolingConvModelWithDropout, Model, PoolingConvModelWithDropout,\
    PoolingModelWithDropout, FadingPoolingModelWithDropout

extract = False
load = False
poolSize = 1000
numSegmentTypes = 6
modelName = "PoolingModelWithDropout80"


if extract:
    mfccFeature = MFCCFeature("mfcc_input_1", "wavs", hop_length=4096, n_fft=8192, n_mfcc=10)
    delayFeature = DelayFeature("delay_input_1", mfccFeature, 1)
    featureStringList = []
    featureList = []
    featureList.append(mfccFeature)
    featureStringList.append("mfcc_input_1")
    for i in range(1,poolSize):
        delayFeature = DelayFeature("delay_input_" + str(i), mfccFeature, i)
        featureStringList.append("delay_input_" + str(i))
        featureList.append(delayFeature)

    #pool = Pooling1DFeature("pooling_input_1", mfccFeature, 10)
    annot = AnnotatedSongLabeler("annotations", sample_rate=22050, hop_length=4096)
    #featuresList = [delayFeature, mfccFeature, pool]

    saveTrainingData("features", featureList, annot)



model = FadingPoolingModelWithDropout(modelName)
if not load:
    model.build(dropoutRate=0.35, numInputs=poolSize, perInputDimension=10, numPerRecurrentLayer=100, numRecurrentLayers=2,
                numDenseLayerUnits=100, outputDimension=numSegmentTypes, fadingMaxUnits=5)
    model.summary()

featureStringList = []
featureStringList.append("mfcc_input_1")
for i in range(1,poolSize):
    featureStringList.append("delay_input_" + str(i))

evaluator = ModelEvaluator("features", "labels", featureStringList, coalesceInput=False)

if load:
    evaluator.trainWithSavedKFoldEval(modelName, 20, saveBestOnly=False)
else:
    evaluator.trainWithKFoldEval(model=model, k=4, modelName=modelName, epochs=30, saveBestOnly=False)
