from keras.engine.saving import load_model

from feat_extract import saveTrainingData, MFCCFeature, Pooling1DFeature, AnnotatedSongLabeler, DataGenerator, \
    DelayFeature
from ml import modelV4, trainModelWithGenerator, poolingConvModel, ModelEvaluator, poolingRecConvModel, \
    poolingConvModelWithDropout

extract = False
if extract:
    mfccFeature = MFCCFeature("mfcc_input_1", "wavs", hop_length=4096, n_fft=8192, n_mfcc=10)
    delayFeature = DelayFeature("delay_input_1", mfccFeature, 1)
    delayFeatureStringList = []
    delayFeatureList = []
    for i in range(1,401):
        delayFeature = DelayFeature("delay_input_" + str(i), mfccFeature, i)
        delayFeatureStringList.append("delay_input_" + str(i))
        delayFeatureList.append(delayFeature)

    #pool = Pooling1DFeature("pooling_input_1", mfccFeature, 10)
    annot = AnnotatedSongLabeler("annotations", sample_rate=22050, hop_length=4096)
    #featuresList = [delayFeature, mfccFeature, pool]

    saveTrainingData("features", delayFeatureList, annot)


numSegmentTypes = 6
load = False
modelname = "poolingConvModelWithDropout40"

if load:
    model = load_model(modelname)
else:
    model = poolingConvModelWithDropout(dropoutRate=0.5, numInputs=400, perInputDimension=10, numPerRecurrentLayer=60, numRecurrentLayers=2, numDenseLayerUnits=40, outputDimension=numSegmentTypes, numConvFiltersPerConv=250, kernelSizePerConv=5, stride=1)
    model.summary()

delayFeatureStringList = []
for i in range(1,401):
    delayFeatureStringList.append("delay_input_" + str(i))

#trainingGenerator = DataGenerator("features", "labels", delayFeatureStringList, coalesceInput=False)
#trainModelWithGenerator(model, trainingGenerator, modelname, 30)

evaluator = ModelEvaluator("features", "labels", delayFeatureStringList, coalesceInput=False)
evaluator.trainWithKFoldEval(model=model, k=4, modelName=modelname, epochs=30, saveBestOnly=False)
#evaluator.trainWithSavedKFoldEval(modelname, 20, saveBestOnly=False)