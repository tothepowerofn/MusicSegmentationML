from keras.engine.saving import load_model

from feat_extract import saveTrainingData, MFCCFeature, Pooling1DFeature, AnnotatedSongLabeler, TrainingGenerator, \
    DelayFeature
from ml import modelV4, trainModelWithGenerator, poolingConvModel

extract = False
if extract:
    mfccFeature = MFCCFeature("mfcc_input_1", "wavs", hop_length=4096, n_fft=8192)
    delayFeature = DelayFeature("delay_input_1", mfccFeature, 1)
    delayFeatureStringList = []
    delayFeatureList = []
    for i in range(1,11):
        delayFeature = DelayFeature("delay_input_" + str(i), mfccFeature, i)
        delayFeatureStringList.append("delay_input_" + str(i))
        delayFeatureList.append(delayFeature)

    #pool = Pooling1DFeature("pooling_input_1", mfccFeature, 10)
    annot = AnnotatedSongLabeler("annotations", sample_rate=22050, hop_length=4096)

    #featuresList = [delayFeature, mfccFeature, pool]

    saveTrainingData("features", delayFeatureList, annot)


numSegmentTypes = 6
load = False
modelname = "pooledConvModel.h5"

if load:
    model = load_model(modelname)
else:
    model = poolingConvModel(numInputs=10, perInputDimension=20, numPerRecurrentLayer=75, numRecurrentLayers=2, numDenseLayerUnits=50, outputDimension=numSegmentTypes, numConvFiltersPerConv=250, kernelSizePerConv=10, stride=1)
    model.summary()

delayFeatureStringList = []
for i in range(1,11):
    delayFeatureStringList.append("delay_input_" + str(i))

trainingGenerator = TrainingGenerator("features", "labels", delayFeatureStringList, coalesceInput=False)
trainModelWithGenerator(model, trainingGenerator, modelname, 30)