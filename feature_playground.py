from keras.engine.saving import load_model

from feat_extract import saveTrainingData, MFCCFeature, Pooling1DFeature, AnnotatedSongLabeler, TrainingGenerator
from ml import modelV4, trainModelWithGenerator

mfccFeature = MFCCFeature("mfcc_input_1", "wavs", hop_length=4096, n_fft=8192)
pool = Pooling1DFeature("pooling_input_1", mfccFeature, 10)
annot = AnnotatedSongLabeler("annotations", sample_rate=22050, hop_length=4096)

featuresList = [mfccFeature, pool]

saveTrainingData("features", featuresList, annot)


numSegmentTypes = 6
load = False
modelname = "temp.h5"

if load:
    model = load_model(modelname)
else:
    model = modelV4(inputDimension=200, numPerRecurrentLayer=75, numRecurrentLayers=2,
                                            outputDimension=numSegmentTypes, kernelSize=10, stride=1)
    model.summary()

trainingGenerator = TrainingGenerator("features", "labels", ["pooling_input_1"])
trainModelWithGenerator(model, trainingGenerator, modelname, 30)