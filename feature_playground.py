from keras.engine.saving import load_model
from MusicSegmentationML.feat_extract import *
from MusicSegmentationML.ml import *

extract = False
train = True
load = False
hop_length = 4096
n_fft = 8192
n_mfcc = 10
poolSize = 500
k = 4
numSegmentTypes = 6
maxSegmentListLength = 15
epochs = 10
modelName = "PoolingModelWithDropoutAndSegOrder500"

if extract:
    songMetadataIndexerModule = SongMetadataIndexerModule()
    mfccDimIndexerModule = MFCCDimensionsIndexerModule(hop_length, n_mfcc)
    indexerModulesList = [songMetadataIndexerModule, mfccDimIndexerModule]
    indexer = Indexer("wavs", indexerModulesList)
    mfccFeature = MFCCFeature("mfcc_input_1", "wavs", hop_length=hop_length, n_fft=n_fft, n_mfcc=n_mfcc)
    segmentOrderFeature = SegmentOrderFeature("segment_order_input_1", "annotations", numSegmentTypes, maxSegmentListLength)
    featureList = []
    featureList.append(mfccFeature)
    featureList.append(segmentOrderFeature)
    annot = AnnotatedSongLabeler("annotations", sample_rate=22050, hop_length=hop_length)
    saveTrainingData("features", featureList, indexer, annot)

if train:
    model = FadingPoolingModelWithDropoutAndSegOrder(modelName)
    if not load:
        model.build(dropoutRate=0.45, numInputs=poolSize, perInputDimension=n_mfcc, numPerRecurrentLayer=100, numRecurrentLayers=2,
                    numDenseLayerUnits=100, outputDimension=numSegmentTypes, fadingMaxUnits=5, segOrderDimension=numSegmentTypes*maxSegmentListLength)
        model.summary()

    modulesList =[]
    pooledDataGeneratorModule = PooledDataGeneratorModule(poolSize, "features", "mfcc_input_1")
    segmentOrderDataGeneratorModule = SegmentOrderDataGeneratorModule("features", "segment_order_input_1")
    modulesList.append(pooledDataGeneratorModule)
    modulesList.append(segmentOrderDataGeneratorModule)
    evaluator = ModelEvaluator("features", "labels")

    if load:
        evaluator.trainWithSavedKFoldEval(modelName, epochs, saveBestOnly=False)
    else:
        evaluator.trainWithKFoldEval(model=model, k=k, modelName=modelName, modulesList=modulesList, epochs=epochs, saveBestOnly=False)
