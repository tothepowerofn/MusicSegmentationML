from keras.engine.saving import load_model
from MusicSegmentationML.feat_extract import *
from MusicSegmentationML.ml import *

extract = False
train = True
load = False
hop_length = 4096
n_fft = 8192
n_mfcc = 10
poolSize = 10
k = 4
numSegmentTypes = 6
maxSegmentListLength = 15
epochs = 10
modelName = "Test2D"

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
    model = Simple2DTest(modelName)
    if not load:
        model.build(52, n_mfcc)
        model.summary()

    modulesList =[]
    pooledDataGeneratorModule = PooledDataGeneratorModule(poolSize, "features", "mfcc_input_1", outputExtraDim=True)
    segmentOrderDataGeneratorModule = SegmentOrderDataGeneratorModule("features", "segment_order_input_1", outputExtraDim=False)
    chunkedMFCCDataGeneratorModule = ChunkedMFCCDataGeneratorModule("features", "mfcc_input_1", 52)
    modulesList.append(chunkedMFCCDataGeneratorModule)
    #modulesList.append(segmentOrderDataGeneratorModule)
    generatorLabeler = GeneratorLabeler2D("annotations", 22050, hop_length, 52)
    modularDataGenerator = ModularDataGenerator("features", "labels", modulesList, generatorLabeler, samplesShapeIndex=1, outputExtraDim=True)
    evaluator = ModelEvaluator(modularDataGenerator)
    if load:
        evaluator.trainWithSavedKFoldEval(modelName, epochs, saveBestOnly=False, outputExtraDim=True)
    else:
        evaluator.trainWithKFoldEval(model=model, k=k, modelName=modelName, epochs=epochs, saveBestOnly=False, outputExtraDim=True)
