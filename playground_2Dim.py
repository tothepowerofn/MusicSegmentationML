from MusicSegmentationML.feat_extract import *
from MusicSegmentationML.ml import *

extract = False
train = True
load = False
hop_length = 4096
n_fft = 8192
n_mfcc = 10
poolSize = 10
k = 12
numSegmentTypes = 6
maxSegmentListLength = 15
epochs = 15
modelName = "Faded2DConvModel"

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
    model = Faded2DConvModel(modelName)
    if not load:
        model.build(numClasses=numSegmentTypes, inputShapeList=[(26,10),(104,10),(208,10),(208,10),(208,10),(208,10),(208,10)],
                    inputConvFilterNumList=[256, 128, 128, 64, 64, 64, 64],
                    inputConvKernelSizeList=[8,8,8,8,8,8,8], convMaxPoolSizeList=[None, 2, 4, 8, 16, 16, 16],
                    convDenseSizeList=[100,50,35,25,20,20,20],
                    postConvDropout=None, preRNNDropout=0.5, numRNNLayers=2, rNNUnitsList=[100,100], rnnDropoutList=[0.5,0.5],
                    postRNNDropout=0.5)
        model.summary()

    modulesList =[]
    chunkedMFCCDataGeneratorModuleNear = ChunkedMFCCDataGeneratorModule("features", "mfcc_input_1", 26, 13)
    chunkedMFCCDataGeneratorModuleMid = ChunkedMFCCDataGeneratorModule("features", "mfcc_input_1", 104, 13)
    chunkedMFCCDataGeneratorModuleFar = ChunkedMFCCDataGeneratorModule("features", "mfcc_input_1", 208, 13)

    delayed2DDataGeneratorModuleMid = Delayed2DDataGeneratorModule(chunkedMFCCDataGeneratorModuleMid, 2)
    delayed2DDataGeneratorModuleFar = Delayed2DDataGeneratorModule(chunkedMFCCDataGeneratorModuleFar, 10)
    delayed2DDataGeneratorModuleVeryFar = Delayed2DDataGeneratorModule(chunkedMFCCDataGeneratorModuleFar, 26)
    delayed2DDataGeneratorModuleVeryVeryFar = Delayed2DDataGeneratorModule(chunkedMFCCDataGeneratorModuleFar, 42)
    delayed2DDataGeneratorModuleSuperFar = Delayed2DDataGeneratorModule(chunkedMFCCDataGeneratorModuleFar, 58)
    delayed2DDataGeneratorModuleSuperDuperFar = Delayed2DDataGeneratorModule(chunkedMFCCDataGeneratorModuleFar, 74)

    modulesList.append(chunkedMFCCDataGeneratorModuleNear)
    modulesList.append(delayed2DDataGeneratorModuleMid)
    modulesList.append(delayed2DDataGeneratorModuleFar)
    modulesList.append(delayed2DDataGeneratorModuleVeryFar)
    modulesList.append(delayed2DDataGeneratorModuleVeryVeryFar)
    modulesList.append(delayed2DDataGeneratorModuleSuperFar)
    modulesList.append(delayed2DDataGeneratorModuleSuperDuperFar)

    generatorLabeler = GeneratorLabeler2D("annotations", 22050, hop_length, 13)
    modularDataGenerator = ModularDataGenerator("features", "labels", modulesList, generatorLabeler, samplesShapeIndex=0, outputExtraDim=True)
    evaluator = ModelEvaluator(modularDataGenerator)
    if load:
        evaluator.trainWithSavedKFoldEval(modelName, epochs, generatorLabeler=generatorLabeler, saveBestOnly=False, outputExtraDim=True)
    else:
        evaluator.trainWithKFoldEval(model=model, k=k, modelName=modelName, epochs=epochs, saveBestOnly=False, outputExtraDim=True)
