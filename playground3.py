import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display

from feat_extract import saveMFCCForWavs, AnnotatedSongLabeler, generateLabeledFeatures, Pooling1DFeature, saveFeatures
from feat_extract import generateTrainingDataForAudio
from feat_extract import generateTrainingDataForAudios
from ml import stupidSimpleRNNModel
from ml import stupidSimpleRNNModelTimeDisLess
from ml import dumbSimpleRNNModel
from ml import trainWithModelSingleSong
from ml import trainModel
from ml import trainModelWithGenerator
from feat_extract import getFeatsAndClassificationsFromFile
from feat_extract import trainingGeneratorFromFolder
from keras.models import load_model
from numpy import zeros, newaxis
from feat_extract import MFCCFeature


mfccFeature = MFCCFeature("wavs", hop_length=4096, n_fft=8192)
pool = Pooling1DFeature(mfccFeature, 10)
annot = AnnotatedSongLabeler("annotations", sample_rate=22050, hop_length=4096)
labeledFeatures = generateLabeledFeatures(annot, pool)
saveFeatures(labeledFeatures, "features")


