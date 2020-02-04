import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display

from feat_extract import saveMFCCForWavs, AnnotatedSongLabeler, generateFeatures
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

mfccFeature = MFCCFeature("wavs", hop_length=2048, n_fft=8192)
mfccFeature2 = MFCCFeature("wavs", hop_length=2048, n_fft=8192)
annot = AnnotatedSongLabeler("annotations", sample_rate=22050, hop_length=2048)

generateFeatures(annot, mfccFeature, mfccFeature2)

