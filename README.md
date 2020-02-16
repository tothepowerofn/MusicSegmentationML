# MusicSegmentationML

![banner](./banner.png)

MusicSegmentationML is a supervised sequence-to-sequence machine learning project that labels parts of songs (audio WAV files) with their corresponding song segment type (such as verse, chorus, bridge, etc.).

This project provides classes for extracting features (feat_extract.py), generating data to feed into the models ((feat_extract.py), and training various models (ml.py).

The primary features extracted from the audio WAV files (at the moment) are MFCC features. Right now, the best model, Faded2DConvModel (see Model_Thoughts for params), achieves ~64% 4-fold validation accuracy on a dataset of 27 songs.
