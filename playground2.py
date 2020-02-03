import numpy as np
from keras.models import load_model
from feat_extract import getFeatsAndClassificationsFromFile
model = load_model('stupidSimpleRNN_1')
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
feats, classifications = getFeatsAndClassificationsFromFile("features/Grateful_1-18-2020-feats.csv")
predictionss = model.predict(feats)
for predictions in predictionss:
    np.savetxt("Grateful_1-18-2020-predicted.csv", predictions, delimiter=",")