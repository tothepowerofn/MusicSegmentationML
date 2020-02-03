import numpy as np
from keras.models import load_model
from feat_extract import getFeatsAndClassificationsFromFile
model = load_model('dumbSimpleRNN_Big')
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
feats, classifications = getFeatsAndClassificationsFromFile("features/King-of-my-Heart_1-18-2020-feats.csv")
predictionss = model.predict(feats)
for predictions in predictionss:
    print(predictions.shape)
    print(np.argmax(predictions, axis=1)[:,None].shape)
    output = np.append(predictions, np.zeros(predictions.shape[0])[:,None], axis=1)
    output = np.append(output, np.arange(1, output.shape[0] + 1)[:, None] * 512 / 20500, axis=1)
    output = np.append(output, np.argmax(predictions, axis=1)[:,None], axis=1)

    np.savetxt("King-of-my-Heart_1-18-2020-predicted.csv", output, delimiter=",")