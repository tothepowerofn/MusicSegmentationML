import numpy as np
from keras.models import load_model
from feat_extract import getFeatsAndClassificationsFromFile
model = load_model('pooled_model_v3')
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
feats, classifications = getFeatsAndClassificationsFromFile("hold_out/Hallelujah_For_the_Cross_01-25-2020-feats.csv")
print(model.metrics_names)
print(model.evaluate(feats, classifications))
predictionss = model.predict(feats)
for predictions in predictionss:
    print(predictions.shape)
    print(np.argmax(predictions, axis=1)[:,None].shape)
    output = np.append(predictions, np.zeros(predictions.shape[0])[:,None], axis=1)
    output = np.append(output, np.arange(1, output.shape[0] + 1)[:, None] * 4096 / 20500, axis=1)
    output = np.append(output, np.argmax(predictions, axis=1)[:,None], axis=1)
    classificationsIntegers = np.array([np.where(r==1)[0][0] for r in classifications[0]])[:, None]
    print(classificationsIntegers.shape)
    output = np.append(output, classificationsIntegers, axis=1)

    np.savetxt("Hallelujah_For_the_Cross_01-25-2020-classifications.csv", output, delimiter=",")