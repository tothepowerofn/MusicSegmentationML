from feat_extract import TrainingGenerator
from keras.models import load_model

delayFeatureStringList = []
for i in range(1,101):
    delayFeatureStringList.append("delay_input_" + str(i))

model = load_model('pooledConvModel.h5')
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
trainingGenerator = TrainingGenerator("features_holdout", "labels", delayFeatureStringList, coalesceInput=False)
print(model.evaluate(trainingGenerator.generate(),steps=1))
