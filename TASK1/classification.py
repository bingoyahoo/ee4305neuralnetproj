# Train model and make predictions
import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline


# Define baseline model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(4, input_dim=561, init='normal', activation='relu')) # 4 hidden nodes
	model.add(Dense(50, init='normal', activation='relu')) # 50 hidden nodes
	model.add(Dense(12, init='normal', activation='sigmoid')) # 12 outputs
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model


# Fix random seed for reproducibility of results
seed = 7
numpy.random.seed(seed)

# Load dataset
dataframe = pandas.read_csv("datasets/hapt/haptAttr.txt", sep="\t", header=None)
dataset = dataframe.values
NUM_ATTRIBUTES = 561
X = dataset[:, 0:NUM_ATTRIBUTES].astype(float) # Read all 561 attributes 
print X

dataframe2 = pandas.read_csv("datasets/hapt/haptLabel.txt", sep="\t", header=None)
dataset2 = dataframe2.values
Y = dataset2[:, 0]
print Y

# Encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
print "Encoded Y ", encoded_Y

# Convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)
print "Dummy ", dummy_y

estimator = KerasClassifier(build_fn=baseline_model, nb_epoch=150, batch_size=5, verbose=1)


# Split into train and test dataset randomly
TEST_SIZE = 0.30 # Change this for experimentation
X_train, X_test, Y_train, Y_test = train_test_split(X, dummy_y, test_size=TEST_SIZE, random_state=seed)

estimator.fit(X_train, Y_train)
predictions = estimator.predict(X_test)
print(predictions)
yooo = np_utils.to_categorical(predictions)

# Count how many in the test set were correctly predicted
count = 0
count = numpy.sum(numpy.all(numpy.equal(Y_test, yooo), axis=1))
print "Correctly predicted", count/float(TEST_SIZE*8000.0)
predictions_labels = [int(x) for x in encoder.inverse_transform(predictions)]

# with open("test.txt", "w") as f:
# 	for x in predictions_labels:
# 		f.write(str(x) + "\n")

# results = cross_val_score(estimator, X, dummy_y, cv=kfold)
# print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
