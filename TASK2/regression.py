import pandas
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

def map_to_numerical(dataset_with_categories, col_index):
	"""Maps categorical data to numerical form using pandas factorize() function
		
	Params:
		dataset_with_catgories: the dataset with categorical values
		col_index: which column to map to numerical data

	Returns:
		the mapped dataset_with_catgories with the specified column changed to unique numerical values
	"""
	labels, uniques = pandas.factorize(dataset_with_categories[:, col_index]) 
	dataset_with_categories[: , col_index] = labels
	if Verbose:
		# print "labels for column %d " % col_index , labels # Uncomment this line to see the mapped column
		print "uniques for column %d: " % col_index, uniques
	return dataset_with_categories


def baseline_model():
	"""Define, create and compile base NN model with 30 or 32 inputs"""
	model = Sequential()
	model.add(Dense(10, input_dim=NUM_ATTRIBUTES, init='normal', activation='relu'))
	model.add(Dense(10, init='normal', activation='relu'))
	model.add(Dense(1, init='normal'))
	model.compile(loss='mean_squared_error', optimizer='adam') # compile model
	return model


Verbose = True
seed = 7 # Fix random seed for reproducibility
numpy.random.seed(seed)

NUM_ATTRIBUTES = 32
# Load dataset
dataframe = pandas.read_csv("Datasets/Students/students.csv", header=None)
dataset = dataframe.values

# Split dataset into input (X) and output (Y) variables
X = dataset[1:, 0:NUM_ATTRIBUTES] # Read all rows (except first), col 0 up to 29/31 inclusive (30/32 data attributes)
Y = dataset[1:, NUM_ATTRIBUTES] # Read all rows (except first) of col 32 (G3)

# Map two classes in school, address, famsize, Pstatus to 0 or 1
X = map_to_numerical(X, 0) # school is col 0. Map to 0 or 1
X = map_to_numerical(X, 1) # sex is in col 1. Map to 0 or 1
X = map_to_numerical(X, 3) # address is in col 3. Map to 0 or 1
X = map_to_numerical(X, 4) # famsize is in col 4. Map to 0 or 1
X = map_to_numerical(X, 5) # Pstatus is in col 5. Map to 0 or 1

# Map multiple categories in school, address, famsize, Pstatus to 0 -> 4
X = map_to_numerical(X, 8) # Mjob is in col 8. Map to 0 -> 4
X = map_to_numerical(X, 9) # Fjob is in col 9. Map to 0 -> 4
X = map_to_numerical(X, 10) # reason is in col 10. Map to 0 -> 3
X = map_to_numerical(X, 11) # guardian is in col 11. Map to 0 -> 2

# Map yes or no in schoolsup, famsup, paid, activities, nursery, higher, internet, romantic to 0 or 1
X = map_to_numerical(X, 15) # schoolsup is in col 15. Map to 0 or 1
X = map_to_numerical(X, 16) # famsup is in col 16. Map to 0 or 1
X = map_to_numerical(X, 17) # paid is in col 17. Map to 0 or 1
X = map_to_numerical(X, 18) # activities is in col 18. Map to 0 or 1
X = map_to_numerical(X, 19) # nursery is in col 19. Map to 0 or 1
X = map_to_numerical(X, 20) # higher is in col 20. Map to 0 or 1
X = map_to_numerical(X, 21) # internet is in col 21. Map to 0 or 1
X = map_to_numerical(X, 22) # romantic is in col 22. Map to 0 or 1

X = X.astype(float)

# Evaluate model with standardized dataset
estimator = KerasRegressor(build_fn=baseline_model, nb_epoch=100, batch_size=5, verbose=1)

kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(estimator, X, Y, cv=kfold)
print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))