import numpy as np
import os

# Prepare label dictionaries to translate between strings in text files and ints in numpy.
label_to_index = { s: i for i, s in enumerate(["Iris-versicolor", "Iris-setosa", "Iris-virginica"]) }
index_to_label = { i: s for s, i in label_to_index.items() }
index_to_feature = ["Petal length", "Petal width", "Sepal length", "Sepal width"]

def _load_iris(filename, onehot):
	data_path = os.path.dirname(os.path.realpath(__file__)) + '/data/' + filename
	
	X, y = [], []
	with open(data_path, 'r') as f:
		for l in f:
			if len(l) == 0: continue

			# Example line:
			# 6.6,2.9,4.6,1.3,"Iris-versicolor"
			l = l.replace('"', '') # Remove "

			x_ = [float(s) for s in l.split(',')[:-1]]
			y_ = label_to_index[l.split(',')[-1].strip()]

			X.append(x_)
			y.append(y_)

	n = len(y)
	d = len(label_to_index)

	y = np.array(y)
	if onehot:
		# Make (n, 3) array with one-hot encodings of the data.
		y_onehot = np.zeros( (n, d) )
		y_onehot[np.arange(n), y] = 1

	return np.array(X), y_onehot

def load_iris(onehot=True):
	"""
		Loads full iris dataset
	"""
	return _load_iris('iris.txt', onehot)

def load_iris_PC(onehot=True):
	return _load_iris('iris-PC.txt', onehot)


if __name__ == "__main__": 
	# Example usage. Just run
	# (dm20) > python load_data.py 
	# to see this execution. 

	for n, fn in [('iris.txt', load_iris), ('iris-PC.txt', load_iris_PC)]:
		X, y = fn()
		print("%-15s shapes: " % n, X.shape, y.shape)
