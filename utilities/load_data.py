import numpy as np
import os
import urllib.request
import gzip

__all__ = [
        'load_iris',
        'load_iris_PC',
        'load_t7',
        'load_synthetic_data'
        'label_to_index',
        'index_to_label',
        'index_to_feature'
    ]

# Prepare label dictionaries to translate between strings in text files and ints in numpy.
label_to_index = { s: i for i, s in enumerate(["Iris-versicolor", "Iris-setosa", "Iris-virginica"]) }
index_to_label = { i: s for s, i in label_to_index.items() }
index_to_feature = ["Petal length", "Petal width", "Sepal length", "Sepal width"]

def _load_data(filename, onehot, lab_to_idx=label_to_index):
    data_path = os.path.dirname(os.path.realpath(__file__)) + '/data/' + filename
    print(data_path)
    
    X, y = [], []
    with open(data_path, 'r') as f:
        for l in f:
            if len(l) == 0: continue

            # Example line:
            # 6.6,2.9,4.6,1.3,"Iris-versicolor"
            l = l.replace('"', '') # Remove "

            x_ = [float(s) for s in l.split(',')[:-1]]
            y_ = lab_to_idx[l.split(',')[-1].strip()]

            X.append(x_)
            y.append(y_)

    n = len(y)
    d = len(lab_to_idx)

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
    return _load_data('iris.txt', onehot)

def load_iris_PC(onehot=True):
    """
        Loads 2 principal components from iris dataset
    """
    return _load_data('iris-PC.txt', onehot)

def load_t7():
    """
        Loads dataset with non-convex clusters from [Zaki, p. 376].
    """
    return _load_data('t7-4k.txt', onehot=True, lab_to_idx={"0": 0})

def load_synthetic_data(index=0, dims=10):
    data_file = 'synth_multidim_%03i_%03i.arff' % (dims, index)
    pth = "%s/data/%s" % (os.path.dirname(os.path.abspath(__file__)), data_file)
    print(pth)

    with open(pth, 'r') as f:
        l = f.readline()
        # Skip header lines
        while not '@data' == l.strip(): 
            l = f.readline()

        data = []
        labels = []
        for l in f:
            splt = l.strip().split(',')
            d = [float(s) for s in splt[:-1]]
            l = int(splt[-1])
            data.append(d)
            labels.append(l)

    return np.array(data), np.array(labels)

def load_mnist():
    base_dir = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + '/../utilities/data/') + '/'
    X_train_file = 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'
    y_train_file = 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz'
    X_test_file  = 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz'
    y_test_file  = 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'

    names = ['X_train.gz', 'y_train.gz', 'X_test.gz', 'y_test.gz']
    files = [X_train_file, y_train_file, X_test_file, y_test_file]
    sizes = [60000,        60000,        10000,       10000]

    out = []
    for (url, fn, size) in zip(files, names, sizes):
        file_path = base_dir + fn

        if not os.path.exists(file_path):
            print("Downloading %s to %s" % (url, file_path))
            urllib.request.urlretrieve(url, file_path)

        is_img = 'X' in fn
        image_size = 28

        with gzip.open(file_path,'r') as f:
            f.read(16) if is_img else f.read(8)
            read_size = size * 28**2 if is_img else size

            buf = f.read(read_size)
            data = np.frombuffer(buf, dtype=np.uint8)
            if is_img: 
                data = data.astype(np.float32)
                data = data.reshape(size, image_size, image_size)
            out.append(data)
    
    return tuple(out)

if __name__ == "__main__": 
    # Example usage. Just run
    # (dm20) > python load_data.py 
    # to see this execution. 
    datasets = [
            ('iris.txt', load_iris), 
            ('iris-PC.txt', load_iris_PC),
            ('t7-4k.txt', load_t7),
            ('mnist', load_mnist),
        ]
    for n, fn in datasets:
        out = fn()
        print("%-15s shapes: " % n, [o.shape for o in out])

