import numpy as np

data = np.load('./BJTaxi/train.npz')
for file in data.files:
    print(file, data[file].shape)
# import ipdb; ipdb.set_trace()
