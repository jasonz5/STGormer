import numpy as np

data = np.load('./NYCBike1/train.npz')
for file in data.files:
    print(file, data[file].shape)
import ipdb; ipdb.set_trace()

''' TaxiBJ dataset
x (3023, 19, 128, 2)
y (3023, 1, 128, 2)
x_offsets (19, 1)
y_offsets (1, 1)
'''