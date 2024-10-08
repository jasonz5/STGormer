import os
import time
import torch 
import numpy as np 

class StandardScaler:
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        if type(data) == torch.Tensor and type(self.mean) == np.ndarray:
            self.std = torch.from_numpy(self.std).to(data.device).type(data.dtype)
            self.mean = torch.from_numpy(self.mean).to(data.device).type(data.dtype)
        return (data * self.std) + self.mean

class MinMax01Scaler:
    """
    Standard the input
    """

    def __init__(self, min, max):
        self.min = min
        self.max = max

    def transform(self, data):
        return (data - self.min) / (self.max - self.min)

    def inverse_transform(self, data):
        if type(data) == torch.Tensor and type(self.min) == np.ndarray:
            self.min = torch.from_numpy(self.min).to(data.device).type(data.dtype)
            self.max = torch.from_numpy(self.max).to(data.device).type(data.dtype)
        return (data * (self.max - self.min) + self.min)

class MinMax11Scaler:
    """
    Standard the input
    """

    def __init__(self, min, max):
        self.min = min
        self.max = max

    def transform(self, data):
        return ((data - self.min) / (self.max - self.min)) * 2. - 1.

    def inverse_transform(self, data):
        if type(data) == torch.Tensor and type(self.min) == np.ndarray:
            self.min = torch.from_numpy(self.min).to(data.device).type(data.dtype)
            self.max = torch.from_numpy(self.max).to(data.device).type(data.dtype)
        return ((data + 1.) / 2.) * (self.max - self.min) + self.min

def STDataloader(X, Y, batch_size, shuffle=True, drop_last=True):
    cuda = True if torch.cuda.is_available() else False
    TensorFloat = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    X, Y = TensorFloat(X), TensorFloat(Y)
    data = torch.utils.data.TensorDataset(X, Y)
    dataloader = torch.utils.data.DataLoader(
        data, 
        batch_size=batch_size,
        shuffle=shuffle, 
        drop_last=drop_last,
    )
    return dataloader

def normalize_data(data, d_output, scalar_type='Standard'):
    scalar = None
    if scalar_type == 'MinMax01':
        scalar = MinMax01Scaler(min=data[..., :d_output].min(), max=data[..., :d_output].max())
    elif scalar_type == 'MinMax11':
        scalar = MinMax11Scaler(min=data[..., :d_output].min(), max=data[..., :d_output].max())
    elif scalar_type == 'Standard':
        scalar = StandardScaler(mean=data[..., :d_output].mean(), std=data[..., :d_output].std())
    else:
        raise ValueError('scalar_type is not supported in data_normalization.')
    # print('{} scalar is used!!!'.format(scalar_type))
    # time.sleep(3)
    return scalar

def get_dataloader_from_train_val_test(data_dir, dataset, d_input, d_output, batch_size, test_batch_size, scalar_type='Standard'):
    data = {}
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(data_dir, dataset, category + '.npz'))
        data['x_' + category] = cat_data['x']
        data['y_' + category] = cat_data['y']
    scaler = normalize_data(np.concatenate([data['x_train'], data['x_val']], axis=0), d_output, scalar_type)
    
    # Data format
    for category in ['train', 'val', 'test']:
        data['x_' + category][..., :d_output] = scaler.transform(data['x_' + category][..., :d_output])
        data['y_' + category][..., :d_output] = scaler.transform(data['y_' + category][..., :d_output])
    # Construct dataloader
    dataloader = {}
    dataloader['train'] = STDataloader(
        data['x_train'][..., :d_input], 
        data['y_train'][..., :d_output], 
        batch_size, 
        shuffle=True
    )
    dataloader['val'] = STDataloader(
        data['x_val'][..., :d_input], 
        data['y_val'][..., :d_output], 
        test_batch_size, 
        shuffle=False
    )
    dataloader['test'] = STDataloader(
        data['x_test'][..., :d_input], 
        data['y_test'][..., :d_output], 
        test_batch_size, 
        shuffle=False, 
        drop_last=False
    )
    dataloader['scaler'] = scaler
    return dataloader

def vrange(starts, stops):
    stops = np.asarray(stops)
    l = stops - starts  # Lengths of each range. Should be equal, e.g. [12, 12, 12, ...]
    assert l.min() == l.max(), "Lengths of each range should be equal."
    indices = np.repeat(stops - l.cumsum(), l) + np.arange(l.sum())
    return indices.reshape(-1, l[0])

def get_dataloader_from_index_data(
    data_dir, dataset, d_input, d_output, batch_size, test_batch_size, scalar_type='Standard'
):
    data_all = np.load(os.path.join(data_dir, dataset, "data.npz"))["data"].astype(np.float32)
    index_all = np.load(os.path.join(data_dir, dataset, "index.npz")) # (num_samples, 3)

    data = {}
    index = {}
    for category in ['train', 'val', 'test']:
        index['x_' + category] = vrange(index_all[category][:, 0], index_all[category][:, 1])
        index['y_' + category] = vrange(index_all[category][:, 1], index_all[category][:, 2])
    for category in ['train', 'val', 'test']:
        data['x_' + category] = data_all(index['x_' + category])
        data['y_' + category] = data_all(index['y_' + category])
    scaler = normalize_data(data['x_train'], d_output, scalar_type)

    # Data format
    for category in ['train', 'val', 'test']:
        data['x_' + category][..., :d_output] = scaler.transform(data['x_' + category][..., :d_output])
        data['y_' + category][..., :d_output] = scaler.transform(data['y_' + category][..., :d_output])

    # Construct dataloader
    dataloader = {}
    dataloader['train'] = STDataloader(
        data['x_train'][..., :d_input], 
        data['y_train'][..., :d_output], 
        batch_size, 
        shuffle=True
    )
    dataloader['val'] = STDataloader(
        data['x_val'][..., :d_input], 
        data['y_val'][..., :d_output], 
        test_batch_size, 
        shuffle=False
    )
    dataloader['test'] = STDataloader(
        data['x_test'][..., :d_input], 
        data['y_test'][..., :d_output], 
        test_batch_size, 
        shuffle=False, 
        drop_last=False
    )
    dataloader['scaler'] = scaler
    return dataloader

if __name__ == '__main__':
    loader = get_dataloader_from_train_val_test('../data/', 'NYCBike1', batch_size=64, test_batch_size=64)
    for key in loader.keys():
        print(key)
    # import ipdb; ipdb.set_trace()