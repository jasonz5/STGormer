import numpy as np
import torch

def mae_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.mean(torch.abs(true-pred))

def rmse_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.sqrt(torch.mean(torch.square(true - pred)))

def mape_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.mean(torch.abs(torch.div((true - pred), true)))

def mae_np(pred, true, mask_value=None):
    if mask_value != None:
        mask = np.where(true > (mask_value), True, False)
        true = true[mask]
        pred = pred[mask]
    return np.mean(np.absolute(pred-true))

def rmse_np(pred, true, mask_value=None):
    if mask_value != None:
        mask = np.where(true > (mask_value), True, False)
        pred = pred[mask]
        true = true[mask]
    return np.sqrt(np.mean(np.square(true - pred)))

def mape_np(pred, true, mask_value=None):
    if mask_value != None:
        mask = np.where(true > (mask_value), True, False)
        true = true[mask]
        pred = pred[mask]
    return np.mean(np.absolute(np.divide((true - pred), true)))

def test_metrics(pred, true, mask=5.0):
    assert type(pred) == type(true)
    if type(pred) == np.ndarray:
        mae  = mae_np(pred, true, mask)
        rmse = rmse_np(pred, true, mask)
        mape = mape_np(pred, true, mask)
    elif type(pred) == torch.Tensor:
        mae  = mae_torch(pred, true, mask).item()
        rmse = rmse_torch(pred, true, mask).item()
        mape = mape_torch(pred, true, mask).item()
    else:
        raise TypeError
    return mae, rmse, mape


