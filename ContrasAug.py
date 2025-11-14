import torch
import numpy as np


def my_transform(data, config):
    data1 = extend_amp(data, 0.98, 1.02)
    data11 = extend_amp(data, 0.6, 1.4)
    data2 = leftAndRight(data)
    data3 = randomNoise(data)
    data4 = overLap(data, config)
    data5 = baseFloat(data)
    data_arrays = [data, data1, data2, data3, data4, data5, data11]
    temp = torch.from_numpy(np.array(data_arrays))
    return temp  


def overLap(data, config):
    if config.feature_dim is not None:
        transform = config.feature_dim // 2
    else:
        raise ValueError("config.feature_dim is not set")
    ret = data.copy()
    scale = np.random.uniform(0.01, 0.03, 1)  
    temp = data.copy()
    temp = leftAndRight(temp, transform)
    ret[:] += temp * scale
    return ret


def baseFloat(data):
    temp = data.copy()
    t = np.random.uniform(-0.03, 0.03)
    temp += t
    return temp


def setZero(data):
    temp = data.copy()
    if np.random.random() < 0.1:
        temp[:] = 0
    temp += np.random.normal(0, 0.01, temp.shape)

    return temp


def randomNoise(data):
    temp = data.copy()
    temp *= np.random.uniform(0.98, 1.02, temp.shape)
    # temp += np.random.normal(0, 0.08, temp.shape)

    return temp


def leftAndRight(data, transform=3):
    left = np.random.randint(0, 2)
    temp = data.copy()
    if transform == 3:
        movement = np.random.randint(0, transform) + 1
        if left == 1:
            temp[:-movement] = temp[movement:]
            temp[-movement:] = temp[-movement]
        else:
            temp[movement:] = temp[:-movement]
            temp[:movement] = temp[movement].reshape(-1, 1)
    else:
        movement = np.random.randint(0, transform)
        if movement != 0:
            if left == 1:
                temp[:-movement] = temp[movement:]
                temp[-movement:] = 0
            else:
                temp[movement:] = temp[:-movement]
                temp[:movement] = 0

    return temp


def extend_amp(data, amp_range_a, amp_range_b):
    amp = np.random.uniform(amp_range_a, amp_range_b)
    temp = data * amp
    return temp