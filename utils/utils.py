import numpy as np
import re


def to_tensor(x, **kwargs):

    return x.transpose(2, 0, 1).astype('float32')


def tryint(s):
    try:
        return int(s)
    except:
        return s



