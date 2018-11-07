import collections
import numpy as np


def num_non_integer_values(data: np.ndarray, idx: int) -> int:
    non_integer_idx = [np.floor(val) == np.ceil(val) for _, val in enumerate(data[:, idx])]
    if not all(non_integer_idx):
        counter = collections.Counter(non_integer_idx)
        return counter[False]
    return 0
