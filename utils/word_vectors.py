import numpy as np


def cos_similarity(x):
    """Compute cosine similarity matrix in CPU & memory sensitive way

    Args:
        x (np.ndarray): embeddings, 2D array, embeddings are in rows

    Returns:
        np.ndarray: cosine similarity matrix

    """
    assert x.ndim == 2, f'x has {x.ndim} dimensions, it must be matrix'
    x = x / (np.sqrt(np.sum(np.square(x), axis=1, keepdims=True)) + 1.0e-32)
    # assert np.allclose(np.ones_like(x[:, 0]), np.sum(np.square(x), axis=1))
    max_n_elm = 200000000
    step = max(max_n_elm // (x.shape[0] * x.shape[0]), 1)
    retval = np.zeros(shape=(x.shape[0], x.shape[0]), dtype=np.float64)
    x0 = np.expand_dims(x, 0)
    x1 = np.expand_dims(x, 1)
    for i in range(0, x.shape[1], step):
        product = x0[:, :, i:i + step] * x1[:, :, i:i + step]
        retval += np.sum(product, axis=2, keepdims=False)
    assert np.all(retval >= -1.0001), retval
    assert np.all(retval <= 1.0001), retval
    return retval
