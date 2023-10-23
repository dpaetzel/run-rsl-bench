import numpy as np


def clamp_transform(lowers, uppers, X_min, X_max, transformer_X=None):
    lowers = np.clip(lowers, X_min, X_max)
    uppers = np.clip(uppers, X_min, X_max)

    if transformer_X is not None:
        lowers = transformer_X.inverse_transform(lowers)
        uppers = transformer_X.inverse_transform(uppers)
    else:
        lowers = np.array(lowers)
        uppers = np.array(uppers)

    return lowers, uppers
