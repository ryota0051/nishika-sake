import numpy as np

from src.utils import l2norm_numpy


def emsamble_features(
    features_and_weights: list[tuple[np.ndarray, float]]
) -> np.ndarray:
    concat_feat = np.hstack(
        [
            l2norm_numpy(features) * weights
            for (features, weights) in features_and_weights
        ]
    )
    return l2norm_numpy(concat_feat)
