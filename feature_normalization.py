import numpy as np

def feature_normalization(feature_vector):
    min = min(feature_vector)
    max = max(feature_vector)
    range = max - min

