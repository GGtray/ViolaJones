from typing import List
from typing import Tuple
import numpy as np


class WeakClassifier:
    def __init__(self, threshold, polarity, feature_idx, alpha):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.polarity = polarity
        self.alpha = alpha

    def get_vote(self, x) -> int:
        return 1 if (self.polarity * x[self.feature_idx]) < (self.polarity * self.threshold) else -1


# def build_StrongClassifier():
#     return


def build_running_sums(labels: np.ndarray, weights: np.ndarray) -> Tuple[float, float, List[float], List[float]]:
    s_minus, s_plus = 0., 0.
    t_minus, t_plus = 0., 0.
    s_minuses, s_pluses = [], []

    for y, w in zip(labels, weights):
        if y < .5:
            s_minus += w
            t_minus += w
        else:
            s_plus += w
            t_plus += w
        s_minuses.append(s_minus)
        s_pluses.append(s_plus)
    return t_minus, t_plus, s_minuses, s_pluses


def find_best_threshold(samples: np.ndarray, t_minus: float, t_plus: float, s_minuses: List[float],
                        s_pluses: List[float]):
    min_error = float('inf')
    best_threshold, polarity = 0, 0
    for sample_value, s_m, s_p in zip(samples, s_minuses, s_pluses):
        error_1 = s_p + (t_minus - s_m)
        error_2 = s_m + (t_plus - s_p)
        if error_1 < min_error:
            min_error = error_1
            best_threshold = sample_value
            polarity = -1
        elif error_2 < min_error:
            min_error = error_2
            best_threshold = sample_value
            polarity = 1
    return [best_threshold, polarity, min_error]


def find_best_threshold_false_positive(samples: np.ndarray, t_minus: float, t_plus: float, s_minuses: List[float],
                                        s_pluses: List[float]):
    min_false_positive = float('inf')
    best_threshold = samples[np.argmin(samples)]
    polarity = 0
    for sample_value, s_m, s_p in zip(samples, s_minuses, s_pluses):
        error_1 = s_p
        error_2 = s_m
        if error_1 < min_false_positive:
            min_false_positive = s_p
            best_threshold = sample_value
            polarity = -1
        elif error_2 < min_false_positive:
            min_false_positive = s_m
            best_threshold = sample_value
            polarity = 1
    return [best_threshold, polarity, min_false_positive]


def find_best_threshold_false_negative(samples: np.ndarray, t_minus: float, t_plus: float, s_minuses: List[float],
                                        s_pluses: List[float]):
    min_false_negative = float('inf')
    best_threshold = samples[np.argmin(samples)]
    for sample_value, s_m, s_p in zip(samples, s_minuses, s_pluses):
        error_1 = t_minus - s_m
        error_2 = t_plus - s_p
        if error_1 < min_false_negative:
            min_false_negative = error_1
            best_threshold = sample_value
            polarity = -1
        elif error_2 < min_false_negative:
            best_threshold = sample_value
            min_z = z
            polarity = 1
    return [best_threshold, polarity, min_false_negative]
