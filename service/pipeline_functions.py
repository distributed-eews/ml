from abc import ABC, abstractmethod
from typing import List

import numpy as np


class PipelineFunction(ABC):
    def __init__(self, x: np.ndarray =None):
        pass

    @abstractmethod
    def compute(self, x: np.ndarray) -> np.ndarray:
        pass


class Slope(PipelineFunction):
    def __init__(self, sampling_rate: float, x: np.ndarray =None):
        super().__init__()
        self.prev_val = 0
        if x:
            self.prev_val = x[0]
        self.sampling_rate = sampling_rate

    def compute(self, x: np.ndarray) -> np.ndarray:
        first_slope = np.array([(x[0] - self.prev_val) * self.sampling_rate])
        rest_slope = (x[1:] - x[:-1]) * self.sampling_rate
        return np.concatenate([first_slope, rest_slope], axis=0)


class ExponentialSmoothing(PipelineFunction):
    def __init__(self, alpha: float, avg_until: int = 0, x: np.ndarray =None):
        super().__init__()

        assert 0.0 <= alpha <= 1.0, "Alpha must be in range [0.0, 1.0]"

        self.alpha = alpha
        self.alpha_ = 1 - self.alpha
        self.avg_until = avg_until

        self.prev_val = 0
        if avg_until > 0:
            self.prev_val = np.mean(x[:avg_until], axis=0)

    def compute(self, x: np.ndarray) -> np.ndarray:
        ema_vals = []
        prev_ema_val = self.prev_val
        for val in x:
            curr_ema_val = prev_ema_val * self.alpha + val * self.alpha_
            prev_ema_val = curr_ema_val
            ema_vals.append(curr_ema_val)
        return np.array(ema_vals)


class MultiExponentialSmoothing(PipelineFunction):
    def __init__(self, alpha: List[float], avg_until: int = 0, x: np.ndarray =None):
        super().__init__()

        for a in alpha:
            assert 0.0 <= a <= 1.0, "Alpha must be in range [0.0, 1.0]"

        self.alpha = np.array(alpha)
        self.alpha_ = 1 - self.alpha
        self.avg_until = avg_until

        self.prev_val = 0
        if avg_until > 0:
            self.prev_val = np.mean(x[:avg_until], axis=0)

    def compute(self, x: np.ndarray) -> np.ndarray:
        ema_vals = []
        prev_ema_val = self.prev_val
        for val in x:
            curr_ema_val = prev_ema_val * self.alpha + val * self.alpha_
            prev_ema_val = curr_ema_val
            ema_vals.append(curr_ema_val)
        return np.array(ema_vals)


class Square(PipelineFunction):
    def __init__(self, x: np.ndarray =None):
        super().__init__()

    def compute(self, x: np.ndarray) -> np.ndarray:
        return x*x


class AddChannels(PipelineFunction):
    def __init__(self, x: np.ndarray =None):
        super().__init__()

    def compute(self, x: np.ndarray) -> np.ndarray:
        return np.sum(x, axis=-1, keepdims=True)


class PairwiseRatio(PipelineFunction):
    def __init__(self, alpha: List[float], x: np.ndarray =None):
        # Generate Order
        idx_area = [(i, -np.log(1 - a) * (a) / (1 - a)) for i, a in enumerate(alpha)]
        triplets = [(j, i, idx_area[i][1] / idx_area[j][1]) for i in range(len(alpha)) for j in range(len(alpha)) if
                    i > j]
        triplets = sorted(triplets, key=lambda t: t[2])
        self.order = [(t[0], t[1]) for t in triplets]

    def compute(self, x: np.ndarray) -> np.ndarray:
        new_x = []
        for i, j in self.order:
            new_x.append((x[:,i]+1)/(x[:,j]+1))
        return np.array(new_x).T