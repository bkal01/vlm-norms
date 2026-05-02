from abc import ABC
import torch

class Intervention(ABC):
    def __init__(self):
        pass

    def reduce_norm(self, x):
        pass

class ScaledIntervention(Intervention):
    """
    scales activations down by alpha (e.g. 0.01)
    """
    def __init__(self, alpha):
        self.alpha = alpha

    def reduce_norm(self, x):
        return self.alpha * x


class RMSNormIntervention(Intervention):
    """
    scales activations down via a LayerNorm.
    """
    def reduce_norm(self, x):
        return torch.nn.functional.rms_norm(x, x.shape)
