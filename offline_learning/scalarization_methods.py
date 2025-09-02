from abc import abstractmethod, ABC

import torch as th


class ScalarizationFunction(ABC):
    def __init__(self, num_objs, weights=None):
        self.num_objs = num_objs
        self._device = None
        if weights is not None:
            self.weights = th.Tensor(weights)
        else:
            self.weights = None

    def update_weights(self, weights):
        if weights is not None:
            self.weights = th.Tensor(weights)

    def to_device(self, device: th.device):
        if self._device != device:
            if self.weights is not None:
                self.weights = self.weights.to(device)
            self._device = device

    @abstractmethod
    def evaluate(self, objs):
        pass


class WeightedSumScalarization(ScalarizationFunction):
    def __init__(self, num_objs, weights=None):
        super(WeightedSumScalarization, self).__init__(num_objs, weights)

    def update_z(self, z):
        pass

    def evaluate(self, objs):
        if not isinstance(objs, th.Tensor):
            objs = th.Tensor(objs)
        return (objs * self.weights).sum(axis=-1)
