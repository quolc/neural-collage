import numpy

from chainer import initializers
from chainer.utils import argument
from chainer.links import EmbedID
import chainer.functions as F
from source.links.conditional_batch_normalization import ConditionalBatchNormalization


class SpatialCategoricalConditionalBatchNormalization(ConditionalBatchNormalization):

    def __init__(self, size, n_cat, decay=0.9, eps=2e-5, dtype=numpy.float32,
                 initial_gamma=None, initial_beta=None):
        super(SpatialCategoricalConditionalBatchNormalization, self).__init__(
            size=size, n_cat=n_cat, decay=decay, eps=eps, dtype=dtype)

        with self.init_scope():
            if initial_gamma is None:
                initial_gamma = 1
            initial_gamma = initializers._get_initializer(initial_gamma)
            initial_gamma.dtype = dtype
            self.gammas = EmbedID(n_cat, size, initialW=initial_gamma)
            if initial_beta is None:
                initial_beta = 0
            initial_beta = initializers._get_initializer(initial_beta)
            initial_beta.dtype = dtype
            self.betas = EmbedID(n_cat, size, initialW=initial_beta)

    def __call__(self, x, **kwargs):
        # shapes:
        # self.gammas.W : (n_classes, n_ch)
        # weights : (n_batch, x, y, n_classes)

        weights, = argument.parse_kwargs(kwargs, ('weights', None))
        _weights = F.reshape(weights, (weights.shape[0] * weights.shape[1] * weights.shape[2], weights.shape[3]))

        gamma_c = F.reshape(F.matmul(_weights, self.gammas.W),
                            (weights.shape[0], weights.shape[1], weights.shape[2], self.gammas.W.shape[1]))
        gamma_c = F.transpose(gamma_c, (0, 3, 1, 2))
        beta_c = F.reshape(F.matmul(_weights, self.betas.W),
                           (weights.shape[0], weights.shape[1], weights.shape[2], self.gammas.W.shape[1]))
        beta_c = F.transpose(beta_c, (0, 3, 1, 2))

        return super(SpatialCategoricalConditionalBatchNormalization, self).__call__(x, gamma_c, beta_c, **kwargs)


def start_finetuning(self):
    self.N = 0
