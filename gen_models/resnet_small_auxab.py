import chainer
import chainer.links as L
from chainer import functions as F
from gen_models.resblocks import Block
from source.miscs.random_samples import sample_categorical, sample_continuous

import numpy as np


class MyAdaGrad:
    def __init__(self, var, xp, eps=1e-8, lr=0.001):
        self.r = xp.ones_like(var) * eps
        self.lr = lr

    def calc_update(self, grads):
        self.r = self.r + grads * grads
        eta = F.broadcast_to(self.lr, grads.shape) / F.sqrt(self.r)
        return - eta * grads


class ResNetAuxABGenerator(chainer.Chain):
    def __init__(self, ch=64, dim_z=128, bottom_width=4, activation=F.relu, n_classes=0, distribution="normal",
                 dim_a=1000, dim_b=1000, dim_zeta=10000, T=1, learned_lr=False,
                 initial_fast_alpha=0.001, limit_fast_alpha=0.01, step_fast_alpha=0.000001):
        super(ResNetAuxABGenerator, self).__init__()
        initializer = chainer.initializers.GlorotUniform()
        self.ch = ch
        self.bottom_width = bottom_width
        self.activation = activation
        self.distribution = distribution
        self.dim_z = dim_z
        self.dim_zeta = dim_zeta
        self.n_classes = n_classes

        self.T = T
        self.learned_lr = learned_lr
        self.initial_fast_alpha = initial_fast_alpha
        self.limit_fast_alpha = limit_fast_alpha
        self.step_fast_alpha = step_fast_alpha
        self.fast_loss = None

        with self.init_scope():
            # parameters to be slow-updated
            self.lA1 = L.Linear(dim_z, dim_a, initialW=initializer)
            self.lA2 = L.Linear(dim_a, dim_zeta, initialW=initializer)
            self.lB1 = L.Linear(dim_zeta, dim_b, initialW=initializer)
            self.lB2 = L.Linear(dim_b, dim_z, initialW=initializer)
            self.preluW = L.Parameter(np.ones((dim_z,), dtype=np.float32) * 0.25)
            self.preluMiddleW = L.Parameter(np.ones((dim_zeta,), dtype=np.float32) * 0.25)

            # inherited from ResNetGenerator
            self.l1 = L.Linear(dim_z, (bottom_width ** 2) * ch * 8, initialW=initializer)
            self.block2 = Block(ch * 8, ch * 8, activation=activation, upsample=True, n_classes=n_classes)
            self.block3 = Block(ch * 8, ch * 4, activation=activation, upsample=True, n_classes=n_classes)
            self.block4 = Block(ch * 4, ch * 4, activation=activation, upsample=True, n_classes=n_classes)
            self.block5 = Block(ch * 4, ch * 2, activation=activation, upsample=True, n_classes=n_classes)
            self.block6 = Block(ch * 2, ch, activation=activation, upsample=True, n_classes=n_classes)
            self.b7 = L.BatchNormalization(ch)
            self.l7 = L.Convolution2D(ch, 3, ksize=3, stride=1, pad=1, initialW=initializer)

            if self.learned_lr:
                self._fast_alpha = chainer.links.Parameter(self.xp.ones((dim_zeta,), dtype=self.xp.float32) * initial_fast_alpha)
            else:
                self._fast_alpha = initial_fast_alpha

    def set_fast_loss(self, fast_loss):
        self.fast_loss = fast_loss

    def fast_alpha(self):
        if self.learned_lr:
            return self._fast_alpha()
        else:
            return chainer.Variable(self.xp.array(self._fast_alpha, dtype=self.xp.float32))

    # PReLU implementation
    def prelu(self, inp, parameter):
        x = F.reshape(inp, (inp.shape[0], 1, inp.shape[1]))
        zeros = self.xp.zeros_like(x.data)
        c = F.transpose(F.concat((x, zeros), axis=1), (0, 2, 1))
        return F.max(c, axis=2) + F.broadcast_to(parameter, inp.shape) * F.min(c, axis=2)

    def forward(self, z=None, y=None, zeta=None, noAB=False, return_zs=False, **kwargs):
        if noAB:
            h = z
        else:
            h = z
            if zeta is None:
                h = self.activation(self.lA1(h))
                zeta = self.prelu(self.lA2(h), self.preluMiddleW())
            h = self.activation(self.lB1(zeta))
            z_prime = self.prelu(self.lB2(h), self.preluW())
            h = z_prime

        h = self.l1(h)
        h = F.reshape(h, (h.shape[0], -1, self.bottom_width, self.bottom_width))

        h = self.block2(h, y, **kwargs)
        h = self.block3(h, y, **kwargs)
        h = self.block4(h, y, **kwargs)
        h = self.block5(h, y, **kwargs)
        h = self.block6(h, y, **kwargs)

        h = self.b7(h)
        h = self.activation(h)

        h = self.l7(h)
        h = F.tanh(h)
        if return_zs:
            return h, z, zeta, z_prime
        else:
            return h

    # calculate zeta with A
    def forward_A(self, z):
        h = z
        h = self.activation(self.lA1(h))
        zeta = self.prelu(self.lA2(h), self.preluMiddleW())
        return zeta

    def __call__(self, batchsize=64, z=None, y=None, gt=None, **kwargs):
        outs = []
        fast_losses = []

        if z is None:
            z = sample_continuous(self.dim_z, batchsize, distribution=self.distribution, xp=self.xp)
        if y is None:
            y = sample_categorical(self.n_classes, batchsize, distribution="uniform",
                                   xp=self.xp) if self.n_classes > 0 else None
        if (y is not None) and z.shape[0] != y.shape[0]:
            raise Exception('z.shape[0] != y.shape[0], z.shape[0]={}, y.shape[0]={}'.format(z.shape[0], y.shape[0]))

        # forward calculation without auxiliary network
        out_noab = self.forward(z=z, y=y, noAB=True, **kwargs)

        out, z, zeta, z_recon = self.forward(batchsize, z=z, y=y, return_zs=True, **kwargs)
        outs.append(out)

        # beta1=0, beta2=0.9 <-> initial_t = 100
        optimizer = MyAdaGrad(zeta, self.xp, lr=self.fast_alpha())

        for _ in range(self.T):
            loss = F.sum(self.fast_loss(out, gt))
            fast_losses.append(loss)

            grads = chainer.grad([loss], [zeta], enable_double_backprop=True)[0]
            # use learned learning rate
            # z2 += - F.broadcast_to(self.lr(), grads[0].shape) * grads[0]
            zeta += optimizer.calc_update(grads)

            # forward run with z2 supply
            out = self.forward(z=z, y=y, zeta=zeta)
            outs.append(out)

        return outs, fast_losses, out_noab, zeta, z_recon
