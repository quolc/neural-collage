import numpy as np
import sys

import chainer
import chainer.functions as F
from chainer import Variable
import chainercv


def reconstruction_loss(dis, recon, gt):
    with chainer.using_config('train', False):
        v1 = dis.feature_vector(recon)
        v2 = dis.feature_vector(gt)
    denom = F.sqrt(F.batch_l2_norm_squared(v1) * F.batch_l2_norm_squared(v2))

    xp = dis.xp
    sum = Variable(xp.array(0.0, dtype=xp.float32))
    for i in range(gt.shape[0]):
        sum += F.matmul(v1[i], v2[i], transb=True) / denom[i]
    cos_dist2 = - sum
    return cos_dist2


def pixel_loss(recon, gt):
    return F.mean_squared_error(recon, gt)


class UpdaterAuxAB(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        self.models = kwargs.pop('models')
        if 'input_size' in kwargs:
            self.input_size = kwargs.pop('input_size')
        else:
            self.input_size = None
        self.loss_func = reconstruction_loss
        super(UpdaterAuxAB, self).__init__(*args, **kwargs)

    def get_batch(self, xp):
        batch = self.get_iterator('main').next()
        batchsize = len(batch)
        x = []
        gt = []
        c = []
        for j in range(batchsize):
            x.append(np.asarray(batch[j][0]).astype("f"))
            gt.append(np.asarray(batch[j][1]).astype("f"))
            c.append(np.asarray(batch[j][2]).astype(np.int32))
        x = Variable(xp.asarray(x))
        gt = Variable(xp.asarray(gt))
        c = Variable(xp.asarray(c))
        return x, gt, c

    def update_core(self):
        gen = self.models['gen']
        dis = self.models['dis']
        enc = self.models['enc']
        gen_optimizer = self.get_optimizer('opt_gen')
        xp = enc.xp

        x, gt, c = self.get_batch(xp)
        if self.input_size is not None:
            _x = []
            for img in x.data.get():
                _x.append(chainercv.transforms.resize(img, (self.input_size, self.input_size)))
            x = Variable(xp.asarray(_x))

        # obtain initial z by encoder
        if enc.n_classes != 0:
            z = enc(x, y=c)
        else:
            z = enc(x)

        # fast updating
        with chainer.using_config('train', False):
            # out_noab : reconstruction results without auxiliary network
            outs, fast_losses, out_noab, zeta, z_prime = gen(batchsize=len(z), z=z, y=c, gt=gt)

        lmd_pixel = 0.05
        fast_losses.append(reconstruction_loss(dis, outs[-1], gt) + lmd_pixel * pixel_loss(outs[-1], gt))

        loss = 0
        weights = [20, 2.0, 1.0]

        for i in range(0, len(outs)):
            loss += fast_losses[i] * weights[i]

        # reconstruction loss as an autoencoder
        lmd_ae = 100

        # lmd_ae = 0
        ae_loss = F.mean_squared_error(z, z_prime) * z.shape[0]
        loss += lmd_ae * ae_loss

        # sparse regularization
        # lmd_sparse = 0.000
        # sparse_loss = lmd_sparse * F.sum(F.absolute(zeta))
        # loss += sparse_loss

        gen.cleargrads()

        # double backprop
        loss.backward()

        gen_optimizer.update()

        # reporting
        report = dict()
        for i, loss_i in enumerate(fast_losses):
            report["loss{}".format(i+1)] = loss_i
        report["loss_ae"] = ae_loss

        report["loss_noab"] = reconstruction_loss(dis, out_noab, gt) + lmd_pixel * pixel_loss(out_noab, gt)

        report["fast_alpha"] = gen.fast_alpha().data.mean()
        report["fast_benefit"] = report["loss{}".format(len(fast_losses))] - report["loss1"]
        report["min_slope"] = F.min(gen.preluW())
        report["max_slope"] = F.max(gen.preluW())
        report["min_slope_middle"] = F.min(gen.preluMiddleW())
        report["max_slope_middle"] = F.max(gen.preluMiddleW())

        chainer.reporter.report(report)

        if not gen.learned_lr:
            gen._fast_alpha = min(gen.limit_fast_alpha,
                    gen.initial_fast_alpha + gen.step_fast_alpha * self.iteration)
