import numpy as np

import chainer
import chainer.functions as F
from chainer import Variable
import chainercv


def reconstruction_loss(dis, recon, gt):
    with chainer.using_config('train', False):
        v1 = dis.feature_vector(recon)
        v2 = dis.feature_vector(gt)
    denom = F.sqrt(F.batch_l2_norm_squared(v1) * F.batch_l2_norm_squared(v2))
    return -F.sum(F.reshape(F.batch_matmul(v1, v2, transa=True), (v1.shape[0],)) / denom)


class UpdaterEnc(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        self.models = kwargs.pop('models')
        if 'input_size' in kwargs:
            self.input_size = kwargs.pop('input_size')
        else:
            self.input_size = None
        self.loss_func = reconstruction_loss
        super(UpdaterEnc, self).__init__(*args, **kwargs)

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
        enc_optimizer = self.get_optimizer('opt_enc')
        xp = enc.xp

        # fetch batch
        x, gt, c = self.get_batch(xp)
        if self.input_size is not None:
            _x = []
            for img in x.data.get():
                _x.append(chainercv.transforms.resize(img, (self.input_size, self.input_size)))
            x = Variable(xp.asarray(_x))
        z = enc(x, y=c)

        with chainer.using_config('train', False):
            recon = gen(batchsize=len(z), z=z, y=c)

        loss = reconstruction_loss(dis, recon, gt)
        enc.cleargrads()
        loss.backward()
        enc_optimizer.update()
        chainer.reporter.report({'loss': loss})
        chainer.reporter.report({'min_slope': xp.min(enc.prelu_out.W.data)})
        chainer.reporter.report({'max_slope': xp.max(enc.prelu_out.W.data)})
        chainer.reporter.report({'min_z': xp.min(z.data)})
        chainer.reporter.report({'max_z': xp.max(z.data)})
