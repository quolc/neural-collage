import chainer
import chainer.links as L
from chainer import functions as F
from gen_models.resblocks import SCBNBlock
from source.miscs.random_samples import sample_categorical, sample_continuous


class SCBNResNetGenerator(chainer.Chain):
    def __init__(self, ch=64, dim_z=128, bottom_width=4, activation=F.relu, n_classes=0, distribution="normal"):
        super(SCBNResNetGenerator, self).__init__()
        initializer = chainer.initializers.GlorotUniform()
        self.bottom_width = bottom_width
        self.activation = activation
        self.distribution = distribution
        self.dim_z = dim_z
        self.n_classes = n_classes
        with self.init_scope():
            self.l1 = L.Linear(dim_z, (bottom_width ** 2) * ch * 8, initialW=initializer)
            self.block2 = SCBNBlock(ch * 8, ch * 8, activation=activation, upsample=True, n_classes=n_classes)
            self.block3 = SCBNBlock(ch * 8, ch * 4, activation=activation, upsample=True, n_classes=n_classes)
            self.block4 = SCBNBlock(ch * 4, ch * 4, activation=activation, upsample=True, n_classes=n_classes)
            self.block5 = SCBNBlock(ch * 4, ch * 2, activation=activation, upsample=True, n_classes=n_classes)
            self.block6 = SCBNBlock(ch * 2, ch, activation=activation, upsample=True, n_classes=n_classes)
            self.b7 = L.BatchNormalization(ch)
            self.l7 = L.Convolution2D(ch, 3, ksize=3, stride=1, pad=1, initialW=initializer)

    def __call__(self, batchsize=64, z=None, **kwargs):
        if z is None:
            z = sample_continuous(self.dim_z, batchsize, distribution=self.distribution, xp=self.xp)
        h = z
        h = self.l1(h)
        h = F.reshape(h, (h.shape[0], -1, self.bottom_width, self.bottom_width))
        h = self.block2(h, **kwargs)
        h = self.block3(h, **kwargs)
        h = self.block4(h, **kwargs)
        h = self.block5(h, **kwargs)
        h = self.block6(h, **kwargs)
        h = self.b7(h)
        h = self.activation(h)
        h = F.tanh(self.l7(h))
        return h

    # n: data_id in the batch (Note: we expect batchsize=1 in inference time)
    # i: input_id (0=target, 1,2,...=reference)
    # j: channel
    # k: vertical
    # l: horizontal
    def blend_featuremap(self, hs, blend):
        return F.einsum('nijkl,kli->njkl', hs, blend)

    # class-map: weights[LAYER_ID][DATA_ID][H][W][CLASS_ID]
    # blending weights: blends[H][W][INPUT_ID]
    def spatial_interpolation(self, z=None, weights=None, zs=None, blends=None, **kwargs):
        if zs is None: # no feature blending
            h = z
            h = self.l1(h)
            h = F.reshape(h, (h.shape[0], -1, self.bottom_width, self.bottom_width))
            h = self.block2(h, weights1=weights[0], weights2=weights[1], **kwargs)
            h = self.block3(h, weights1=weights[2], weights2=weights[3], **kwargs)
            h = self.block4(h, weights1=weights[4], weights2=weights[5], **kwargs)
            h = self.block5(h, weights1=weights[6], weights2=weights[7], **kwargs)
            h = self.block6(h, weights1=weights[8], weights2=weights[9], **kwargs)
            h = self.b7(h)
            h = self.activation(h)
            h = F.tanh(self.l7(h))
            return h
        else: # feature blending
            hs = []
            # l1
            for z in zs:
                h = self.l1(z)
                h = F.reshape(h, (h.shape[0], -1, self.bottom_width, self.bottom_width))
                hs.append(h)
            hs.append(chainer.Variable(self.xp.zeros_like(hs[0]))) # dummy
            hs[-1] = self.blend_featuremap(F.stack(hs, axis=1), blends[0])
            # block2
            _hs = []
            for h in hs:
                h = self.block2(h, weights1=weights[0], weights2=weights[1], **kwargs)
                _hs.append(h)
            hs = _hs
            hs[-1] = self.blend_featuremap(F.stack(hs, axis=1), blends[1]) # blend
            # block3
            _hs = []
            for h in hs:
                h = self.block3(h, weights1=weights[2], weights2=weights[3], **kwargs)
                _hs.append(h)
            hs = _hs
            hs[-1] = self.blend_featuremap(F.stack(hs, axis=1), blends[2])
            # block4
            _hs = []
            for h in hs:
                h = self.block4(h, weights1=weights[4], weights2=weights[5], **kwargs)
                _hs.append(h)
            hs = _hs
            hs[-1] = self.blend_featuremap(F.stack(hs, axis=1), blends[3])
            # block5
            _hs = []
            for h in hs:
                h = self.block5(h, weights1=weights[6], weights2=weights[7], **kwargs)
                _hs.append(h)
            hs = _hs
            hs[-1] = self.blend_featuremap(F.stack(hs, axis=1), blends[4])
            # block6
            _hs = []
            for h in hs:
                h = self.block6(h, weights1=weights[8], weights2=weights[9], **kwargs)
                _hs.append(h)
            hs = _hs
            hs[-1] = self.blend_featuremap(F.stack(hs, axis=1), blends[5])

            # l7
            _hs = []
            for h in hs:
                h = self.b7(h)
                h = self.activation(h)
                h = F.tanh(self.l7(h))
                _hs.append(h)
            return _hs
