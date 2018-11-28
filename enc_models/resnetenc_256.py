import chainer
import chainer.links as L
from chainer import functions as F
from enc_models.resblocks import Block


# encoder architecture
class ResNetEncoder(chainer.Chain):
    def __init__(self, ch=64, dim_z=128, bottom_width=4, n_classes=0, activation=F.relu):
        super(ResNetEncoder, self).__init__()
        initializer = chainer.initializers.GlorotUniform()
        self.ch = ch
        self.dim_z = dim_z
        self.bottom_width = bottom_width
        self.n_classes = n_classes
        self.activation = activation
        with self.init_scope():
            self.block1 = L.Convolution2D(3, ch, ksize=3, stride=1, pad=1, initialW=initializer)
            self.block2 = Block(ch, ch * 2, activation=activation, downsample=True, n_classes=n_classes)
            self.block3 = Block(ch * 2, ch * 4, activation=activation, downsample=True, n_classes=n_classes)
            self.block4 = Block(ch * 4, ch * 8, activation=activation, downsample=True, n_classes=n_classes)
            self.block5 = Block(ch * 8, ch * 8, activation=activation, downsample=True, n_classes=n_classes)
            self.block6 = Block(ch * 8, ch * 16, activation=activation, downsample=True, n_classes=n_classes)
            self.block7 = Block(ch * 16, ch * 16, activation=activation, downsample=True, n_classes=n_classes)
            self.l8_enc = L.Linear((bottom_width ** 2) * ch * 16, dim_z, initialW=initializer)
            self.prelu_out = L.PReLU((dim_z,), init=0.1)

    def __call__(self, x, y=None, **kwargs):
        h = x
        h = self.block1(h)
        h = self.block2(h, y, **kwargs)
        h = self.block3(h, y, **kwargs)
        h = self.block4(h, y, **kwargs)
        h = self.block5(h, y, **kwargs)
        h = self.block6(h, y, **kwargs)
        h = self.block7(h, y, **kwargs)
        h = F.reshape(h, (h.shape[0], (self.bottom_width ** 2) * self.ch * 16))
        h = self.prelu_out(self.l8_enc(h))
        return h
